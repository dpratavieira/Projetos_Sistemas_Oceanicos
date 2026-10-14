# -*- coding: utf-8 -*-
"""
wamit_io.py - leitura dos arquivos de entrada e saída do WAMIT (v7, ULEN = 1) para 6 graus de liberdade.

Todas as grandezas são devolvidas em unidades SI (kg, N, m, s, rad) como xarray, com os mesmos rótulos
de graus de liberdade usados pelo Capytaine: Surge, Sway, Heave, Roll, Pitch, Yaw.

Arquivos lidos
  poten.pot   frequências (IPERIN = 2 -> rad/s) e direções de onda [graus]
  ship.frc    massa, amortecimento e rigidez externos (unidades de tonelada, coerentes com RHO = 1,025)
  force.hst   rigidez hidrostática adimensional
  force.1     massa adicional e amortecimento potencial adimensionais
  force.2/3   força de excitação (Haskind / difração) adimensional
  force.4     RAO (resposta acoplada em 6 graus de liberdade)

Adimensionalização do WAMIT com L = ULEN = 1 m (os fatores de L desaparecem):
  A_ij = Ā ρ      B_ij = B̄ ρ ω      C_ij = C̄ ρ g      X_i = X̄ ρ g A      ξ_i = ξ̄ A  (rad/m nas rotações)
"""
from pathlib import Path
import numpy as np
import xarray as xr

DOFS = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']


def ler_poten(pasta):
    """Lê frequências [rad/s] e direções [graus] do poten.pot (formato com IPERIN = 2)."""
    tok = Path(pasta, 'poten.pot').read_text().split('\n')
    linhas = [l.split() for l in tok if l.strip()]
    # linha 0: título | 1: HBOT | 2: IRAD IDIFF | 3: NPER | 4: PER(...) | 5: NBETA | 6: BETA(...)
    nper = int(linhas[3][0])
    per = np.array(linhas[4][:nper], float)
    nbeta = int(linhas[5][0])
    beta = np.array(linhas[6][:nbeta], float)
    return per, beta


def ler_frc(pasta, arquivo='ship.frc'):
    """Lê as matrizes externas do arquivo .frc (formato alternativo 1) e devolve em SI."""
    tok = Path(pasta, arquivo).read_text().split()
    i = 0
    titulo = tok[i]; i += 1
    i += 9                      # IOPTN(1:9)
    rho_t = float(tok[i]); i += 1          # densidade em t/m3
    xcg = np.array(tok[i:i + 3], float); i += 3
    mats = {}
    for nome in ('massa', 'amortecimento', 'rigidez'):
        flag = int(tok[i]); i += 1
        if flag == 1:
            mats[nome] = np.array(tok[i:i + 36], float).reshape(6, 6) * 1e3   # t -> kg, kN -> N
            i += 36
        else:
            mats[nome] = np.zeros((6, 6))
    return dict(rho=rho_t * 1e3, xcg=xcg, **mats)


def ler_hst(pasta, rho=1025.0, g=9.80665):
    d = np.loadtxt(Path(pasta, 'force.hst'))
    C = np.zeros((6, 6))
    for i, j, c in d:
        C[int(i) - 1, int(j) - 1] = c
    return xr.DataArray(C * rho * g, dims=['influenced_dof', 'radiating_dof'],
                        coords={'influenced_dof': DOFS, 'radiating_dof': DOFS}, name='hydrostatic_stiffness')


def ler_force1(pasta, rho=1025.0):
    """Massa adicional A(ω) [kg, kg·m, kg·m²] e amortecimento B(ω) em SI, só ω finitos."""
    d = np.loadtxt(Path(pasta, 'force.1'))
    d = d[d[:, 0] > 0]                                   # descarta PER = 0 e -1, se existirem
    T = np.unique(d[:, 0])[::-1]                          # ordem crescente de ω
    w = 2 * np.pi / T
    A = np.zeros((len(T), 6, 6)); B = np.zeros_like(A)
    for per, i, j, a, b in d:
        k = np.where(np.isclose(T, per))[0][0]
        A[k, int(i) - 1, int(j) - 1] = a * rho
        B[k, int(i) - 1, int(j) - 1] = b * rho * w[k]
    c = {'omega': w, 'influenced_dof': DOFS, 'radiating_dof': DOFS}
    dims = ['omega', 'influenced_dof', 'radiating_dof']
    return xr.Dataset({'added_mass': (dims, A), 'radiation_damping': (dims, B)}, coords=c)


def _ler_vetor(pasta, arquivo, fator):
    d = np.loadtxt(Path(pasta, arquivo))
    d = d[d[:, 0] > 0]
    T = np.unique(d[:, 0])[::-1]
    betas = np.unique(d[:, 1])
    X = np.zeros((len(T), len(betas), 6), complex)
    for per, beta, i, mod, fase, re, im in d:
        k = np.where(np.isclose(T, per))[0][0]
        b = np.where(np.isclose(betas, beta))[0][0]
        X[k, b, int(i) - 1] = (re + 1j * im) * fator
    return xr.DataArray(X, dims=['omega', 'wave_direction', 'influenced_dof'],
                        coords={'omega': 2 * np.pi / T, 'wave_direction': np.radians(betas), 'influenced_dof': DOFS})


def ler_excitacao(pasta, rho=1025.0, g=9.80665, arquivo='force.2'):
    """Força de excitação por metro de amplitude [N/m e N·m/m]. force.2 = Haskind, force.3 = difração."""
    return _ler_vetor(pasta, arquivo, rho * g).rename('excitation_force')


def ler_rao(pasta):
    """RAO do WAMIT (force.4): m/m nas translações e rad/m nas rotações (ULEN = 1)."""
    return _ler_vetor(pasta, 'force.4', 1.0).rename('rao')


if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else '../Exercicio6_codigos/WAMIT-FPSO'
    per, beta = ler_poten(p)
    frc = ler_frc(p)
    print(f'{len(per)} frequências ({per.min()} a {per.max()} rad/s), {len(beta)} direções')
    print('massa [kg]:', np.diag(frc['massa']))
    print('amortecimento externo [SI]:', np.diag(frc['amortecimento']))
    print('C hidrostática [SI]:', np.diag(ler_hst(p).values))
    print(ler_force1(p).added_mass.sel(influenced_dof='Heave', radiating_dof='Heave').values[:5])
