# -*- coding: utf-8 -*-
"""
PNV3425 - Projeto de Sistemas Oceânicos
Dashboard interativo do modelo analítico de heave (Exercício 6): forças de onda e RAO.

Modelo (slides da aula): casco prismático L x B x T, onda de proa, águas profundas,
forças por faixas 2D integradas no comprimento, massa adicional Ma2D = Ca ρπ/2 (B/2)²,
sistema de 1 grau de liberdade com fator de amortecimento φ.

Requisitos:  pip install "streamlit>=1.50" plotly numpy pandas capytaine xarray
Uso:         streamlit run dashboard_rao_heave.py
             (abre no navegador em http://localhost:8501)

A comparação com o Capytaine (BEM) usa ex6_capytaine_tabela.csv para o FPSO original (250 x 40 x 12 m)
e, ao mudar as dimensões, permite rodar o Capytaine de novo pelo botão da página (RAO só com amortecimento
de radiação, como em exercicio6_capytaine.py).
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PASTA = os.path.dirname(os.path.abspath(__file__))
G = 9.81
REF = dict(L=250.0, B=40.0, T=12.0, rho=1.025, Ca=1.3, phi=0.05)   # dados do Exercício 6

COR = dict(fk='#1F6FA8', ma='#E8702A', tot='#0B2E40', rao='#1F6FA8', ref='#9AA5AB', wn='#E8702A', cap='#7A4EAB')


# ------------------------------------------------------------------ modelo
def modelo(L, B, T, rho, Ca, phi, w):
    """Propriedades e curvas do modelo analítico (unidades: t, m, s, kN; A = 1 m)."""
    vol = L * B * T
    M = rho * vol
    Ma2D = Ca * rho * np.pi / 2 * (B / 2) ** 2
    Ma = Ma2D * L
    c = rho * G * L * B
    wn = np.sqrt(c / (M + Ma))
    k = w ** 2 / G
    att = np.exp(-k * T)
    s = np.sin(k * L / 2)
    F_FK = 2 * rho * G ** 2 * B / w ** 2 * att * s
    F_MA = -2 * Ma2D * G * att * s
    F = F_FK + F_MA
    r = w / wn
    rao = np.abs(F) / c / np.sqrt((1 - r ** 2) ** 2 + (2 * phi * r) ** 2)
    props = dict(vol=vol, M=M, Ma2D=Ma2D, Ma=Ma, c=c, wn=wn, Tn=2 * np.pi / wn,
                 b=2 * phi * wn * (M + Ma),
                 w_lam=np.sqrt(2 * np.pi * G / L),                  # λ = L -> sen(kL/2) = 0
                 w_eq=np.sqrt(rho * G * B / Ma2D))                  # F_FK + F_MA = 0
    return props, dict(F_FK=F_FK, F_MA=F_MA, F=F, rao=rao)


def rodar_capytaine(L, B, T, rho, dx, w_max, progresso):
    """Roda o Capytaine (heave, onda de proa, águas profundas) para o casco L x B x T.
    rho em t/m³; progresso(i, n) é chamado a cada frequência. Retorna ω, |F_exc| [kN] e RAO (só radiação)."""
    import capytaine as cpt
    import xarray as xr
    rho_si = rho * 1e3
    nx, ny, nz = (int(np.ceil(v / dx)) for v in (L, B, T))
    mesh = cpt.mesh_parallelepiped(size=(L, B, T), center=(0, 0, -T / 2), resolution=(nx, ny, nz),
                                   missing_sides={'top'}, name='FPSO_caixa')
    # tampa (lid) explícita em z=0 com a mesma resolução do casco; mesh.generate_lid()
    # devolve tampa vazia/incompleta em algumas malhas
    lid = cpt.mesh_rectangle(size=(L, B), center=(0, 0, 0), resolution=(nx, ny), normal=(0, 0, -1), name='lid')
    body = cpt.FloatingBody(mesh=mesh, lid_mesh=lid,
                            dofs=cpt.rigid_body_dofs(only=['Heave'], rotation_center=(0, 0, 0)),
                            center_of_mass=(0, 0, 0))
    M = rho_si * mesh.disp_volume
    C33 = rho_si * G * mesh.waterplane_area
    wc = np.unique(np.concatenate([np.linspace(2 * np.pi / 35, 0.40, 8),
                                   np.linspace(0.40, 0.80, 33 if dx <= 3 else 11),
                                   np.linspace(0.80, w_max, 13 if dx <= 3 else 5)]))
    solver = cpt.BEMSolver()
    A33, B33, F3 = [], [], []
    # frequências em lotes: cada lote roda em paralelo (n_jobs=-1 = todos os processadores)
    # e o progresso é atualizado a cada lote
    lote = max(1, os.cpu_count() or 1)
    for i in range(0, len(wc), lote):
        progresso(i, len(wc))
        ds = solver.fill_dataset(xr.Dataset(coords={'omega': wc[i:i + lote], 'wave_direction': [0.0],
                                                    'radiating_dof': list(body.dofs), 'water_depth': [np.inf],
                                                    'rho': rho_si, 'g': G}), body, n_jobs=-1)
        A33.extend(ds.added_mass.sel(radiating_dof='Heave', influenced_dof='Heave').values)
        B33.extend(ds.radiation_damping.sel(radiating_dof='Heave', influenced_dof='Heave').values)
        F3.extend(ds.excitation_force.sel(influenced_dof='Heave', wave_direction=0.0).values)
    progresso(len(wc), len(wc))
    A33, B33, F3 = np.array(A33), np.array(B33), np.array(F3)
    rao = np.abs(F3 / (C33 - wc ** 2 * (M + A33) - 1j * wc * B33))
    return dict(w=wc, F=np.abs(F3) / 1e3, rao=rao)


@st.cache_data
def capytaine_referencia():
    """Resultado já calculado do Capytaine para o FPSO do exercício (ex6_capytaine_tabela.csv)."""
    f = os.path.join(PASTA, 'ex6_capytaine_tabela.csv')
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, sep=';')
    return dict(w=d['omega_rad_s'].values, F=d['|F_exc|_kN'].values, rao=d['RAO_radiacao'].values)


def layout(fig, xt, yt, h=430):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), hovermode='x unified',
                      xaxis_title=xt, yaxis_title=yt, template='plotly_white',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    fig.update_xaxes(showgrid=True, gridcolor='#ECEFF1')
    fig.update_yaxes(showgrid=True, gridcolor='#ECEFF1', zeroline=True, zerolinecolor='#B0BEC5')
    return fig


# ------------------------------------------------------------------ interface
st.set_page_config(page_title='RAO de heave – FPSO', page_icon='🌊', layout='wide')
st.title('RAO de heave de um FPSO – modelo analítico')
st.caption('PNV3425 · Exercício 6 · casco prismático, onda de proa (0°), águas profundas, amplitude de onda A = 1 m')

with st.sidebar:
    st.header('Casco')
    L = st.slider('Comprimento L [m]', 50.0, 400.0, REF['L'], 5.0)
    B = st.slider('Boca B [m]', 10.0, 80.0, REF['B'], 1.0)
    T = st.slider('Calado T [m]', 2.0, 30.0, REF['T'], 0.5)
    rho = st.number_input('Densidade da água ρ [t/m³]', 1.000, 1.050, REF['rho'], 0.005, format='%.3f')
    st.header('Hidrodinâmica')
    Ca = st.slider('Coeficiente de massa adicional Ca', 0.5, 2.0, REF['Ca'], 0.05)
    phi = st.slider('Fator de amortecimento φ [% do crítico]', 1.0, 30.0, REF['phi'] * 100, 0.5) / 100
    st.header('Gráficos')
    w_max = st.slider('Frequência máxima ω [rad/s]', 0.5, 3.0, 1.6, 0.1)
    eixo_T = st.toggle('Gráficos em função do período da onda', value=False)
    comparar_cap = st.toggle('Comparar com o Capytaine (BEM)', value=True,
                             help='RAO só com amortecimento de radiação. Para o FPSO original usa o resultado salvo; '
                                  'com outras dimensões é preciso rodar o Capytaine de novo.')
    malha = st.segmented_control('Malha do Capytaine', ['Rápida (6 m)', 'Padrão (3 m)'], default='Rápida (6 m)',
                                 disabled=not comparar_cap, help='A malha padrão é mais precisa, mas leva alguns minutos.')
    if st.button('Restaurar dados do exercício'):
        st.session_state.clear()
        st.rerun()

w = np.linspace(2 * np.pi / 60, w_max, 1500)
x = (2 * np.pi / w) if eixo_T else w
xt = 'Período da onda T [s]' if eixo_T else 'Frequência da onda ω [rad/s]'
x_lim = [2 * np.pi / w_max, 35]
p, c = modelo(L, B, T, rho, Ca, phi, w)
p0, c0 = modelo(REF['L'], REF['B'], REF['T'], REF['rho'], REF['Ca'], REF['phi'], w)

# ---------------- indicadores
def br(x, dec=0):
    """número no formato brasileiro: 123.000,5"""
    return f'{x:,.{dec}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


m = st.columns(6)
m[0].metric('Deslocamento Δ [t]', br(p['M']),
            delta=f'{br(100 * (p["M"] / p0["M"] - 1), 1)}% vs. exercício' if abs(p['M'] - p0['M']) > 1 else None,
            delta_color='off')
m[1].metric('Massa adicional Ma [t]', br(p['Ma']), help=f'Ma/M = {br(p["Ma"] / p["M"], 2)}')
m[2].metric('Rigidez c [kN/m]', br(p['c']))
m[3].metric('Freq. natural ωn [rad/s]', br(p['wn'], 3))
m[4].metric('Período natural Tn [s]', br(p['Tn'], 2),
            delta=f'{br(p["Tn"] - p0["Tn"], 2)} s vs. exercício' if abs(p['Tn'] - p0['Tn']) > 1e-3 else None,
            delta_color='off')
r_ = c['rao']
imax = np.where((r_[1:-1] > r_[:-2]) & (r_[1:-1] >= r_[2:]))[0] + 1        # máximos locais
if imax.size:
    i_res = imax[np.argmin(np.abs(w[imax] - p['wn']))]                     # o mais próximo de ωn
    m[5].metric('Pico do RAO [m/m]', br(r_[i_res], 3), help=f'em ω = {br(w[i_res], 3)} rad/s')
else:
    m[5].metric('Pico do RAO [m/m]', '—')

# ---------------- Capytaine
cap = None
if comparar_cap:
    malha = malha or 'Rápida (6 m)'
    chave = (L, B, T, round(rho, 3), malha)
    exec_ = st.session_state.get('cap_res')
    dims_ref = chave[:4] == (REF['L'], REF['B'], REF['T'], REF['rho'])
    ja_rodou = bool(exec_) and exec_['chave'] == chave
    if ja_rodou:
        cap = exec_
    elif dims_ref:
        cap = capytaine_referencia()
    if not ja_rodou:
        try:
            import capytaine  # noqa: F401
            disponivel = True
        except ImportError:
            disponivel = False
        with st.container(horizontal=True, vertical_alignment='center'):
            clicou = st.button('Recalcular com o Capytaine' if dims_ref else 'Rodar Capytaine para estas dimensões',
                               icon=':material/play_arrow:', type='secondary' if cap else 'primary',
                               disabled=not disponivel)
            if cap is None:
                st.caption('Dimensões diferentes do exercício: rode o Capytaine para ver a curva do BEM.')
            else:
                st.caption('Mostrando o resultado salvo do exercício (malha de 3 m).')
            if not disponivel:
                st.caption('Capytaine não instalado (pip install capytaine).')
        if clicou:
            dx = 3.0 if malha.startswith('Padrão') else 6.0
            barra = st.progress(0.0, text='Gerando malha e iniciando o Capytaine…')

            def progresso(i, n):
                barra.progress(i / n, text=(f'Capytaine: frequência {i + 1} de {n}…' if i < n
                                            else 'Capytaine: concluído'))

            cap = rodar_capytaine(L, B, T, rho, dx, 1.2 if dx > 3 else 1.6, progresso)
            cap['chave'] = chave
            st.session_state['cap_res'] = cap
            barra.empty()

col_f, col_r = st.columns(2)

# ---------------- forças
with col_f:
    st.subheader('Forças de onda')
    fig = go.Figure()
    fig.add_scatter(x=x, y=np.abs(c['F']) / 1e3, name='|F| = |F_FK + F_MA|', line=dict(color=COR['tot'], width=3))
    fig.add_scatter(x=x, y=np.abs(c0['F']) / 1e3, name='|F| – exercício', line=dict(color=COR['ref'], width=1.5, dash='dot'))
    if cap is not None:
        sel = cap['w'] <= w_max
        fig.add_scatter(x=(2 * np.pi / cap['w'][sel]) if eixo_T else cap['w'][sel], y=cap['F'][sel] / 1e3,
                        name='Capytaine |F_exc|', mode='lines+markers', line=dict(color=COR['cap'], width=2),
                        marker=dict(size=5, symbol='square'))
    fig.add_vline(x=p['Tn'] if eixo_T else p['wn'], line=dict(color=COR['wn'], dash='dash', width=1),
                  annotation_text=(f'Tn = {p["Tn"]:.2f} s' if eixo_T else f'ωn = {p["wn"]:.3f} rad/s'),
                  annotation_position='top right')
    fig.update_traces(hovertemplate='%{y:.2f} MN')
    fig = layout(fig, xt, '|Força de heave| [MN por m de amplitude]', 470)
    if eixo_T:
        fig.update_xaxes(range=x_lim)
    st.plotly_chart(fig, width='stretch')
    st.latex(r'F^h=2gA\,e^{-kT}\sin\!\left(\tfrac{kL}{2}\right)\left(\tfrac{\rho g B}{\omega^2}-M_{a2D}\right),'
             r'\quad M_{a2D}=C_a\tfrac{\rho\pi}{2}\left(\tfrac{B}{2}\right)^2,\quad k=\tfrac{\omega^2}{g}')

# ---------------- RAO
with col_r:
    st.subheader('RAO de heave')
    fig = go.Figure()
    fig.add_scatter(x=x, y=c['rao'], name=f'Analítico (Ca = {Ca:.2f}, φ = {100*phi:.1f}%)',
                    line=dict(color=COR['rao'], width=3))
    fig.add_scatter(x=x, y=c0['rao'], name='Analítico – exercício (Ca = 1,3, φ = 5%)',
                    line=dict(color=COR['ref'], width=1.5, dash='dot'))
    if cap is not None:
        sel = cap['w'] <= w_max
        fig.add_scatter(x=(2 * np.pi / cap['w'][sel]) if eixo_T else cap['w'][sel], y=cap['rao'][sel],
                        name='Capytaine (só radiação)', mode='lines+markers', line=dict(color=COR['cap'], width=2),
                        marker=dict(size=5, symbol='square'))
    xn = p['Tn'] if eixo_T else p['wn']
    fig.add_vline(x=xn, line=dict(color=COR['wn'], dash='dash', width=1),
                  annotation_text=(f'Tn = {p["Tn"]:.2f} s' if eixo_T else f'ωn = {p["wn"]:.3f} rad/s'),
                  annotation_position='top right')
    fig.update_traces(hovertemplate='%{y:.3f} m/m')
    fig = layout(fig, xt, 'RAO de heave |Z|/A [m/m]', 470)
    if eixo_T:
        fig.update_xaxes(range=x_lim)
    st.plotly_chart(fig, width='stretch')
    st.latex(r'RAO=\frac{|F^h|/(\rho g L B A)}{\sqrt{\left[1-(\omega/\omega_n)^2\right]^2+(2\varphi\,\omega/\omega_n)^2}},'
             r'\qquad \omega_n=\sqrt{\frac{\rho g L B}{\rho\nabla+M_a}}')

# ---------------- exportação
df = pd.DataFrame({'ω [rad/s]': w, 'T [s]': 2 * np.pi / w,
                   'F_FK [kN]': c['F_FK'], 'F_MA [kN]': c['F_MA'], 'F total [kN]': c['F'], 'RAO [m/m]': c['rao'],
                   'F total exercício [kN]': c0['F'], 'RAO exercício [m/m]': c0['rao']})
st.download_button('Exportar CSV (força e RAO)', df.to_csv(sep=';', decimal=',', index=False).encode('utf-8-sig'),
                   file_name='rao_heave_analitico.csv', mime='text/csv', icon=':material/download:')

with st.expander('Hipóteses do modelo'):
    st.markdown(
        '- Casco prismático com L ≫ B, T: forças calculadas na seção 2D e integradas ao longo do comprimento.\n'
        '- Força de Froude-Krylov: pressão da onda incidente no fundo (z = −T); costados verticais não contribuem.\n'
        '- Difração aproximada por uma massa adicional 2D constante, Ma2D = Ca ρπ/2 (B/2)².\n'
        '- Massa M = ρLBT (deslocamento da caixa); rigidez hidrostática c = ρgLB.\n'
        '- Amortecimento viscoso equivalente constante, b = 2φωn(M + Ma).\n'
        '- Heave desacoplado de pitch (casco simétrico proa–popa, sem velocidade de avanço).')
