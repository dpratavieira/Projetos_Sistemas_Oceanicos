# Projeto de Sistemas Oceânicos

Repositório com rotinas e programas para suporte às atividades didáticas de PNV3425 - Projeto de Sistemas Oceânicos, disciplina do quarto ano da grade obrigatória do curso de Engenharia Naval e Oceânica da USP

## Instalação e execução do dashboard

Com o Python instalado, abra um terminal na pasta deste repositório e instale as dependências:

```bash
python -m pip install "streamlit>=1.50" plotly numpy pandas capytaine xarray
```

Para iniciar o dashboard de RAO de heave do FPSO, execute:

```bash
python -m streamlit run dashboard_rao_heave_FPSO_caixa.py
```

O dashboard será aberto no navegador. Se isso não acontecer automaticamente, acesse <http://localhost:8501>.
Para encerrar o processo, pressione `Ctrl+C` no terminal.

O Capytaine e o xarray são usados para executar as simulações BEM pelo dashboard.
Se disponível na mesma pasta do script, o arquivo `ex6_capytaine_tabela.csv` fornece os resultados já calculados para o FPSO original.

## Tutoriais em Jupyter Notebook

Os tutoriais em arquivos `.ipynb` combinam explicações e células de código Python. Para utilizá-los, instale o Jupyter Notebook e inicie-o na pasta deste repositório:

```bash
python -m pip install notebook
python -m notebook
```

No navegador, abra o tutorial desejado e execute as células na ordem, de cima para baixo, usando `Shift+Enter`. Instale também as dependências indicadas em cada tutorial e mantenha os arquivos de dados nos caminhos esperados pelo notebook.

Os notebooks também podem ser executados diretamente no VS Code, com as extensões **Python** e **Jupyter** instaladas. Abra o arquivo `.ipynb`, selecione o kernel correspondente ao seu ambiente Python e execute as células em ordem.
