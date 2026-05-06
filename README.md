# Trabalho-IA-Redes-Neurais-NASA

## Descrição do projeto

Este repositório reúne o desenvolvimento do trabalho prático da disciplina de Inteligência Artificial, com foco em aprendizado supervisionado e redes neurais aplicadas a uma base pública da NASA. O projeto utiliza o conjunto **Kepler Objects of Interest (KOI)** para investigar um problema real de classificação a partir de dados astronômicos observacionais. [NASA Open Data Portal](https://data.nasa.gov/dataset/kepler-objects-of-interest-koi) [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)

A proposta central é construir, treinar e avaliar modelos de classificação, com ênfase em uma rede neural do tipo **MLP (Multi-layer Perceptron)**, comparando seu desempenho com pelo menos um modelo baseline mais simples. O experimento será implementado em **Google Colab**, com análise crítica dos resultados e relação com aplicações de Engenharia. [Slides de Aprendizado de Máquina](https://page.gensparksite.com/get_upload_url/89b66972225152d69f20c4e41ac153431cea47cbf574508a080fb186f012bf13/default/a2fe2d4f-2f18-488b-aab0-c56475e664e7)

---

## Objetivo

O objetivo deste projeto é aplicar conceitos de aprendizado de máquina e redes neurais para resolver um problema de classificação supervisionada usando a base KOI. O recorte inicial adotado neste repositório é a classificação binária com base em `koi_pdisposition`, que no arquivo baixado aparece com as classes **CANDIDATE** e **FALSE POSITIVE**. O dataset completo também contém a coluna `koi_disposition`, que apresenta três estados no arquivo atual: **FALSE POSITIVE**, **CONFIRMED** e **CANDIDATE**. [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html)

Em termos de trabalho acadêmico, o projeto busca documentar todo o ciclo: compreensão da base, pré-processamento, seleção de atributos, divisão entre treino/validação/teste, treinamento dos modelos, avaliação por métricas e interpretação dos resultados no contexto de aplicações reais. [Slides de Aprendizado de Máquina](https://page.gensparksite.com/get_upload_url/89b66972225152d69f20c4e41ac153431cea47cbf574508a080fb186f012bf13/default/a2fe2d4f-2f18-488b-aab0-c56475e664e7)

---

## Dataset utilizado: [Kepler Objects of Interest (KOI)](https://data.nasa.gov/dataset/kepler-objects-of-interest-koi)

O conjunto KOI é publicado pela NASA e mantido pelo **MAST Archive**. Ele reúne alvos observados pela missão **Kepler** que foram sinalizados como possíveis candidatos a exoplanetas, mas que podem incluir também falsos positivos decorrentes de artefatos instrumentais ou outros fenômenos astrofísicos. [NASA Open Data Portal](https://data.nasa.gov/dataset/kepler-objects-of-interest-koi) [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)

A documentação oficial destaca que a tabela **Cumulative** é a forma mais prática de trabalhar com os resultados atuais, pois consolida em um único lugar um superconjunto de KOIs vindos de diferentes quarters da missão. Por isso, ela é especialmente adequada para projetos acadêmicos e experimentos em Google Colab. [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)

A tabela cumulativa foi construída para reunir as disposições mais atuais e consistentes dos KOIs, com ênfase em estados como **CANDIDATE** e **FALSE POSITIVE**. A própria documentação informa que a tabela agrega resultados de diferentes tabelas de atividade e funciona como um catálogo consolidado de “best-knowledge” para os KOIs. [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html)

### Resumo validado do arquivo baixado

A versão do arquivo utilizada neste projeto foi baixada da tabela `cumulative` do Exoplanet Archive via TAP e, após inspeção local, apresenta as seguintes características:

- **Arquivo:** `kepler_koi_cumulative.csv`
- **Linhas:** 9.564 registros
- **Colunas:** 153 atributos
- **Atributos numéricos:** 133
- **Atributos não numéricos:** 20
- **Coluna-alvo binária viável:** `koi_pdisposition`
- **Coluna-alvo multiclasses possível:** `koi_disposition`

Validação feita diretamente sobre o CSV baixado para este repositório.

### Distribuição das classes observada no arquivo

#### `koi_pdisposition`
- `FALSE POSITIVE`: 4.847
- `CANDIDATE`: 4.717

#### `koi_disposition`
- `FALSE POSITIVE`: 4.839
- `CONFIRMED`: 2.747
- `CANDIDATE`: 1.978

Esses números mostram que o dataset permite tanto um problema **binário** quanto uma extensão **multiclasse**, sendo o cenário binário o mais indicado para a primeira versão do trabalho por simplicidade metodológica e melhor controle experimental.

### Observações importantes para pré-processamento

A inspeção do arquivo mostra que algumas colunas apresentam ausência total de dados na versão atual do CSV, como `koi_gmag_err`, `koi_imag_err`, `koi_zmag_err`, `koi_rmag_err`, `koi_kepmag_err`, além de certos atributos derivados e respectivos erros. Isso indica que a etapa de limpeza e seleção de atributos será essencial antes do treinamento dos modelos.

---

## Abordagem inicial do problema

A abordagem inicial adotada neste projeto será a classificação supervisionada binária para prever a disposição planetária usando `koi_pdisposition`. A ideia é transformar os atributos físicos e observacionais dos KOIs em variáveis de entrada para um modelo de classificação e verificar se uma rede neural consegue distinguir adequadamente objetos classificados como **CANDIDATE** e **FALSE POSITIVE**.

Como baseline, será utilizado pelo menos um modelo mais simples, como Regressão Logística ou Perceptron, para comparação com a rede neural MLP. Depois disso, serão avaliadas métricas como acurácia, precisão, recall, F1-score, matriz de confusão e, se aplicável, curva ROC/AUC. Esse fluxo está diretamente alinhado ao conteúdo da disciplina sobre aprendizado supervisionado, treino/teste/validação e avaliação de desempenho. [Slides de Aprendizado de Máquina](https://page.gensparksite.com/get_upload_url/89b66972225152d69f20c4e41ac153431cea47cbf574508a080fb186f012bf13/default/a2fe2d4f-2f18-488b-aab0-c56475e664e7)

---

## Estrutura de pastas

```text
Trabalho-IA-Redes-Neurais-NASA/
├── README.md
├── data/
│   └── kepler_koi_cumulative.csv
├── notebooks/
│   ├── 01_eda_koi.ipynb
│   ├── 02_preprocessamento_modelagem.ipynb
│   └── 03_trabalho_final.ipynb
├── figures/
│   ├── matriz_confusao.png
│   ├── roc_curve.png
│   └── comparacao_modelos.png
└── docs/
    └── planejamento.md
```

### Função de cada pasta

- **data/**: armazenamento do dataset bruto ou versões tratadas.
- **notebooks/**: notebooks usados para EDA, modelagem e versão final do trabalho.
- **figures/**: gráficos e imagens gerados durante análise e avaliação.
- **docs/**: planejamento, anotações metodológicas e apoio à escrita.

---

## Tecnologias utilizadas

Este projeto deverá utilizar principalmente as seguintes tecnologias:

- **Python 3**
- **Google Colab**
- **Pandas** para manipulação tabular
- **NumPy** para operações numéricas
- **Matplotlib** e **Seaborn** para visualização
- **Scikit-learn** para pré-processamento, baseline e avaliação
- **MLPClassifier** do Scikit-learn ou, alternativamente, **TensorFlow/Keras** para rede neural

A escolha dessas bibliotecas está alinhada ao enunciado do trabalho, que recomenda o uso de ferramentas modernas como Scikit-learn, TensorFlow ou Keras para implementação de redes neurais e avaliação de modelos.

---

## Instruções para execução no Google Colab

### Opção 1: usar o arquivo já salvo no projeto

1. Envie o arquivo `kepler_koi_cumulative.csv` para a pasta `data/` do repositório.
2. Abra o notebook no Google Colab.
3. Garanta que o caminho do CSV esteja correto.
4. Execute as células na ordem do notebook.

Exemplo de leitura com `pandas`:

```python
import pandas as pd

df = pd.read_csv('/content/kepler_koi_cumulative.csv')
df.head()
```

### Opção 2: usar Google Drive no Colab

```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
path = '/content/drive/MyDrive/Trabalho-IA-Redes-Neurais-NASA/kepler_koi_cumulative.csv'
df = pd.read_csv(path)
df.head()
```

### Opção 3: baixar programaticamente da fonte oficial

A documentação oficial informa que o acesso programático à tabela KOI Cumulative deve ser feito via **TAP**, já que o suporte legado de API para essa tabela foi descontinuado em 2023. [Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html)

Exemplo de consulta em CSV:

```python
import pandas as pd

url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv'
df = pd.read_csv(url)
df.head()
```

---

## Etapas previstas do projeto

1. Carregamento e inspeção do dataset
2. Análise exploratória dos dados
3. Limpeza e seleção de atributos
4. Definição do problema binário e/ou multiclasse
5. Separação treino/validação/teste
6. Treinamento de baseline
7. Treinamento da rede neural MLP
8. Avaliação com métricas e gráficos
9. Discussão crítica dos resultados
10. Conclusão com aplicações em Engenharia

---

## Possíveis métricas de avaliação

Para o cenário de classificação, serão priorizadas métricas como:

- Acurácia
- Precisão
- Recall
- F1-score
- Matriz de confusão
- Curva ROC e AUC, quando aplicável

Essas métricas estão em conformidade com o conteúdo discutido na disciplina para avaliação de modelos supervisionados de classificação. [Slides de Aprendizado de Máquina](https://page.gensparksite.com/get_upload_url/89b66972225152d69f20c4e41ac153431cea47cbf574508a080fb186f012bf13/default/a2fe2d4f-2f18-488b-aab0-c56475e664e7)

---

## Referências e fontes oficiais

- Dataset oficial KOI no NASA Open Data Portal: https://data.nasa.gov/dataset/kepler-objects-of-interest-koi
- Documentação KOI no NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html
- Explicação sobre a tabela cumulativa e disposições: https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html
- Acesso programático via TAP: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
- Consultas prontas para TAP/API: https://exoplanetarchive.ipac.caltech.edu/docs/API_queries.html

---

## Status inicial do projeto

- Dataset KOI baixado e validado
- Estrutura inicial do repositório em definição
- Planejamento de issues concluído
- Próxima etapa: EDA e definição final do pipeline de pré-processamento
