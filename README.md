# Trabalho-IA-Redes-Neurais-NASA

## Descrição do projeto

Este repositório reúne o desenvolvimento do trabalho prático da disciplina de Inteligência Artificial, com foco em aprendizado supervisionado e redes neurais aplicadas a uma base pública da NASA. O projeto utiliza o conjunto **Kepler Objects of Interest (KOI)** para investigar um problema real de classificação a partir de dados astronômicos observacionais. [NASA Open Data Portal](https://data.nasa.gov/dataset/kepler-objects-of-interest-koi) [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)

A proposta central foi construir, treinar e avaliar modelos de classificação, com ênfase em uma rede neural do tipo **MLP (Multi-layer Perceptron)** implementada com o `MLPClassifier` do **Scikit-Learn**, comparando seu desempenho com um modelo baseline de **Regressão Logística**. O experimento foi desenvolvido em notebooks Jupyter compatíveis com **Google Colab**, com análise crítica dos resultados e relação com aplicações de Engenharia.

---

## Objetivo

O objetivo deste projeto é aplicar conceitos de aprendizado de máquina e redes neurais para resolver um problema de classificação supervisionada usando a base KOI. O recorte adotado é a **classificação binária** com base em `koi_pdisposition`, com as classes **CANDIDATE** e **FALSE POSITIVE**. O dataset completo também contém a coluna `koi_disposition`, que apresenta três estados: **FALSE POSITIVE**, **CONFIRMED** e **CANDIDATE**. [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html)

O projeto documenta todo o ciclo: compreensão da base, pré-processamento, seleção de atributos, divisão entre treino/validação/teste, treinamento dos modelos, avaliação por métricas e interpretação dos resultados no contexto de aplicações reais. [Slides de Aprendizado de Máquina](docs/Slides-Aprendizado-Maquina.pdf)

---

## Dataset utilizado: [Kepler Objects of Interest (KOI)](https://data.nasa.gov/dataset/kepler-objects-of-interest-koi)

O conjunto KOI é publicado pela NASA e mantido pelo **MAST Archive**. Ele reúne alvos observados pela missão **Kepler** que foram sinalizados como possíveis candidatos a exoplanetas, mas que podem incluir também falsos positivos decorrentes de artefatos instrumentais ou outros fenômenos astrofísicos. [NASA Open Data Portal](https://data.nasa.gov/dataset/kepler-objects-of-interest-koi) [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)

A tabela **Cumulative** consolida em um único lugar um superconjunto de KOIs vindos de diferentes quarters da missão. Por isso, ela é especialmente adequada para projetos acadêmicos e experimentos reprodutíveis. [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)

### Resumo validado do arquivo baixado

A versão do arquivo utilizada neste projeto foi baixada da tabela `cumulative` do Exoplanet Archive via **TAP (Table Access Protocol)**:

- **Linhas:** 9.564 registros
- **Colunas originais:** 153 atributos
- **Colunas após pré-processamento:** 108 features (removidas colunas com >50% de nulos e metadados)
- **Coluna-alvo:** `koi_pdisposition`

### Distribuição das classes

#### `koi_pdisposition`
- `FALSE POSITIVE`: 4.847
- `CANDIDATE`: 4.717

O dataset é **bem balanceado**, não exigindo técnicas de oversampling para o treinamento.

### Observações sobre pré-processamento

Algumas colunas apresentavam ausência total de dados na versão do CSV, como `koi_gmag_err`, `koi_imag_err`, `koi_zmag_err`, `koi_rmag_err`, `koi_kepmag_err`, além de atributos derivados e erros associados. Todas as colunas com mais de 50% de valores nulos foram removidas. Os nulos remanescentes foram imputados pela **mediana**.

---

## Abordagem adotada

O projeto implementou a classificação supervisionada binária para prever `koi_pdisposition`. O pipeline completo foi:

1. **Carga programática via TAP** (sem dependência de arquivo local)
2. **Limpeza e seleção de features** (remoção de IDs, metadados e colunas com >50% nulos)
3. **Label Encoding** do alvo (`CANDIDATE` → 0, `FALSE POSITIVE` → 1)
4. **Imputação pela mediana** dos valores ausentes restantes
5. **Split estratificado** 70% Treino / 15% Validação / 15% Teste
6. **Padronização** com `StandardScaler`
7. **Baseline** com Regressão Logística (Acurácia: **93,45%**)
8. **MLP v1** com 100 neurônios, ReLU, Adam (Acurácia: **98,33%**)
9. **Refinamento** com 3 configurações adicionais — melhor resultado com Alpha=0.01 (Acurácia: **98,40%**)

### Resultados obtidos

| Modelo | Acurácia (Validação) | Configuração |
| :--- | :---: | :--- |
| Baseline (Regressão Logística) | 93,45% | N/A |
| MLP v1 | 98,33% | (100,) ReLU, Adam |
| Exp A — Profunda | 95,82% | (100, 50) |
| Exp B — Larga | 96,59% | (200,) |
| **Exp C — Regularizada** ⭐ | **98,40%** | **(100,) Alpha=0.01** |

O **Experimento C** foi selecionado como modelo final por apresentar a maior acurácia com regularização L2 que reduz o risco de overfitting.

---

## Estrutura de pastas

```text
Trabalho-IA-Redes-Neurais-NASA/
├── README.md
├── requirements.txt
├── .cursorrules
├── data/
│   └── (dataset carregado programaticamente via TAP)
├── notebooks/
│   ├── 01_eda_koi.ipynb               ← EDA e diagnóstico dos dados
│   └── 02_preprocessamento_modelagem.ipynb  ← Pipeline + Baseline + MLP + Refinamento
├── figures/
│   └── (gráficos gerados durante avaliação)
└── docs/
    ├── Enunciado.md
    └── Slides-Aprendizado-Maquina.pdf
```

### Função de cada pasta

- **data/**: reservada para armazenamento local do dataset, se necessário. O carregamento padrão é feito via TAP.
- **notebooks/**: contém os notebooks executáveis com todo o pipeline do projeto.
- **figures/**: destino dos gráficos gerados (Matriz de Confusão, Curva ROC, etc.).
- **docs/**: enunciado do trabalho e slides da disciplina.

---

## Tecnologias utilizadas

As seguintes tecnologias foram efetivamente utilizadas no projeto:

- **Python 3.12**
- **Jupyter Notebooks** (compatível com Google Colab)
- **Pandas 3.0** — manipulação e limpeza tabular
- **NumPy 2.4** — operações numéricas e imputação
- **Matplotlib 3.10** e **Seaborn 0.13** — visualizações e gráficos
- **Scikit-learn 1.8** — pré-processamento (`StandardScaler`, `LabelEncoder`), baseline (`LogisticRegression`), rede neural (`MLPClassifier`) e métricas

> **Nota:** O projeto utiliza exclusivamente o `MLPClassifier` do Scikit-Learn para a rede neural MLP. TensorFlow/Keras não foram utilizados.

---

## Como executar

### Pré-requisitos (execução local)

```bash
# Ativar o ambiente virtual
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### Executar no Google Colab

Abra o notebook desejado diretamente no Colab. Os dados são carregados automaticamente via TAP:

```python
import pandas as pd

url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv'
df = pd.read_csv(url)
```

> A Opção 3 (TAP) é a forma padrão adotada neste projeto, garantindo acesso sempre à versão mais atualizada da tabela cumulativa.

---

## Etapas realizadas

1. ✅ Carregamento e inspeção do dataset (EDA — `01_eda_koi.ipynb`)
2. ✅ Análise exploratória: distribuição de classes, diagnóstico de nulos
3. ✅ Limpeza e seleção de atributos (108 features finais)
4. ✅ Definição do problema binário (`koi_pdisposition`)
5. ✅ Separação estratificada treino/validação/teste (70/15/15)
6. ✅ Treinamento e avaliação do baseline (Regressão Logística, 93,45%)
7. ✅ Treinamento da MLP v1 (98,33%)
8. ✅ Refinamento com 3 configurações — modelo final com 98,40%
9. ⬜ Avaliação visual (Matriz de Confusão, Curva ROC)
10. ⬜ Discussão crítica e conclusão com aplicações em Engenharia

---

## Possíveis métricas de avaliação

Para o cenário de classificação, foram e serão utilizadas:

- Acurácia
- Precisão
- Recall
- F1-score
- Matriz de confusão
- Curva ROC e AUC

Essas métricas estão em conformidade com o conteúdo discutido na disciplina para avaliação de modelos supervisionados de classificação. [Slides de Aprendizado de Máquina](docs/Slides-Aprendizado-Maquina.pdf)

---

## Referências e fontes oficiais

- Dataset oficial KOI no NASA Open Data Portal: https://data.nasa.gov/dataset/kepler-objects-of-interest-koi
- Documentação KOI no NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html
- Explicação sobre a tabela cumulativa e disposições: https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html
- Acesso programático via TAP: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
- Documentação do MLPClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
