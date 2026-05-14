# Classificação de candidatos a Exoplanetas utilizando Multicamadas(MLP)

## Introdução: 

Exoplanetas são objetos difíceis de serem detectados pois combinam distância, pequeno tamanho e baixo brilho gerando diversos sinais que dificultam na detecção entre um planeta real, ruído ou apenas um erro. A base escolhida para o projeto contém registros reais captadas pela missão Kepler da NASA entre 2009 e 2018 na busca por exoplanetas (planetas fora do sistema solar), fornecendo uma forte base de dados envolvendo órbitas, brilho, anos, tamnho entre outros.

 O objetivo desse projeto é processar e analisar esses dados documentando e analisando esses dados com pré-processamento, seleção de atributos, divisão entre treino/validação/teste, treinamento dos models, avaliação por métricas e interpretação dos resultados. Através do treinamento, busca-se criar um classificador que maximize a detecção de planetas reais e minimize os "Falsos Positivos".

## Implementação
Com a implementação realizada em Python, utilizando bibliotecas como Scikit-Learn para a rede Neural e Matplotlib/Seaborn para os gráficos de análise e avaliação

### Preparação e Limpeza de Dados

1. **Tratamento de Valores Ausentes (NaN):** Identificação de colunas com excesso de valores nulos que poderiam comprometer o aprendizado. Foi aplicada a remoção de linhas ou colunas vazias para evitar que o modelo tentasse "adivinhar" informações inexistentes.

```py
# Definindo o alvo
target = 'koi_pdisposition'  # Define a coluna alvo da classificação

# Colunas para remover (IDs e metadados irrelevantes para treinamento)
cols_to_drop = [  # Lista de colunas irrelevantes a remover (IDs e metadados)
 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 
 'koi_tce_delivname', 'koi_fittype', 'ra_str', 'dec_str'
]

# Identificando colunas com mais de 50% de valores nulos (conforme observado na EDA)
null_threshold = 0.5 * len(df_raw)  # Limite de 50% de nulos para remoção de coluna
high_null_cols = df_raw.columns[df_raw.isnull().sum() > null_threshold].tolist()  # Soma os valores nulos por coluna

total_drop = list(set(cols_to_drop + high_null_cols))  # Lista de colunas irrelevantes a remover (IDs e metadados)
df = df_raw.drop(columns=total_drop)  # Combina lista de colunas para remoção

print(f"Colunas removidas: {len(total_drop)}")  # Combina lista de colunas para remoção
print(f"Novo formato: {df.shape}")  # Retorna (linhas, colunas) do dataframe
```

2. **Seleção de Atributos (Feature Selection):** Eliminação de colunas irrelevantes para a classificação, como IDs de identificação dos objetos e aquelas que continham comentários textuais, focando apenas nos dados físicos e astronômicos (como brilho e órbita).


3. **Codificação de Rótulos (Label Encoding):** A coluna alvo (target) continha textos como CANDIDATE e FALSE POSITIVE, essas categorias foram transformadas em números (0 e 1) para que a rede neural pudesse processar os cálculos matemáticos. 

```py
# Encoding do alvo
le = LabelEncoder()  # Cria codificador de rótulos categóricos
df[target] = le.fit_transform(df[target])  # Aprende e transforma rótulos CANDIDATE/FALSE POSITIVE → 0/1
print(f"Classes mapeadas: {dict(zip(le.classes_, le.transform(le.classes_)))}")  # Transforma categorias usando mapeamento já aprendido

# Imputação pela mediana em colunas numéricas
df_numeric = df.select_dtypes(include=[np.number])  # Seleciona apenas colunas numéricas
df_final = df_numeric.fillna(df_numeric.median())  # Preenche nulos com a mediana de cada coluna

print(f"Nulos remanescentes: {df_final.isnull().sum().sum()}")  # Soma os valores nulos por coluna
```

4. **Escalonamento (StandardScaler):** Como o dataset possui variáveis em escalas muito diferentes, utilizamos o StandardScaler para colocar todos os dados na mesma média e desvio padrão. Sem isso, a MLP demoraria muito mais para aprender ou poderia nem convergir.

```py
scaler = StandardScaler()  # Cria o padronizador (média=0, desvio=1)
X_train_scaled = scaler.fit_transform(X_train)  # Aprende e aplica escalonamento no treino
X_val_scaled = scaler.transform(X_val)  # Escala validação com parâmetros do treino
X_test_scaled = scaler.transform(X_test)  # Escala teste com parâmetros do treino

print("Escalonamento concluído com StandardScaler.")  # Confirma conclusão do escalonamento
```

### Implementação da Rede Neural
Após a etapa de preparação e limpeza, seguimos para a estruturação e execução do modelo de Aprendizado de Máquina:

5. **Divisão do Dataset (Holdout):**

Para garantir uma avaliação imparcial, os dados foram segmentados em três conjuntos distintos:

* **Treino (70%):** Utilizado para o ajuste dos pesos da rede neural.

* **Validação (15%):** Aplicado para a comparação de diferentes arquiteturas e ajuste de hiperparâmetros (como visto na Tabela Comparativa).

* **Teste (15%):** Reservado exclusivamente para a verificação final da capacidade de generalização do modelo em dados inéditos.

```py
X = df_final.drop(columns=[target])  # Remove as colunas especificadas
y = df_final[target]  # Extrai a coluna alvo como variável dependente

# Primeiro split: Treino vs Resto (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  # Primeiro split: 70% treino, 30% restante

# Segundo split: Validação vs Teste (50% do Resto = 15% cada)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)  # Segundo split: divide 30% em 15% validação e 15% teste

print(f"Treino: {X_train.shape}")  # Retorna (linhas, colunas) do dataframe
print(f"Validação: {X_val.shape}")  # Retorna (linhas, colunas) do dataframe
print(f"Teste: {X_test.shape}")  # Retorna (linhas, colunas) do dataframe
```

6. Regressão Logística
   
Esta etapa estabelece um modelo de referência (baseline) usando Regressão Logística para classificação binária de exoplanetas. O objetivo é criar um ponto de comparação simples antes de aplicar redes neurais mais complexas. O modelo utiliza 108 features processadas e escalonadas para prever se um objeto é CANDIDATE (0) ou FALSE POSITIVE (1).

```py
# Instanciando o modelo
baseline = LogisticRegression(max_iter=1000, random_state=42)  # Cria modelo baseline de Regressão Logística
baseline.fit(X_train_scaled, y_train)  # Treina a Regressão Logística com dados de treino

y_pred_base = baseline.predict(X_val_scaled)  # Gera predições da Regressão no conjunto de validação
acc_base = accuracy_score(y_val, y_pred_base)  # Calcula acurácia do baseline na validação
f1_base = f1_score(y_val, y_pred_base, average='weighted')  # Calcula F1-score ponderado do baseline

print(f"Baseline treinado! Acurácia: {acc_base:.4f} | F1: {f1_base:.4f}")  # Exibe métricas do modelo baseline
```

7. **Construção e Treinamento da MLP:**

Utilizando a classe MLPClassifier para instanciar a rede neural. Com base nos experimentos realizados, a configuração final selecionada foi:

* **Camadas Ocultas:** Uma camada com 100 neurônios, que demonstrou ser o tamanho ideal para a complexidade deste problema astronômico.
  
```py
mlp_v1 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True)  # Cria rede MLP v1: 1 camada, 100 neurônios, ReLU, Adam
mlp_v1.fit(X_train_scaled, y_train)  # Treina a MLP v1 com os dados de treino

y_pred_v1 = mlp_v1.predict(X_val_scaled)  # Gera predições da MLP v1 na validação
acc_v1 = accuracy_score(y_val, y_pred_v1)  # Calcula acurácia da MLP v1
f1_v1 = f1_score(y_val, y_pred_v1, average='weighted')  # Calcula F1-score ponderado da MLP v1

print(f"MLP v1 treinada! Acurácia: {acc_v1:.4f} | F1: {f1_v1:.4f}")  # Exibe métricas da MLP v1
```

* **Regularização (Alpha):** Definido em 0.01 para atuar como um mecanismo de controle, evitando que o modelo "decore" ruídos do dataset (overfitting).

```py
# Experimento A: Arquitetura Profunda (2 camadas)
mlp_a = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)  # Cria MLP Exp A: 2 camadas (100+50 neurônios)
mlp_a.fit(X_train_scaled, y_train)  # Treina o Exp A com dados de treino
acc_a = accuracy_score(y_val, mlp_a.predict(X_val_scaled))  # Calcula acurácia do Exp A

# Experimento B: Arquitetura Larga (200 neurônios)
mlp_b = MLPClassifier(hidden_layer_sizes=(200,), max_iter=500, random_state=42, early_stopping=True)  # Cria MLP Exp B: 1 camada com 200 neurônios
mlp_b.fit(X_train_scaled, y_train)  # Treina o Exp B com dados de treino
acc_b = accuracy_score(y_val, mlp_b.predict(X_val_scaled))  # Calcula acurácia do Exp B

# Experimento C: Regularização Forte (Alpha=0.01)
mlp_c = MLPClassifier(hidden_layer_sizes=(100,), alpha=0.01, max_iter=500, random_state=42, early_stopping=True)  # Cria MLP Exp C: regularização L2 (alpha=0.01)
mlp_c.fit(X_train_scaled, y_train)  # Treina o Exp C com dados de treino
acc_c = accuracy_score(y_val, mlp_c.predict(X_val_scaled))  # Calcula acurácia do Exp C

print("Experimentos concluídos!")  # Confirma fim dos experimentos
```

1. **Execução e Avaliação:**

O treinamento foi monitorado através da função de perda (loss function), que apresentou uma convergência estável ao longo de aproximadamente 70 épocas. Por fim, foi utilizado o ``classification_report`` e a Matriz de Confusão para extrair métricas de precisão e revocação, garantindo que o modelo identifique corretamente tanto os candidatos a planetas quanto os alarmes falsos, esse e outros gráficos estão presentes no segundo arquivo EDA. Para confirmar essa precisão da rede foi elaborado um terceiro arquivo EDA com uma demonstração prática. 

## Metodologia

* **Preparação dos Dados:** Os dados brutos passaram por uma limpeza de valores nulos e codificação de rótulos (Label Encoding). Devido à diferença de escala entre variáveis (como o período orbital em dias vs. o tamanho do raio planetário), aplicamos a Padronização (StandardScaler), essencial para o bom funcionamento de Redes Neurais.

* **Divisão do Dataset:** Os dados foram divididos em 70% para treino, 15% para validação (ajuste de hiperparâmetros) e 15% para teste (avaliação final), garantindo que o modelo seja testado em dados nunca vistos anteriormente.

* **Seleção de Algoritmos:** Utilizada uma Regressão Logística como Baseline para estabelecer um desempenho mínimo. Para a solução principal, foi escolhida a Rede Neural Multicamadas (MLP).

* **Arquitetura da Rede:** Após diversos experimentos (tabela comparativa), a arquitetura selecionada foi a MLP com uma camada oculta de 100 neurônios e o parâmetro de regularização Alpha de 0.01, que previne o overfitting (quando a rede apenas decora os dados de treino).

## Discussão
A análise crítica dos resultados revela que a arquitetura de Redes Neurais (MLP) é altamente eficaz para a classificação de dados astronômicos complexos.

* **Pontos Positivos:** O modelo demonstrou uma capacidade de generalização excepcional, atingindo 98% de precisão e revocação para ambas as classes. Tal dado indica que a rede não apenas aprendeu a identificar planetas, mas também se tornou um filtro confiável para descartar ruídos estelares, um dos maiores desafios da missão Kepler.

* **Desafios e Limitações:** O maior desafio foi o  ajuste da arquitetura. Inicialmente, acreditou-se que modelos mais profundos (com mais camadas) trariam maior precisão. No entanto, os experimentos demonstraram que redes excessivamente complexas tendem a perder performance ou estagnar o aprendizado. Outra limitação é a dependência da qualidade dos dados; sensores com ruídos diferentes dos da NASA poderiam exigir um novo treinamento do modelo.

* **Melhorias Futuras:** Como passos seguintes, há a possibilidade de explorar técnicas de Ensemble Learning (combinação de múltiplos modelos) ou o uso de Redes Neurais Concorrentes (CNNs) para analisar diretamente as imagens das curvas de luz, em vez de apenas os dados tabulares extraídos.

## Conclusão
Os resultados obtidos neste projeto reforçam o papel da Engenharia e do Machine Learning na aceleração da descoberta científica. A substituição ou auxílio da triagem manual por um modelo automatizado de alta precisão representa uma economia significativa de recursos e tempo de pesquisa.

* **Contexto da Engenharia:** A capacidade de filtrar sinais em meio ao ruído é uma competência transversal. O sucesso deste experimento prova que bons algoritmos podem realizar tarefas de classificação com confiabilidade.

* **Aplicações Práticas:** Além da astronomia, a metodologia — envolvendo limpeza, escalonamento e uso de MLPs regularizadas — pode ser replicada em sistemas de manutenção preditiva (detecção de falhas em máquinas via sensores), diagnóstico médico por imagem e monitoramento de redes elétricas inteligentes (Smart Grids), onde a identificação de "anomalias" é o objetivo central.
