# Classificação de candidatos a Exoplanetas utilizando Multicamadas(MLP)

## Introdução: 

Exoplanetas são objetos difíceis de serem detectados pois combinam distância, pequeno tamanho e baixo brilho gerando diversos de sinais que dificultam na detecção entre um planeta real, ruído ou apenas um erro. A base escolhida para o projeto contém registros reais captadas pela missão Kepler da NASA entre 2009 e 2018 na busca por exoplanetas (planetas fora do sistema solar), fornecendo uma forte base de dados envolvendo órbitas, brilho, anos, tamnho entre outros.

 O objetivo desse projeto é processar e analisar esses dados documentando e analisando esses dados com pré-processamento, seleção de atributos, divisão entre treino/validação/teste, treinamento dos models, avaliação por métricas e interpretação dos resultados. Através do treinamento, busca-se criar um classificador que maximize a detecção de planetas reais e minimeze os "Falsos Positivos".

## Implementação
Com a implementação realizada em Python, utilizando bibliotecas como Scikit-Learn para a rede Neural e Matplotlib/Seaborn para os gráficos de análise e avaliação

### Preparação e Limpeza de Dados

1. **Tratamento de Valores Ausentes (NaN):** Identificação de colunas com excesso de valores nulos que poderiam comprometer o aprendizado. Foi aplicada a remoção de linhas ou colunas vazias para evitar que o modelo tentasse "adivinhar" informações inexistentes.

2. **Seleção de Atributos (Feature Selection):** Eliminação de colunas irrelevantes para a classificação, como IDs de identificação dos objetos e aquelas que continham comentários textuais, focando apenas nos dados físicos e astronômicos (como brilho e órbita).

3. **Codificação de Rótulos (Label Encoding):** A coluna alvo (target) continha textos como CANDIDATE e FALSE POSITIVE, essas categorias foram transformadas em números (0 e 1) para que a rede neural pudesse processar os cálculos matemáticos. 

4. **Escalonamento (StandardScaler):** Como o dataset possui variáveis em escalas muito diferentes, utilizamos o StandardScaler para colocar todos os dados na mesma média e desvio padrão. Sem isso, a MLP demoraria muito mais para aprender ou poderia nem convergir.

### Implementação da Rede Neural
Após a etapa de preparação e limpeza, seguimos para a estruturação e execução do modelo de Aprendizado de Máquina:

5. **Divisão do Dataset (Holdout):**
   
Para garantir uma avaliação imparcial, os dados foram segmentados em três conjuntos distintos:

* **Treino (70%):** Utilizado para o ajuste dos pesos da rede neural.

* **Validação (15%):** Aplicado para a comparação de diferentes arquiteturas e ajuste de hiperparâmetros (como visto na Tabela Comparativa).

* **Teste (15%):** Reservado exclusivamente para a verificação final da capacidade de generalização do modelo em dados inéditos.

6. **Construção e Treinamento da MLP:**
   
Utilizando a classe MLPClassifier para instanciar a rede neural. Com base nos experimentos realizados, a configuração final selecionada foi:

* **Camadas Ocultas:** Uma camada com 100 neurônios, que demonstrou ser o tamanho ideal para a complexidade deste problema astronômico.

* **Regularização (Alpha):** Definido em 0.01 para atuar como um mecanismo de controle, evitando que o modelo "decore" ruídos do dataset (overfitting).

7. **Execução e Avaliação:**
   
O treinamento foi monitorado através da função de perda (loss function), que apresentou uma convergência estável ao longo de aproximadamente 70 épocas. Por fim, utilizamos o classification_report e a Matriz de Confusão para extrair métricas de precisão e revocação, garantindo que o modelo identifique corretamente tanto os candidatos a planetas quanto os alarmes falsos.

## Metodologia

* **Preparação dos Dados:** Os dados brutos passaram por uma limpeza de valores nulos e codificação de rótulos (Label Encoding). Devido à diferença de escala entre variáveis (como o período orbital em dias vs. o tamanho do raio planetário), aplicamos a Padronização (StandardScaler), essencial para o bom funcionamento de Redes Neurais.

* **Divisão do Dataset:** Os dados foram divididos em 70% para treino, 15% para validação (ajuste de hiperparâmetros) e 15% para teste (avaliação final), garantindo que o modelo seja testado em dados nunca vistos anteriormente.

* **Seleção de Algoritmos:** Utilizada uma Regressão Logística como Baseline para estabelecer um desempenho mínimo. Para a solução principal, foi escolhida a Rede Neural Multicamadas (MLP).

* **Arquitetura da Rede:** Após diversos experimentos (tabela comparativa), a arquitetura selecionada foi uma MLP com uma camada oculta de 100 neurônios e o parâmetro de regularização Alpha de 0.01, que previne o overfitting (quando a rede apenas decora os dados de treino).

## Discussão
A análise crítica dos resultados revela que a arquitetura de Redes Neurais (MLP) é altamente eficaz para a classificação de dados astronômicos complexos.

* **Pontos Positivos:** O modelo demonstrou uma capacidade de generalização excepcional, atingindo 98% de precisão e revocação para ambas as classes. Tal dado indica que a rede não apenas aprendeu a identificar planetas, mas também se tornou um filtro confiável para descartar ruídos estelares, um dos maiores desafios da missão Kepler.

* **Desafios e Limitações:** O maior desafio foi o ajuste da arquitetura. Inicialmente, acreditou-se que modelos mais profundos (com mais camadas) trariam maior precisão. No entanto, os experimentos demonstraram que redes excessivamente complexas tendem a perder performance ou estagnar o aprendizado. Outra limitação é a dependência da qualidade dos dados; sensores com ruídos diferentes dos da NASA poderiam exigir um novo treinamento do modelo.

* **Melhorias Futuras:** Como passos seguintes, poderiam ser exploradas técnicas de Ensemble Learning (combinação de múltiplos modelos) ou o uso de Redes Neurais Concorrentes (CNNs) para analisar diretamente as imagens das curvas de luz, em vez de apenas os dados tabulares extraídos.

## Conclusão
Os resultados obtidos neste projeto reforçam o papel da Engenharia de Dados e do Machine Learning na aceleração da descoberta científica. A substituição ou auxílio da triagem manual por um modelo automatizado de alta precisão representa uma economia significativa de recursos e tempo de pesquisa.

* **Contexto da Engenharia:** A capacidade de filtrar sinais em meio ao ruído é uma competência transversal. O sucesso deste experimento prova que bons algoritmos podem realizar tarefas de classificação com confiabilidade.

* **Aplicações Práticas:** Além da astronomia, a metodologia — envolvendo limpeza, escalonamento e uso de MLPs regularizadas — pode ser replicada em sistemas de manutenção preditiva (detecção de falhas em máquinas via sensores), diagnóstico médico por imagem e monitoramento de redes elétricas inteligentes (Smart Grids), onde a identificação de "anomalias" é o objetivo central.