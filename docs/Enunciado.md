# Enunciado do Trabalho

Você deverá aplicar os conceitos de [aprendizado de máquina](Slides-Aprendizado-Maquina.pdf) e redes neurais estudados em aula para resolver um problema real usando uma base de dados aberta. Esta atividade tem como finalidade integrar fundamentos teóricos e práticos, além de incentivar o desenvolvimento de habilidades analíticas, experimentação e avaliação crítica dos resultados de modelos computacionais

---

# Orientações Gerais

- Escolha uma base de dados pública e reconhecida: Exemplos incluem Iris, Wine Quality, MNIST, Boston Housing, PIMA Diabetes, etc. Dê preferência para conjuntos disponíveis na internet ou nos livros indicados.

- Defina o problema de interesse: Pode ser classificação (prever categorias, como espécie de flor, ocorrência de doença), ou regressão (prever valores contínuos, como preço de casas).

- Selecione e implemente uma arquitetura de rede neural: Pode ser Perceptron, MLP (Multi-layer Perceptron), CNN, RNN, entre outras, de acordo com a natureza do problema. Utilize bibliotecas modernas como Scikit-learn, TensorFlow ou Keras (exemplos disponíveis nos livros de Géron, Chollet, Müller & Guido, Raschka).

- Documente e justifique as escolhas: Explique por que selecionou o tipo de rede neural, como foi o pré-processamento dos dados e a divisão entre treino e teste/validação.

- Avalie o desempenho dos modelos: Use métricas de classificação (acurácia, precisão, recall, F1-score, matriz de confusão, curva ROC) ou de regressão (MSE, MAE, R²). Compare e discuta os resultados obtidos.

- Interprete os resultados e relacione com aplicações em Engenharia: Encerre com uma breve análise sobre a viabilidade de uso do seu modelo em sistemas reais (automação, controle, predição, reconhecimento de padrões, etc.).

---

# Entregáveis e Estrutura do Relatório

O trabalho pode ser desenvolvido em notebook Python (Jupyter ou Google Colab), contendo:

- Introdução: Contextualize o problema, a base de dados e objetivos do experimento.

- Metodologia: Descreva o processo de escolha e preparação dos dados, seleção dos algoritmos e arquitetura da rede neural.

- Implementação: Apresente trechos relevantes do código, mostrando importação, construção, treinamento e avaliação do modelo. Use comentários claros e explique o que cada etapa realiza.

- Análise dos resultados: Traga gráficos ilustrativos (ex: matriz de confusão, curvas de erro/perda, classificações), além de tabelas ou visualizações das métricas calculadas. Comparações podem ser incluídas (exemplo: MLP versus Perceptron).

- Discussão: Interprete criticamente os pontos positivos, limitações, desafios enfrentados e possíveis melhorias.

- Conclusão: Relacione os resultados ao contexto da Engenharia e sugira aplicações práticas.
