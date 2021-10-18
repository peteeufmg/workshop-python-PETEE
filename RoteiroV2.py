# ---------- Parte 0: Legenda  ---------- #

#Comentário: Comentário geral
#---> X <---#: Falar sobre X no vídeo
# !!!!!!!!!!!!! #: Executar o código até o momento e comentar sobre os resultados. 

# ---------- Parte 1: Importe as bibliotecas e funções necessárias  ---------- #

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Para dividir o dataset entre teste e treino
from sklearn.datasets import load_iris # Para pegar o dataset iris
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


#---> Explicar o dataset Iris, e como ele pode ser acessado usando o scikit-learn <---#
#---> Explicar o que são amostras e respostas <---#
#---> Explicar quais são as amostras e respostas desse dataset <---#


# ---------- Parte 2: Carregando o dataset ---------- #

iris = load_iris()
print(iris.feature_names) # O que foi medido em cada amostra
print(iris.data) # Amostras
# !!!!!!!!!!!!! #
#---> Cada linha da tabela acima é uma amostra de uma íris <---#

print(iris.target_names) # Possíveis categorias para cada Iris
print(iris.target) # Respostas, que devem ser valores numéricos
# !!!!!!!!!!!!! #
#---> Essas são as categorias possíveis que cada íris se encontra <---#


# ---------- Parte 3: Preparando os dados ---------- #

#---> Quatro requisitos são necessários para utilizar os dados no scikit learn :
	#---> Requisito 1: A amostra e a resposta devem ser numéricas
	#---> Requisito 2: A amostra e a resposta devem ser vetores NumPy
	#---> Requisito 3: A amostra e a resposta devem ter formatos específicos
	#---> Requisito 4: A amostra e resposta devem ser estruturas de dados separadas

#---> O Requisito 1 já foi verificado, e os valores são todos numéricos <---#

#---> O Requisito 2 pode ser verificado com: <---#

print(type(iris.data))
print(type(iris.target))
# !!!!!!!!!!!!! #

#---> O Requisito 3 pode ser verificado com a função shape: <---#

print(iris.data.shape)
print(iris.target.shape)
# !!!!!!!!!!!!! #

#---> Para cumprir o Requisito 4, armazenaremos iris.data e iris.target em duas estruturas de dados diferentes, que são chamadas de X(maiúsculo) e y(minúsculo) por convenção <---#

X = iris.data
y = iris.target


#---> Nesse caso em particular todos os requisitos foram cumpridos com facilidade, mas é muito provável que outros datasets precisarão de algum tipo de mudança dos dados <---#

# ---------- Parte 4: Treinando e Testando os dados ---------- #

#---> Em um determinado dataset, uma parte dos dados são destinados a treinar o sistema, e a outra parte é utilizada para verificar a precisão desse treino <---#
#---> A biblioteca sklearn tem uma função para realizar essa divisão entre treino e teste, chamada train_test_split <---#

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3)

#---> Mencionar que, na maioria das aplicações, o tamanho destinado ao teste fica entre 20 e 30 % <---#


# ---------- Parte 5: Modelo KNN ---------- #

#---> Explicação do KNN <---#

knn = KNeighborsClassifier(n_neighbors = 5) 

knn.fit(X_treino,y_treino)

print(knn.predict(X_teste))
print(y_teste)

# ---------- Parte X ---------- #


# ---------- Parte X ---------- #