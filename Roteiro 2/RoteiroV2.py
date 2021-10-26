# ---------- Parte 0: Legenda  ---------- #

#Comentário: Comentário geral
#---> X <---#: Falar sobre X no vídeo
# !!!!!!!!!!!!! #: Executar o código até o momento e comentar sobre os resultados.
#******************  Y  ******************#: Utilizado para marcar o ponto Y
#### [Ponto Y] XXXXX #####: Faça ou comente sobre X no ponto Y 

# ---------- Parte 1: Importe as bibliotecas e funções necessárias  ---------- #

import numpy as np
from sklearn.model_selection import train_test_split # Para dividir o dataset entre teste e treino
from sklearn.datasets import load_iris # Para pegar o dataset iris
from sklearn.metrics import accuracy_score # Para medir a precisão de um modelo
from sklearn.neighbors import KNeighborsClassifier

#---> Mostrar o Slide 1: Dataset Íris <---#
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

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3) #******************  1  ******************#

#---> Mencionar que, na maioria das aplicações, o tamanho destinado ao teste fica entre 20 e 30 % <---#

# ---------- Parte 5: Modelo KNN ---------- #

#---> Mostrar o Slide 2: "Como o modelo KNN funciona" <---#

#---> Crie uma cópia do modelo(o scikit learn chama modelos de estimadores) <---#
knn = KNeighborsClassifier(n_neighbors = 6) #******************  2  ******************#

# ---------- Parte 6: Treinando e Testando o Modelo KNN ---------- #

#---> Explicar o que é o fit do modelo <---#
knn.fit(X_treino,y_treino)
#---> Agora, esse objeto knn está pronto para realizar previsões no restante dos dados <---#

#---> Vamos prever o resultado de X_teste e comparar esse resultado com o y_teste <---#
print(knn.predict(X_teste))
print(y_teste)
# !!!!!!!!!!!!! #

#---> Normalmente, a função accuracy_score é utilizada para medir a precisão de um modelo <---#
#---> A função accuracy_score recebe como entrada os resultados corretos de cada amostra e os resultados obtidos pelo modelo <---#
#---> A função accuracy_score retorna um valor entre 0 e 1 que representa a precisão do modelo <---#
resultados_corretos = y_teste
resultados_do_modelo = knn.predict(X_teste)
precisao = accuracy_score(resultados_corretos, resultados_do_modelo)
print(precisao)
# !!!!!!!!!!!!! #
#---> Execute o programa diversas vezes para mostrar e explicar porque a precisão não é constante <---#

#### [Ponto 1] Mostre que a variável random_state pode ser utilizada para fixar as amostras que serão utilizadas para o teste ####

# ---------- Parte 7: Ajuste de parâmetros ---------- #

#### [Ponto 2] Mostre, por meio da execução do programa, como que mudanças no parâmetro n_neighbors alteram a precisão do modelo ####
#---> Explique que um projeto de Machine Learning deve levar em consideração não apenas implementar o modelo, mas garantir que sua precisão seja a melhor possível <---#
#---> Para o modelo KNN, na maioria dos casos o valor de n_neighbors que provavelmente terá a maior precisão é o número inteiro mais próximo da raíz quadrada do número de amostras dividido por 2 <---#
#---> Portanto, como temos 150 itens, o valor de n_neighbors deverá ser (raíz de 150)/2, que é 6 <---#
#---> Essa relação para n_neighbors não estará correta em todas as situações, mas é um bom ponto de partida <---#

# ---------- Parte 8: Conclusão do Vídeo ---------- #



#---> Explique o funcionamento geral do código desde o início <---#
#---> Explique que existem diversos parâmetros que podem ser utilizados na função KNeighborsClassifier, que alteram o comportamento do modelo <---#
#---> Fale que, nos próximos vídeos, analisaremos novos modelos, como visualizar graficamente os dados e como utilizar bancos de dados externos <---#  