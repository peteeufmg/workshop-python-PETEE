# ---------- Parte 0: Legenda  ---------- #

#Comentário: Comentário geral
#---> X <---#: Falar sobre X no vídeo
# !!!!!!!!!!!!! #: Executar o código até o momento e comentar sobre os resultados.
#******************  Y  ******************#: Utilizado para marcar o ponto Y
#### [Ponto Y] XXXXX #####: Faça ou comente sobre X no ponto Y 

# ---------- Parte 1: Importar bibliotecas e funções necessárias  ---------- #

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#Survived 	Survival 	0 = No, 1 = Yes
#Pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd
#Sex 	Sex 	
#Age 	Age in years 	
#Sibsp 	# of siblings / spouses aboard the Titanic 	
#Parch 	# of parents / children aboard the Titanic 	
#Ticket 	Ticket number 	
#Fare 	Passenger fare 	
#Cabin 	Cabin number 	
#Embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton


#---> Pegar o banco de dados do Kaggle <---#

# ---------- Parte 2: Explicar o problema e a nossa base de dados  ---------- #

#---> Mostrar o Slide 3.1: Dataset Titanic <---#

# ---------- Parte 3: Ler o arquivo para treino; funções básicas Pandas  ---------- #

#---> Explicar sobre o Pandas <---#
#---> Explicar o que é um arquivo csv <---#
#---> Mostrar o arquivo test.csv no PyCharm <---#

dados = pd.read_csv("train.csv")
#print(dados)
# !!!!!!!!!!!!! #

#---> Mostrar alguns dos recursos do pandas por meio dessas funções <---#
#print(dados.head())
# !!!!!!!!!!!!! #
#print(dados.tail())
# !!!!!!!!!!!!! #
#print(dados.tail(10))
# !!!!!!!!!!!!! #
#print(dados.describe())
# !!!!!!!!!!!!! #
#print(dados.isnull().sum()) # Utilizado para detectar NaN
# !!!!!!!!!!!!! #

#---> Falar que o Pandas possui mais recursos para análise <---#

# ---------- Parte 4: Preparar os dados ---------- #

#---> Ao contrário do dataset Íris, o dataset Titanic possui muito mais colunas. Algumas delas não tem valores numéricos  <---#
#---> Portanto, precisaremos selecionar as colunas e garantir que nossos dados atendam aos 4 requisitos  <---#

#---> Podemos criar um DataFrame e selecionar colunas específicas da seguinte maneira <---#
X = pd.DataFrame(dados, columns = ["Pclass","Sex","Age"])
#print(X)
# !!!!!!!!!!!!! #

#---> Fazendo o mesmo para a resposta <---#
y = pd.DataFrame(dados, columns = ["Survived"])
#print(y)
# !!!!!!!!!!!!! #


#---> A coluna "Sex" contém palavras. Podemos transformá-la para valores numéricos com: <---#
#X["Sex"] = X["Sex"].map( {"male":0,"female":1} ).astype(int)
X["Sex"].replace({"male":0,"female":1}, inplace=True)
#print(X)
# !!!!!!!!!!!!! #


#---> A coluna Age possui valores NaN(Not a Number) <---#
#---> Podemos fazer com que os valores NaN sejam preenchidos com o valor médio da coluna por <---#
X["Age"].fillna(value = X["Age"].median(), inplace = True)
#print(X)
# !!!!!!!!!!!!! #

# ---------- Parte 5: Explicar a árvore de decisões  ---------- #

#---> Mostrar o Slide 3.2: Árvore de Decisões <---#

# ---------- Parte 6: Fazer o treino e o teste, aplicar o modelo árvore de decisões  ---------- #


#---> Dividindo o dataset entre treino e teste <---#
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 5)

#---> Instanciando o modelo e fazendo o fit <---#
clf = DecisionTreeClassifier(criterion="gini")
clf.fit(X_treino,y_treino)

#---> Analisando agora a precisão do modelo <---#
resultados_corretos = y_teste
resultados_da_arvore = clf.predict(X_teste)
precisao = accuracy_score(resultados_corretos, resultados_da_arvore)
print(precisao)

#---> Cuidado com um fenômeno chamado overfitting: a árvore é específica demais para seus dados de treino, em detrimento de seus dados de teste <---#
#---> Para evitar overfitting, existem 3 parâmetros que podem ser alterados: <---#
#---> max_depth(Valores de 1 até 32): Representa quão profunda será a árvore. Árvores muito profundas podem causar overfitting <---#
#---> min_samples_split(Valores inteiros ou float): Quantas amostras são necessárias para se criar um nó de decisão <---#
#---> min_samples_leaf(Valores inteiros ou float): Quantas amostras são necessárias para se criar uma folha. Valores muito altos causam overfitting <---#

# ---------- Parte 7: Representação textual(ou gráfica da árvore)  ---------- #


fig = plt.figure(figsize=(30,10))
_ = plot_tree(clf, filled=True)
plt.show()


