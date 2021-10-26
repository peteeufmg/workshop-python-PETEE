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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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

# ---------- Parte 2: Explicar o problema e a nossa base de dados  ---------- #

#---> Mostrar o Slide 3.1: Dataset Titanic <---#

# ---------- Parte 3: Ler o arquivo para treino; funções básicas Pandas  ---------- #

#---> Explicar o que é um arquivo csv <---#

dados_para_treino = pd.read_csv("train.csv")

print(dados_para_treino)
#print(dados_para_treino.head())
#print(dados_para_treino.tail())
#print(dados_para_treino.tail(10))
#print(dados_para_treino.describe())

#X_treino = dados_para_treino[["Sex","Age","Pclass"]]
#print(X_treino)
X_treino = dados_para_treino[["Pclass","Sex","Age"]]
#print(X_treino)

y_treino = dados_para_treino["Survived"]
#print(y_treino)

X_treino.loc[:,"Sex"] = X_treino["Sex"].map( {"male":0,"female":1} ).astype(int)
#print(X_treino)


dados_para_teste = pd.read_csv("test.csv")
X_teste = dados_para_teste[["Pclass","Sex","Age"]]

X_treino["Age"].fillna(X_treino["Age"].median(), inplace = True)


#X_teste.loc[:,"Sex"] = X_teste["Sex"].map( {"male":0,"female":1} ).astype(int)
#print(X_teste)
#X_teste["Age"].fillna(X_teste["Age"].median(), inplace = True) # Substituindo valores NaN
#print(X_teste)


X_treino, X_teste, y_treino, y_teste = train_test_split(X_treino, y_treino, test_size = 0.3)

clf = DecisionTreeClassifier(criterion="gini")
clf.fit(X_treino,y_treino)

resultados_corretos = y_teste
resultados_do_modelo = clf.predict(X_teste)
precisao = accuracy_score(resultados_corretos, resultados_do_modelo)
print(precisao)