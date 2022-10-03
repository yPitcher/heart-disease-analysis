import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame as df
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, classification_report

# *****************DADOS ESTUDADOS COM BASE NO CORPO DE UM ADULTO (+18)***************************
# pegando os dados do csv, trocando brancos por NaN (Not a Number)
df = pd.read_csv("heart.csv", na_values=' ')

print(df.head())
print(df.describe())

# Fazendo uma cópia temporária
df2 = df.copy()

# removendo os NaN (Not a Number)
df2.dropna(inplace=True)
print(df2.isna().sum())

# média e mediana da pressão sanguínea sem os NaN
media_pressaos_sem_nan = df2['trtbps'].mean()
mediana_pressaos_sem_nan = df2['trtbps'].median()
print("Pressão sanguínea;\n {} {}".format(media_pressaos_sem_nan, mediana_pressaos_sem_nan))

# média e mediana do colesterol sem os NaN
media_cole_sem_nan = df2['chol'].mean()
mediana_cole_sem_nan = df2['chol'].median()
print("Colesterol;\n {} {}".format(media_cole_sem_nan, mediana_cole_sem_nan))

# ajustando o df com mediana // poderia tbem ser a media
df['trtbps'].fillna(mediana_pressaos_sem_nan, inplace=True)
df['chol'].fillna(mediana_cole_sem_nan, inplace=True)

# mediana da pressão sanguínea e Colesterol apos a substituicao dos "faltantes"
mediana_pressao = df['trtbps'].median()
mediana_colest = df['chol'].median()
print(mediana_pressao, mediana_colest)

# moda da idade e sexo sem NaN --default
moda_t_dor_sem_nan = df['cp'].mode()[0]
moda_sexo_sem_nan = df['sex'].mode()[0]
print(moda_t_dor_sem_nan, moda_sexo_sem_nan)

# ajustando o df com a moda
df['cp'].fillna(moda_t_dor_sem_nan, inplace=True)
df['sex'].fillna(moda_sexo_sem_nan, inplace=True)

#Não foi necessário retirar os outliers, pois neste dataset não há divergências no valor determinante. (somente 0 e 1)

# TIPO DE DOR NO PEITO e SEXO
# trabalhar os atributos categóricos
# dor no peito
dor = LabelEncoder()
tipos_dor = dor.fit_transform(df['cp'])
mapa_dor = {index: label for index, label in enumerate(dor.classes_)}
print(mapa_dor)
df['Cod_dor'] = tipos_dor

# Sexo
sexo = LabelEncoder()
tipos_sexo = sexo.fit_transform(df['sex'])
mapa_sexo = {index: label for index, label in enumerate(sexo.classes_)}
print(mapa_sexo)
df['Cod_sexo'] = tipos_sexo

# criar classificação de acordo com a pressão sanguínea (Sistólica)
df_pressao = pd.DataFrame([[0, 'Normal', 0.0, 129.99],
                           [1, 'Normal - Limítrofe', 130.00, 139.99],
                           [2, 'Hipertensão Leve', 140.00, 159.99],
                           [3, 'Hipertensão Moderada', 160.00, 179.99],
                           [4, 'Hipertensão Grave', 180, 200]],
                          index=range(0, 5),
                          columns=['Classe', 'Descricao', 'De', 'Ate'])
print(df_pressao)

df['Classe_pressao'] = -1
for i_df, r_df in df.iterrows():
    for i_df_pressao, r_df_pressao in df_pressao.iterrows():
        if (r_df['trtbps'] >= r_df_pressao['De']) and (r_df['trtbps'] <= r_df_pressao['Ate']):
            df.at[i_df, 'Classe_pressao'] = df_pressao.at[i_df_pressao, 'Classe']

# removendo colunas "desnecessárias" do df
df.drop(columns=['cp', 'sex'], inplace=True)
print(df)

# *****************************************ÁRVORES DE DECISÃO************************************************#
# criar um df só pra arvore de decisão
cols_X = df[["trtbps", "chol"]]
cols_y = df[['Classe_pressao']]
df_X = cols_X.copy()
df_y = cols_y.copy()
print(df_X)
print(df_y)

#separando o ds em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(df_X, df_y, test_size=0.8, random_state=42)

#parâmetros básicos
clf_arvore = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)

#fase de treinamento
clf_arvore = clf_arvore.fit(X_treino, y_treino)
y_predicao_treino = clf_arvore.predict(X_treino)
print("Acuracidade no treino com DT:", accuracy_score(y_treino, y_predicao_treino))
print("Precisão no treino com DT:", precision_score(y_treino, y_predicao_treino, average='weighted', zero_division=0))

#Matriz de correlação;
plt.matshow(df.corr())
plt.show()

#matriz de confusão - para treino
print(classification_report(y_treino, y_predicao_treino, zero_division=0))

#fase de testes
y_predicao_teste = clf_arvore.predict(X_teste)

#resultado nos testes
print("Acuracidade no teste com DT:", accuracy_score(y_teste, y_predicao_teste))
print("Precisão no teste com DT:", precision_score(y_teste, y_predicao_teste, average='weighted', zero_division=0))

tree.plot_tree(clf_arvore, fontsize=8)
plt.show()
