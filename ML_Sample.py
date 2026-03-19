"""

Exemplo de Exploração de Dados 

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets

# Dados de exemplo
alunos = pd.DataFrame({
    'nome': ['João', 'Maria', 'Pedro', 'Ana', 'Carlos', 'Juliana', 'Felipe', 'Rafaela', 'Daniel', 'Lucas'],
    'nota1': np.random.randint(70, 101, 10),
    'nota2': np.random.randint(70, 101, 10),
    'nota3': np.random.randint(70, 101, 10),
    'media': np.random.uniform(70, 90, 10),
    'situação': np.random.choice(['Aprovado', 'Reprovado', 'Recuperação'], 10)
    'idade': np.random.randint(18, 30, 10)
})

# Intervalo (máximo - mínimo)

Intervalo = alunos ['idade'].max() - alunos ['idade'].min()

print(f"Intervalo: {Intervalo}")

# Variância

variancia = alunos ['idade'].var()

print(f"Variância: {variancia}")

# Desvio Padrão

desvio_padrao = alunos ['idade'].std()

print(f"Desvio Padrão: {desvio_padrao}")

# Coeficiente de Variação

coeficiente_variacao = alunos ['idade'].std() / alunos ['idade'].mean()

print(f"Coeficiente de Variação: {coeficiente_variacao}")

# Histograma 

plt.hist(alunos ['idade'], bins=10)
plt.title('Histograma de Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

sns.histplot(alunos['idade'], bins=10)

print(f"Assimetria: {alunos.idade.skew()}")
print(f"Curtosidade: {alunos.idade.kurtosis()}")
print(alunos.describe())

# Box Plot

sns.catplot(data=alunos, y ='idade', kind='box')
plt.tight_layout()
plt.show()

iris = datasets.load_iris()
print(iris.DESCR)

df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
print(df.head())

print(df.cov())

print(df.corr())

fig, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=(10, 5))
fig.suptitle('Gráficos de Dispersão')
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', ax=ax1)
sns.scatterplot(data=df, x='sepal_length', y='petal_length', ax=ax2)
sns.scatterplot(data=df, x='sepal_length', y='petal_width', ax=ax3)
sns.scatterplot(data=df, x='sepal_width', y='petal_length', ax=ax4)
sns.scatterplot(data=df, x='sepal_width', y='petal_width', ax=ax5)
sns.scatterplot(data=df, x='petal_length', y='petal_width', ax=ax6)
plt.tight_layout()
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))

sns.boxplot(data=df, x='sepal_length', y='sepal_width', ax=ax1)
sns.boxplot(data=df, x='sepal_length', y='petal_length', ax=ax2)
sns.boxplot(data=df, x='sepal_length', y='petal_width', ax=ax3)
sns.boxplot(data=df, x='sepal_width', y='petal_length', ax=ax4)
plt.tight_layout()
plt.show()
