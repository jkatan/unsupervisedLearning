#El conjunto de datos tiene 7 variables, que son: "Area","GDP","Inflation","Life.expect","Military","Pop.growth","Unemployment". 
#Tambien tiene 28 registros

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.linalg as la
from sklearn.decomposition import PCA

dataFrame = pd.read_csv('europe.csv')
del dataFrame['Country']

# Sin estandarizar
plt.boxplot((dataFrame['Area'], dataFrame['GDP'], dataFrame['Inflation'], dataFrame['Life.expect'], dataFrame['Military'], dataFrame['Pop.growth'], dataFrame['Unemployment']), notch=False, sym="o", labels=["Area", "GDP","Inflation","Life.expect","Military","Pop.growth","Unemployment"])
plt.show()


# Estandarizando las variables
standardizedDataframe = (dataFrame-dataFrame.mean())/dataFrame.std()
plt.boxplot((standardizedDataframe['Area'], standardizedDataframe['GDP'], standardizedDataframe['Inflation'], standardizedDataframe['Life.expect'], standardizedDataframe['Military'], standardizedDataframe['Pop.growth'], standardizedDataframe['Unemployment']), labels=["Area", "GDP","Inflation","Life.expect","Military","Pop.growth","Unemployment"], notch=False, sym="o")
plt.show()

labels = ["Area", "GDP","Inflation","Life.expect","Military","Pop.growth","Unemployment"]

# Calculando la matrix de covarianza
covarianceDataFrame = dataFrame.cov()
sns.heatmap(covarianceDataFrame, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
plt.show()

# Calculando la matriz de correlacion
correlationDataFrame = dataFrame.corr()
sns.heatmap(correlationDataFrame, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
plt.show()

# Calculando los autovectores y autovalores de las matrices

# De covarianza
covarianceMatrix = covarianceDataFrame.values
covEigenValues, covEigenVectors = la.eig(covarianceMatrix)
print("Autovalores de la matriz de covarianza: ")
print(covEigenValues)
print("\n\n")
print("Autovectores de la matriz de covarianza: ")
print(covEigenVectors.T)
print("\n\n")

# De correlacion
correlationMatrix = correlationDataFrame.values
corrEigenValues, corrEigenVectors = la.eig(correlationMatrix)
print("Autovalores de la matriz de correlacion: ")
print(corrEigenValues)
print("\n\n")
print("Autovectores de la matriz de correlacion: ")
print(corrEigenVectors.T)
print("\n\n")

pca = PCA(n_components=7)
pca.fit_transform(standardizedDataframe)
print("Componentes principales:")
print(pca.components_ )
print("\n\n")

print("Varianza (relativa) de las componentes: ")
print(pca.explained_variance_ratio_)
print("\n")

print("Varianza (absoluta) de las componentes: ")
print(pca.explained_variance_)
print("Vemos que esta varianza corresponde a los autovalores de la matriz de correlacion calculados previamente: ")
print(corrEigenValues)
print("\n\n")

print("Primer componente principal: ")
print(pca.components_[0])
print("Vemos que las cargas corresponden al siguiente autovector de la matriz de correlacion: ")
print(corrEigenVectors.T[0])