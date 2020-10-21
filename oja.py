import numpy as np
import random
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
import site

def ruleOja(learnRate, inputVector, output, weights):
    deltaWeights = np.zeros(len(weights))
    for i in range(0, len(inputVector)):
        deltaWeights[i] = learnRate * (output * inputVector[i] - pow(output, 2) * weights[i])

    return deltaWeights


def standardizeData():
    dataFrame = pd.read_csv('europe.csv')
    del dataFrame['Country']

    dataFrame['Area'] = dataFrame['Area'].astype(float)
    dataFrame['GDP'] = dataFrame['GDP'].astype(float)
    data = dataFrame.to_records(index=False)

    standardizeVariable('Area', data)
    standardizeVariable('GDP', data)
    standardizeVariable('Inflation', data)
    standardizeVariable('Life.expect', data)
    standardizeVariable('Military', data)
    standardizeVariable('Pop.growth', data)
    standardizeVariable('Unemployment', data)

    return data

def standardizeVariable(varName, data):
    varMean = getVariableMean(data[varName])
    varStd = getVariableStd(data[varName], varMean)
    for i in range(0, len(data[varName])):
        data[varName][i] = (data[varName][i] - varMean) / varStd

def getVariableMean(variableData):
    sum = 0
    for i in range(0, len(variableData)):
        sum += variableData[i]
    return sum/len(variableData)

def getVariableStd(variableData, variableMean):
    squaredVariance = getVariableSquaredVariance(variableData, variableMean)
    return math.sqrt(squaredVariance)

def getVariableSquaredVariance(variableData, variableMean):
    sum = 0
    for i in range(0, len(variableData)):
        sum += pow(variableData[i] - variableMean, 2)
    return float(sum)/(len(variableData) - 1)


def simplePerceptron(amountOfVariables, data, initialEta, maxEpochs):
    weights = np.zeros(amountOfVariables)
    for k in range(0, len(weights) - 1):
        weights[k] = random.random()

    epochs = 0
    eta = initialEta
    while (epochs < maxEpochs):
        for i in range(0, len(data)):
            randomInput = getRandomInput(data, amountOfVariables)
            output = getOutput(randomInput, weights)
            deltaWeights = ruleOja(eta, randomInput, output, weights)
            updateWeights(weights, deltaWeights)

        epochs += 1
        eta = eta / (epochs + 1)
        
    return weights

def updateWeights(weights, deltaWeights):
    for i in range(0, len(weights)):
        weights[i] += deltaWeights[i]

def getRandomInput(data, amountOfVariables):
    index = random.randint(0, len(data) - 1)
    inputArray = np.zeros(amountOfVariables)
    for i in range(0, amountOfVariables):
        inputArray[i] = data[index][i]
    return inputArray

def getOutput(arrayInput, weights):
    output = 0.0
    for i in range(0, len(arrayInput)):
        output += (arrayInput[i] * weights[i])

    return output

eta = 0.13
epochs = 500
amountOfVariables = 7
standardizedData = standardizeData()
weights = simplePerceptron(amountOfVariables, standardizedData, eta, epochs)
print("Las cargas dan...")
print(weights)

# Las cargas originales obtenidas usando una libreria son:
#[ 0.1248739  -0.50050586  0.40651815 -0.48287333  0.18811162 -0.47570355
#  0.27165582]

#Usando etaInicial=0.1 y epochs=20 con la regla de Oja se obtuvo:
#[ 0.19408458 -0.54898837  0.40332293 -0.4517502   0.19758084 -0.4797626
#  0.25261652]

#Usando etaInicial=0.1 y epochs=100 con la regla de Oja se obtuvo:
#[ 0.14337052 -0.48813356  0.46874206 -0.49543837  0.14530174 -0.45747126
#  0.25487497]

#Usando etaInicial=0.15 y epochs=500 con la regla de Oja se obtuvo:
#[ 0.19045154 -0.47182641  0.46641092 -0.50660133  0.1669918  -0.46618425
#  0.24594691]