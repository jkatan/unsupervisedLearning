import numpy as np
import random
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
import site

def ruleOja(learnRate, x, y, weight):
    return (learnRate * (y * x - pow(y, 2) * np.asarray(weight)))


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


def simplePerceptron(amountOfVariables, data, eta, maxIterations):
    i = 0
    weights = []
    deltaWeights = []
    # error = 1.0
    for k in range(amountOfVariables):
        n = random.random()
        weights.append(n)

    for k in range(amountOfVariables):
        n = random.random()
        deltaWeights.append(n)

    while (deltaWeights[0] > 0 and i < maxIterations):
        randomIndex = random.randint(0, len(data) - 1)
        randomInput = getInput(data, amountOfVariables, randomIndex)
        output = getOutput(randomInput, weights)
        deltaWeights = ruleOja(eta, randomInput, output, weights)
        weights = updateWeights(weights, deltaWeights, eta)
        # error = calculateError(data, expectedOutputs, weights)
        i += 1

    return weights

def updateWeights(weights, deltaWeights, eta):
    newWeights = np.zeros(len(weights))
    for i in range(0, len(deltaWeights)):
        newWeights[i] += weights[i] + deltaWeights[i]

    # The last weight is the bias, so we just sum the eta (equivalent to multiply it by 1)
    newWeights[len(weights) - 1] += eta

    return newWeights

def calculatePerceptronExcitement(input, weights):
    # The last weight is the bias
    result = weights[len(weights) - 1]

    for i in range(0, len(input)):
        result += input[i] * weights[i]

    return result

def getInput(data, amountOfVariables, index):
    # data[randomIndex]
    inputArray = np.zeros(amountOfVariables)
    for i in range(0, amountOfVariables):
        inputArray[i] = data[index][i]
    return inputArray

def getOutput(arrayInput, weights):
    output = 0.0

    for i in range(0, len(arrayInput)):
        output += (arrayInput[i] * weights[i])

    # tengo que agregar este término independiente que no está multiplicado por la variable?
    # output[len(weights) - 1] += weights[len(weights) - 1]

    return output



eta = 0.00001
epsilon = 0.00001
maxIterations = 10000
amountOfVariables = 7
standardizedData = standardizeData()
weights = simplePerceptron(amountOfVariables, standardizedData, eta, maxIterations)
# weights = simpleLinearPerceptron(amountOfVariables, standardizedData, eta, epsilon, maxIterations)
# evaluateNetwork(standardizedData, weights, amountOfVariables)
# plotUMatrix(weights, amountOfVariables)
print("Las cargas dan...")
print(weights)
