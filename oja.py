import numpy as np
import random
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random

def ruleOja(learnRate, x, y, weight):
	deltaWeight = learnRate * (y * x - pow(y, 2) * weight)
	# print(deltaWeight)	fixme: SE ME ESTA YENDO AL INFINITO
	return deltaWeight


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


def simpleLinearPerceptron(amountOfVariables, data, eta, epsilon, maxIterations):
    i = 0
    weights = []

    for k in range(amountOfVariables):
        n = random.random()
        weights.append(n)

    error = epsilon + 1

    # lastDeltaWeight1 = 0.0f
    # lastDeltaWeight2 = 0.0f
    # lastDeltaWeight3 = 0.0f
    # lastDeltaWeight4 = 0.0f

    while (error > epsilon and i < maxIterations):
        randomIndex = random.randint(0, len(data) - 1)
        randomInput = data[randomIndex]
        output = getOutput(randomInput, weights)
        # localError = expectedOutputs[randomIndex] - output

        for j in range(len(weights)):
            weights[j] = ruleOja(eta, randomInput[j], output, weights[j]) # learnRate, x, y, weight, step

        # weights[0] += eta * localError * randomInput[0] + momentumAlpha * lastDeltaWeight1
        # weights[1] += eta * localError * randomInput[1] + momentumAlpha * lastDeltaWeight2
        # weights[2] += eta * localError * randomInput[2] + momentumAlpha * lastDeltaWeight3
        # weights[8] += eta * localError + momentumAlpha * lastDeltaWeight4

        # lastDeltaWeight1 = eta * localError * randomInput[0]
        # lastDeltaWeight2 = eta * localError * randomInput[1]
        # lastDeltaWeight3 = eta * localError * randomInput[2]
        # lastDeltaWeight4 = eta * localError

        # error = 0.0f
        # # for j in expectedOutputs: fixme
        #     newOutput = (data[j][0] * weights[0]) + (data[j][1] * weights[1]) + (data[j][2] * weights[2]) + weights[3]
        #     error += Math.pow(newOutput - expectedOutputs[j], 2)

        # error *= 0.5f
        i += 1

    return weights

def getOutput(arrayInput, weights):
    output = 0.0
    for i in range(len(arrayInput)):
        output += (arrayInput[i] * weights[i])

    # tengo que agregar este término independiente que no está multiplicado por la variable?
    # output += weights[len(weights) - 1]
    return output


def evaluateNetwork(data, weights, amountOfVariables):
    grid = np.zeros([amountOfVariables])
    clusters = initializeEmptyClusters(amountOfVariables)
    for i in range(0, len(data)):
        winningNeuron = getWinner(weights, getArrayOfData(data[i]), amountOfVariables)
        grid[winningNeuron[0]] += 1
        # clusters[(winningNeuron[0])].append(data[i]['Country'])

    # printClusters(clusters)
    # fig, ax = plt.subplots()
    # ax.imshow(grid)

    # for i in range(0, amountOfVariables):
    #     ax.text(i, grid[i], ha="center", va="center", color="w")

    # fig.tight_layout()
    plt.show()

def initializeEmptyClusters(amountOfVariables):
    emptyMap = {}
    for i in range(0, amountOfVariables):
        emptyMap[(i)] = []
    return emptyMap

def getWinner(weights, entry, amountOfVariables):
    minimun = None
    for i in range(0, amountOfVariables):
        distance = getDistance(entry, weights[i])
        if minimun == None or minimun[1] > distance:
            minimun = [i, distance]
    return [minimun[0], minimun[1]]

def getArrayOfData(entry):
    return np.array([entry['Area'], entry['GDP'], entry['Inflation'], entry['Life.expect'], entry['Military'], entry['Pop.growth'], entry['Unemployment']])

def getDistance(entry, weight):
    aux = entry - weight
    return np.sqrt(sum(aux * aux.T))

def printClusters(clusters):
    print("Clusters: ")
    for item in clusters.items():
        print(item)

def plotUMatrix(weights, amountOfVariables):
    averageDistances = np.zeros([amountOfVariables])
    # for i in range(0, amountOfVariables):
    #     averageDistances[i] = getNeuronNeighborsAverageDistance(weights, i, amountOfVariables)

    # fig, ax = plt.subplots()
    # im = ax.imshow(averageDistances)
    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel("Avg distance", rotation=-90, va="bottom")

    # fig.tight_layout()
    # plt.show()

# For each neuron, we calculate the average weights distance with the 4 closest neighbors neurons
def getNeuronNeighborsAverageDistance(networkWeights, neuronRow, neuronCol, amountOfVariables):
    neuronNeighborsAmount = 0
    weightsDistancesSum = 0
    if neuronCol - 1 >= 0:  #left neighbor
        neuronNeighborsAmount += 1
        weightsDistancesSum += getDistance(networkWeights[neuronRow][neuronCol], networkWeights[neuronRow][neuronCol-1])
    if neuronCol + 1 < amountOfVariables: #right neighbor
        neuronNeighborsAmount += 1
        weightsDistancesSum += getDistance(networkWeights[neuronRow][neuronCol], networkWeights[neuronRow][neuronCol+1])
    if neuronRow - 1 >= 0: #top neighbor
        neuronNeighborsAmount += 1
        weightsDistancesSum += getDistance(networkWeights[neuronRow][neuronCol], networkWeights[neuronRow-1][neuronCol])
    if neuronRow + 1 < amountOfVariables: #bottom neighbor
        neuronNeighborsAmount += 1
        weightsDistancesSum += getDistance(networkWeights[neuronRow][neuronCol], networkWeights[neuronRow+1][neuronCol])

    return weightsDistancesSum / neuronNeighborsAmount


eta = 0.00001
epsilon = 0.00001
maxIterations = 100000
amountOfVariables = 7
standardizedData = standardizeData()
weights = simpleLinearPerceptron(amountOfVariables, standardizedData, eta, epsilon, maxIterations)
evaluateNetwork(standardizedData, weights, amountOfVariables)
plotUMatrix(weights, amountOfVariables)
