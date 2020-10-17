import numpy as np
import random
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

quantityOfVariables = 7

def trainKohonen(data, maxEphocs, col, row):
    weights = initKohonen(data, col, row)
    Rinit = max([col, row]) / 2
    ephocs = 1
    while ephocs <= maxEphocs:
        randomEntry = getArrayOfData(getRandomEntry(data)) 
        winner = getWinner(weights, randomEntry, col, row)
        eta = 1 / epochs
        R = ((maxEphocs - ephocs) * Rinit) / maxEphocs
        weights = updateWeigthNeighbours(weights, randomEntry, col, row, winner, eta, R)
        ephocs += 1
        
    return weights
        
def initKohonen(data, col, row):
    weights = np.zeros([col, row, quantityOfVariables])
    for i in range(0, col):
        for j in range(0, row):
            randomEntry = getArrayOfData(getRandomEntry(data)) 
            for k in range(0, quantityOfVariables):
                weights[i][j][k] = randomEntry[k]
    return weights;

def getWinner(weights, entry, col, row):
    minimun = None
    for i in range(0, col):
        for j in range(0, row):
            distance = getDistance(entry, weights[i][j])
            if minimun == None or minimun[2] > distance:
                minimun = [i, j, distance]
    return [minimun[0], minimun[1]]

def updateWeigthNeighbours(weights, entry, col, row, winner, eta, R):
    for i in range(0, col):
        for j in range(0, row):
            distance = getDistance(winner, np.array([i, j]))
            if distance < R:
                weights[i][j] = weights[i][j] + eta * (entry - weights[i][j])
    return weights

def getArrayOfData(entry):
    return np.array([entry['Area'], entry['GDP'], entry['Inflation'], entry['Life.expect'], entry['Military'], entry['Pop.growth'], entry['Unemployment']])

def getRandomEntry(data):
    return data[random.randint(0, len(data) - 1)]

def getDistance(entry, weight):
    aux = entry - weight
    return np.sqrt(sum(aux * aux.T))

def initializeEmptyClusters(rows, cols):
    emptyMap = {}
    for i in range(0, rows):
        for j in range(0, cols):
            emptyMap[(i, j)] = []
    return emptyMap

def printClusters(clusters):
    print("Clusters: ")
    for item in clusters.items():
        print(item)

def plotUMatrix(weights, rows, cols):
    averageDistances = np.zeros([rows, cols])
    for i in range(0, rows):
        for j in range(0, cols):
            averageDistances[i][j] = getNeuronNeighborsAverageDistance(weights, i, j, rows, cols)

    fig, ax = plt.subplots()
    im = ax.imshow(averageDistances)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Avg distance", rotation=-90, va="bottom")

    fig.tight_layout()
    plt.show()


# For each neuron, we calculate the average weights distance with the 4 closest neighbors neurons
def getNeuronNeighborsAverageDistance(networkWeights, neuronRow, neuronCol, rows, cols):
    neuronNeighborsAmount = 0
    weightsDistancesSum = 0
    if neuronCol - 1 >= 0:  #left neighbor
        neuronNeighborsAmount += 1
        weightsDistancesSum += getDistance(networkWeights[neuronRow][neuronCol], networkWeights[neuronRow][neuronCol-1])
    if neuronCol + 1 < cols: #right neighbor
        neuronNeighborsAmount += 1
        weightsDistancesSum += getDistance(networkWeights[neuronRow][neuronCol], networkWeights[neuronRow][neuronCol+1])
    if neuronRow - 1 >= 0: #top neighbor
        neuronNeighborsAmount += 1
        weightsDistancesSum += getDistance(networkWeights[neuronRow][neuronCol], networkWeights[neuronRow-1][neuronCol])
    if neuronRow + 1 < rows: #bottom neighbor
        neuronNeighborsAmount += 1
        weightsDistancesSum += getDistance(networkWeights[neuronRow][neuronCol], networkWeights[neuronRow+1][neuronCol])

    return weightsDistancesSum / neuronNeighborsAmount

def standardizeData():
    dataFrame = pd.read_csv('europe.csv')

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

def getVariableSquaredVariance(variableData, variableMean):
    sum = 0
    for i in range(0, len(variableData)):
        sum += pow(variableData[i] - variableMean, 2)
    return float(sum)/(len(variableData) - 1)

def getVariableStd(variableData, variableMean):
    squaredVariance = getVariableSquaredVariance(variableData, variableMean)
    return math.sqrt(squaredVariance)

def evaluateNetwork(data, weights, rows, cols):
    grid = np.zeros([rows, cols])
    clusters = initializeEmptyClusters(rows, cols)
    for i in range(0, len(data)):
        winningNeuron = getWinner(weights, getArrayOfData(data[i]), cols, rows)
        grid[winningNeuron[0]][winningNeuron[1]] += 1
        clusters[(winningNeuron[0], winningNeuron[1])].append(data[i]['Country'])

    printClusters(clusters)
    fig, ax = plt.subplots()
    ax.imshow(grid)

    for i in range(0, rows):
        for j in range(0, cols):
            ax.text(j, i, grid[i, j], ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()

rows = 6
cols = 6
epochs = 20000
standardizedData = standardizeData()
weights = trainKohonen(standardizedData, epochs, cols, rows)
evaluateNetwork(standardizedData, weights, rows, cols)
plotUMatrix(weights, rows, cols)