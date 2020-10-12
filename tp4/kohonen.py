import numpy as np
import random
from math import exp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

quantityOfVariables = 7

def trainKohonen(data, maxEphocs, col, row):
    weights = initKohonen(data, col, row)
    Rinit = max([col, row]) / 2
    ephocs = 0
    while ephocs < maxEphocs:
        randomEntry = getArrayOfData(getRandomEntry(data)) 
        winner = getWinner(weights, randomEntry, col, row)
        eta = 0.2 * (1 - ephocs / maxEphocs)
        R = ((maxEphocs - ephocs * 1.2) * Rinit / maxEphocs) + 1
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

def evaluateNetwork():
    data = pd.read_csv('europe.csv').to_records(index=False)
    rows = 6
    cols = 6
    weights = trainKohonen(data, 25000, cols, rows)
    grid = np.zeros([rows, cols])
    for i in range(0, len(data)):
        winningNeuron = getWinner(weights, getArrayOfData(data[i]), cols, rows)
        grid[winningNeuron[0]][winningNeuron[1]] += 1
    
    fig, ax = plt.subplots()
    ax.imshow(grid)

    for i in range(0, rows):
        for j in range(0, cols):
            ax.text(j, i, grid[i, j], ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()

evaluateNetwork()