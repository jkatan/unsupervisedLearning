import numpy as np
import random
import sys

class hopfieldNetwork:
	def __init__(self, patterns):
		self.weights = self.calculateWeights(patterns)

	def calculateWeights(self, patterns):
		patternsDimension = len(patterns[0])
		K = np.matrix(patterns).transpose()
		K_t = K.transpose()
		W = np.matmul(K, K_t) * (1.0/patternsDimension)
		return self.cleanWeightsDiagonal(W)
		return W

	def cleanWeightsDiagonal(self, weights):
		for i in range(0, weights[0].size):
			weights[i,i] = 0
		return weights

	def predict(self, samplePattern):
		step = 0
		currentState = self.sign(np.matmul(self.weights, samplePattern))
		drawImage(np.squeeze(np.asarray(currentState)), "Current step: " + str(step))
		step += 1
		while True:
			newState = self.sign(np.matmul(self.weights, np.squeeze(np.asarray(currentState))))
			drawImage(np.squeeze(np.asarray(newState)), "Current step: " + str(step))
			if np.array_equal(newState, currentState) == True:
				print("Stabilized")
				return newState
			currentState = newState
			step += 1

	def sign(self, matrix):
		dim = matrix[0].size
		for i in range(0, dim):
			if matrix[0,i] > 0:
				matrix[0,i] = 1
			if matrix[0,i] < 0:
				matrix[0,i] = -1
		return matrix


def drawImage(image, title):
	print(title)
	for i in range(0, 25):
		if i % 5 == 0:
			print()
		if image[i] == 1:
			print("*", end='')
		if image[i] == -1:
			print(" ", end='')
	print("\n")

def flip(number):
	return number*-1

def addNoise(array, amountToRandomize):
	for i in range(0, amountToRandomize):
		rand = random.randrange(0, len(array), 1)
		array[rand] = flip(array[rand])
	return array

def predictNoisyLetter(hopfieldNetwork, letter, noiseToAdd):
	noise = addNoise(letter, noiseToAdd)
	drawImage(noise, "Letter with noise to predict")
	hopfieldNetwork.predict(noise)

def processLetterToPredict(string, letters):
	if string == "S":
		return letters[0]
	if string == "B":
		return letters[1]
	if string == "A":
		return letters[2]
	if string == "L":
		return letters[3]

def displayDotProductInformation(letters):
	S = letters[0]
	B = letters[1]
	A = letters[2]
	L = letters[3]

	print("Letters dot product information")

	SB = S.dot(B)
	SA = S.dot(A)
	SL = S.dot(L)
	print("SB: " + str(SB))
	print("SA: " + str(SA))
	print("SL: " + str(SL))

	BA = B.dot(A)
	BL = B.dot(L)
	print("BA: " + str(BA))
	print("BL: " + str(BL))

	AL = A.dot(L)
	print("AL: " + str(AL))

def startDemo():
	S = np.array([
	    1, 1, 1, 1, 1,
	    1, -1, -1, -1, -1,
	    1, 1, 1, 1, 1,
	    -1, -1, -1, -1, 1,
	    1, 1, 1, 1, 1,
	])

	B = np.array([
	    1, 1, 1, -1, -1,
	    1, -1, -1, 1, -1,
	    1, -1, 1, 1, -1,
	    1, -1, -1, 1, -1,
	    1, 1, 1, -1, -1,
	])

	A = np.array([
	    -1, -1, 1, -1, -1,
	    -1, 1, -1, 1, -1,
	    1, -1, 1, -1, 1,
	    1, -1, -1, -1, 1,
	    1, -1, -1, -1, 1,
	])

	L = np.array([
	    1, -1, -1, -1, -1,
	    1, -1, -1, -1, -1,
	    1, -1, -1, -1, -1,
	    1, -1, -1, -1, -1,
	    1, 1, 1, 1, -1,
	])

	letters = [S, B, A, L]
	print("Letters to use")
	drawImage(S, "S")
	drawImage(B, "B")
	drawImage(A, "A")
	drawImage(L, "L")

	displayDotProductInformation(letters)
	print("\n")

	letterToPredict = processLetterToPredict(sys.argv[1], letters)
	drawImage(letterToPredict, "Original letter:")

	noiseToAdd = int(sys.argv[2])
	print("Adding random noise to " + sys.argv[2] + " elements of the letter")

	network = hopfieldNetwork(letters)
	predictNoisyLetter(network, letterToPredict, noiseToAdd)

startDemo()