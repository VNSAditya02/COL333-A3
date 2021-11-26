# Name Classes as CamelCase format and functions/variables as camelCase (first letter small). 
# Constants are named as FIRST_LAST

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import sys

class State:
	def __init__(self, taxiPos, passPos, passDest):
		self.taxiPos = taxiPos
		self.passPos = passPos
		self.passDest = passDest

		self.northState = None
		self.southState = None
		self.eastState = None
		self.westState = None
		self.pickUpState = None
		self.putDownState = None
		self.pickUpReward = None
		self.putDownReward = None

		self.value = 0
		self.qValues = [0, 0, 0, 0, 0, 0]
		self.action = None
		self.isExit = False

depots = [(0, 0), (4, 0), (0, 3), (4, 4)]
depots10X10 = [(1, 0), (9, 0), (6, 3), (0, 4), (9, 5), (5, 6), (9, 8), (0, 9)]
validActions = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'PICK_UP', 'PUT_DOWN']

def generateStates():
	# (-1, -1) denotes that passenger is in Taxi
	passDepots = [(-1, -1)]
	for i in range(5):
		for j in range(5):
			passDepots.append((i, j))
	states = {}
	states[((-10, -10), (-10, -10), (-10, -10))] = State((-10, -10), (-10, -10), (-10, -10))
	states[((-10, -10), (-10, -10), (-10, -10))].isExit = True
	for taxiX in range(5):
		for taxiY in range(5):
			for source in range(26):
				for dest in range(4):
					taxiPos = (taxiX, taxiY)
					passPos = passDepots[source]
					passDest = depots[dest]
					curState = State(taxiPos, passPos, passDest)
					states[(taxiPos, passPos, passDest)] = curState

	for taxiX in range(5):
		for taxiY in range(5):
			for source in range(26):
				for dest in range(4):
					taxiPos = (taxiX, taxiY)
					passPos = passDepots[source]
					passDest = depots[dest]
					curState = states[(taxiPos, passPos, passDest)]
					if(taxiX == 4):
						curState.northState = curState
					else:
						curState.northState = states[((taxiX + 1, taxiY), passPos, passDest)]
					if(taxiX == 0):
						curState.southState = curState
					else:
						curState.southState = states[((taxiX - 1, taxiY), passPos, passDest)]
					if(taxiY == 4):
						curState.eastState = curState
					else:
						curState.eastState = states[((taxiX, taxiY + 1), passPos, passDest)]
					if(taxiY == 0):
						curState.westState = curState
					else:
						curState.westState = states[((taxiX, taxiY - 1), passPos, passDest)]
					if(taxiPos == passPos):
						curState.pickUpState = states[(taxiPos, (-1, -1), passDest)]
						curState.pickUpReward = -1
					elif(passPos == (-1, -1)):
						curState.pickUpState = curState
						curState.pickUpReward = -1
					else:
						curState.pickUpState = curState
						curState.pickUpReward = -10
					if(passPos == (-1, -1) and taxiPos == passDest):
						curState.putDownState = states[((-10, -10), (-10, -10), (-10, -10))]
						curState.putDownReward = 20
					elif(passPos == (-1, -1) or taxiPos == passPos):
						curState.putDownState = states[(taxiPos, taxiPos, passDest)]
						curState.putDownReward = -1
					else:
						curState.putDownState = curState
						curState.putDownReward = -10
					if(taxiY == 0 and (taxiX == 0 or taxiX == 1)):
						curState.eastState = curState
					if(taxiY == 1 and (taxiX == 0 or taxiX == 1)):
						curState.westState = curState
					if(taxiY == 1 and (taxiX == 3 or taxiX == 4)):
						curState.eastState = curState
					if(taxiY == 2 and (taxiX == 3 or taxiX == 4)):
						curState.westState = curState
					if(taxiY == 2 and (taxiX == 0 or taxiX == 1)):
						curState.eastState = curState
					if(taxiY == 3 and (taxiX == 0 or taxiX == 1)):
						curState.westState = curState
	return states

def generateStates10X10():
	# (-1, -1) denotes that passenger is in Taxi
	passDepots = [(-1, -1)]
	for i in range(10):
		for j in range(10):
			passDepots.append((i, j))
	states = {}
	states[((-10, -10), (-10, -10), (-10, -10))] = State((-10, -10), (-10, -10), (-10, -10))
	states[((-10, -10), (-10, -10), (-10, -10))].isExit = True
	for taxiX in range(10):
		for taxiY in range(10):
			for source in range(101):
				for dest in range(8):
					taxiPos = (taxiX, taxiY)
					passPos = passDepots[source]
					passDest = depots10X10[dest]
					curState = State(taxiPos, passPos, passDest)
					states[(taxiPos, passPos, passDest)] = curState

	for taxiX in range(10):
		for taxiY in range(10):
			for source in range(101):
				for dest in range(8):
					taxiPos = (taxiX, taxiY)
					passPos = passDepots[source]
					passDest = depots10X10[dest]
					curState = states[(taxiPos, passPos, passDest)]
					if(taxiX == 9):
						curState.northState = curState
					else:
						curState.northState = states[((taxiX + 1, taxiY), passPos, passDest)]
					if(taxiX == 0):
						curState.southState = curState
					else:
						curState.southState = states[((taxiX - 1, taxiY), passPos, passDest)]
					if(taxiY == 9):
						curState.eastState = curState
					else:
						curState.eastState = states[((taxiX, taxiY + 1), passPos, passDest)]
					if(taxiY == 0):
						curState.westState = curState
					else:
						curState.westState = states[((taxiX, taxiY - 1), passPos, passDest)]
					if(taxiPos == passPos):
						curState.pickUpState = states[(taxiPos, (-1, -1), passDest)]
						curState.pickUpReward = -1
					elif(passPos == (-1, -1)):
						curState.pickUpState = curState
						curState.pickUpReward = -1
					else:
						curState.pickUpState = curState
						curState.pickUpReward = -10
					if(passPos == (-1, -1) and taxiPos == passDest):
						curState.putDownState = states[((-10, -10), (-10, -10), (-10, -10))]
						curState.putDownReward = 20
					elif(passPos == (-1, -1) or taxiPos == passPos):
						curState.putDownState = states[(taxiPos, taxiPos, passDest)]
						curState.putDownReward = -1
					else:
						curState.putDownState = curState
						curState.putDownReward = -10
					if(taxiY == 0 and (taxiX == 0 or taxiX == 1 or taxiX == 2 or taxiX == 3)):
						curState.eastState = curState
					if(taxiY == 1 and (taxiX == 0 or taxiX == 1 or taxiX == 2 or taxiX == 3)):
						curState.westState = curState
					if(taxiY == 2 and (taxiX == 6 or taxiX == 7 or taxiX == 8 or taxiX == 9)):
						curState.eastState = curState
					if(taxiY == 3 and (taxiX == 6 or taxiX == 7 or taxiX == 8 or taxiX == 9)):
						curState.westState = curState
					if(taxiY == 3 and (taxiX == 0 or taxiX == 1 or taxiX == 2 or taxiX == 3)):
						curState.eastState = curState
					if(taxiY == 4 and (taxiX == 0 or taxiX == 1 or taxiX == 2 or taxiX == 3)):
						curState.westState = curState
					if(taxiY == 5 and (taxiX == 4 or taxiX == 5 or taxiX == 6 or taxiX == 7)):
						curState.eastState = curState
					if(taxiY == 6 and (taxiX == 4 or taxiX == 5 or taxiX == 6 or taxiX == 7)):
						curState.westState = curState
					if(taxiY == 7 and (taxiX != 4 and taxiX != 5)):
						curState.eastState = curState
					if(taxiY == 8 and (taxiX != 4 and taxiX != 5)):
						curState.westState = curState
	return states

def simulator(curState, curAction):
	if(curAction == 'NORTH'):
		nextState = np.random.choice([curState.northState, curState.southState, curState.eastState, curState.westState], p = [0.85, 0.05, 0.05, 0.05])
		reward = -1
	elif(curAction == 'SOUTH'):
		nextState = np.random.choice([curState.northState, curState.southState, curState.eastState, curState.westState], p = [0.05, 0.85, 0.05, 0.05])
		reward = -1
	elif(curAction == 'EAST'):
		nextState = np.random.choice([curState.northState, curState.southState, curState.eastState, curState.westState], p = [0.05, 0.05, 0.85, 0.05])
		reward = -1
	elif(curAction == 'WEST'):
		nextState = np.random.choice([curState.northState, curState.southState, curState.eastState, curState.westState], p = [0.05, 0.05, 0.05, 0.85])
		reward = -1
	elif(curAction == 'PICK_UP'):
		nextState = curState.pickUpState
		reward = curState.pickUpReward
	else:
		nextState = curState.putDownState
		reward = curState.putDownReward
	return nextState, reward

def bellmanUpdate(states, gamma):
	for keys in states:
		curState = states[keys]
		if(curState.isExit):
			continue
		northVal = 0.85*(-1 + gamma*curState.northState.value) + 0.05*(-1 + gamma*curState.southState.value) + 0.05*(-1 + gamma*curState.eastState.value) + 0.05*(-1 + gamma*curState.westState.value)
		southVal = 0.05*(-1 + gamma*curState.northState.value) + 0.85*(-1 + gamma*curState.southState.value) + 0.05*(-1 + gamma*curState.eastState.value) + 0.05*(-1 + gamma*curState.westState.value)
		eastVal = 0.05*(-1 + gamma*curState.northState.value) + 0.05*(-1 + gamma*curState.southState.value) + 0.85*(-1 + gamma*curState.eastState.value) + 0.05*(-1 + gamma*curState.westState.value)
		westVal = 0.05*(-1 + gamma*curState.northState.value) + 0.05*(-1 + gamma*curState.southState.value) + 0.05*(-1 + gamma*curState.eastState.value) + 0.85*(-1 + gamma*curState.westState.value)
		pickUpVal = (curState.pickUpReward + gamma*curState.pickUpState.value)
		putDownVal = (curState.putDownReward + gamma*curState.putDownState.value)
		qValues = [northVal, southVal, eastVal, westVal, pickUpVal, putDownVal]
		curState.qValues = qValues

def isConverged(states, epsilon):
	maxValue = 0
	for keys in states:
		curState = states[keys]
		maxValue = max(maxValue, abs(curState.value - max(curState.qValues)))
	return maxValue < epsilon

def maxNorm(states):
	maxValue = 0
	for keys in states:
		curState = states[keys]
		maxValue = max(maxValue, abs(curState.value - max(curState.qValues)))
	return maxValue

def valueIteration(states, epsilon, gamma):
	itr = 0
	maxNormList = []
	while(itr == 0 or maxNorm(states) > epsilon):
		for keys in states:
			curState = states[keys]
			curState.value = max(curState.qValues)
		bellmanUpdate(states, gamma)
		maxNormList.append(maxNorm(states))
		itr += 1
	for keys in states:
		curState = states[keys]
		curState.action = validActions[np.argmax(curState.qValues)]
	print("No of iterations: " + str(itr))
	return maxNormList

def isSamePolicy(states):
	for keys in states:
		curState = states[keys]
		if(curState.action != validActions[np.argmax(curState.qValues)]):
			return False
	return True

def policyLoss(s1, s2):
	maxValue = 0
	for i in range(len(s1)):
		maxValue = max(maxValue, abs(s1[i] - s2[i]))
	return maxValue

def policyIteration(states, gamma):
	itr = 0
	utilities = []
	for keys in states:
		curState = states[keys]
		curState.action = 'NORTH'
	while(itr == 0 or isSamePolicy(states) == False):
		uIter = []
		for keys in states:
			curState = states[keys]
			curState.action = validActions[np.argmax(curState.qValues)]
		bellmanUpdate(states, gamma)
		for keys in states:
			curState = states[keys]
			curState.value = curState.qValues[validActions.index(curState.action)]
			uIter.append(curState.value)
		bellmanUpdate(states, gamma)
		utilities.append(uIter)
		itr += 1
	print("No of iterations: " + str(itr))
	return utilities

def getTransition(s, sPrime):
	neighbour = [s.northState == sPrime, s.southState == sPrime, s.eastState == sPrime, s.westState == sPrime, s.pickUpState == sPrime, s.putDownState == sPrime]
	if(s.action == 'NORTH'):
		t = 0.85*neighbour[0] + 0.05*neighbour[1] + 0.05*neighbour[2] + 0.05*neighbour[3]
		return t, -1*t
	if(s.action == 'SOUTH'):
		t = 0.05*neighbour[0] + 0.85*neighbour[1] + 0.05*neighbour[2] + 0.05*neighbour[3]
		return t, -1*t
	if(s.action == 'EAST'):
		t = 0.05*neighbour[0] + 0.05*neighbour[1] + 0.85*neighbour[2] + 0.05*neighbour[3]
		return t, -1*t
	if(s.action == 'WEST'):
		t = 0.05*neighbour[0] + 0.05*neighbour[1] + 0.05*neighbour[2] + 0.85*neighbour[3]
		return t, -1*t
	if(s.action == 'PICK_UP' and neighbour[4]):
		return 1, s.pickUpReward
	if(s.action == 'PUT_DOWN' and neighbour[5]):
		return 1, s.putDownReward
	return 0, 0

def policyIterationAlgebra(states, gamma):
	itr = 0
	n = len(states)
	utilities = []
	for keys in states:
		curState = states[keys]
		curState.action = 'NORTH'

	while(itr == 0 or isSamePolicy(states) == False):
		uIter = []
		for keys in states:
			curState = states[keys]
			curState.action = validActions[np.argmax(curState.qValues)]
		i, j = 0, 0
		A = np.zeros((n, n))
		B = np.zeros((n, 1))
		for s1 in states:
			j = 0
			for s2 in states:
				s = states[s1]
				sPrime = states[s2]
				a, b = getTransition(s, sPrime)
				A[i][j] = a
				B[i][0] += b
				j += 1
			i += 1
		V = np.linalg.solve(np.identity(n) - gamma*A, B)
		i = 0
		for keys in states:
			curState = states[keys]
			curState.value = V[i][0]
			uIter.append(curState.value)
			i += 1
		bellmanUpdate(states, gamma)
		utilities.append(uIter)
		itr += 1
	print("No of iterations: " + str(itr))
	return utilities

def qLearning(states, alpha, gamma, epsilon, decay, passDest, maxIter, depots, size, doEvaluate):
	evaluations = []
	iterNum = []
	for itr in range(maxIter):
		# print(itr)
		if(decay):
			epsilon = 0.1/(1 + itr)
		length = 0
		taxiPos = (np.random.choice([i for i in range(size)]), np.random.choice([i for i in range(size)]))
		passPos = depots[np.random.choice([i for i in range(len(depots))])]
		curState = states[(taxiPos, passPos, passDest)]
		while(curState.isExit == False and length < 500):
			optAction = np.argmax(curState.qValues)
			randomAction = np.random.choice([0, 1, 2, 3, 4, 5])
			curAction =  np.random.choice([optAction, randomAction], p = [1 - epsilon, epsilon])
			nextState, reward = simulator(curState, validActions[curAction])
			curState.qValues[curAction] = (1 - alpha)*curState.qValues[curAction] + alpha*(reward + gamma*max(nextState.qValues))
			curState = nextState
			length += 1
		if(itr % 10 == 0 and doEvaluate):
			disReward = evaluate(states, passDest, depots, gamma, size, 100)
			evaluations.append(disReward)
			iterNum.append(itr)
	for keys in states:
		curState = states[keys]
		curState.action = validActions[np.argmax(curState.qValues)]
	return iterNum, evaluations

def sarsaLearning(states, alpha, gamma, epsilon, decay, passDest, maxIter, depots, size, doEvaluate):
	evaluations = []
	iterNum = []
	for itr in range(maxIter):
		# print(itr)
		if(decay):
			epsilon = 0.1/(1 + itr)
		length = 0
		taxiPos = (np.random.choice([i for i in range(size)]), np.random.choice([i for i in range(size)]))
		passPos = depots[np.random.choice([i for i in range(len(depots))])]
		curState = states[(taxiPos, passPos, passDest)]
		optAction = np.argmax(curState.qValues)
		randomAction = np.random.choice([0, 1, 2, 3, 4, 5])
		curAction =  np.random.choice([optAction, randomAction], p = [1 - epsilon, epsilon])
		while(curState.isExit == False and length < 500):
			nextState, reward = simulator(curState, validActions[curAction])
			optAction = np.argmax(nextState.qValues)
			randomAction = np.random.choice([0, 1, 2, 3, 4, 5])
			nextAction =  np.random.choice([optAction, randomAction], p = [1 - epsilon, epsilon])
			curState.qValues[curAction] = (1 - alpha)*curState.qValues[curAction] + alpha*(reward + gamma*nextState.qValues[nextAction])
			curState = nextState
			curAction = nextAction
			length += 1
		if(itr % 10 == 0 and doEvaluate):
			disReward = evaluate(states, passDest, depots, gamma, size, 100)
			evaluations.append(disReward)
			iterNum.append(itr)
	for keys in states:
		curState = states[keys]
		curState.action = validActions[np.argmax(curState.qValues)]
	return iterNum, evaluations

def evaluate(states, passDest, depots, gamma, size, itr):
	avgReward = []
	for j in range(itr):
		# print(j)
		taxiPos = (np.random.choice([i for i in range(size)]), np.random.choice([i for i in range(size)]))
		passPos = depots[np.random.choice([i for i in range(len(depots))])]
		curState = states[(taxiPos, passPos, passDest)]
		disReward = 0
		length = 0
		while(curState.isExit == False and length < 500):
			curAction = np.argmax(curState.qValues)
			nextState, reward = simulator(curState, validActions[curAction])
			disReward += (gamma**length)*reward
			curState = nextState
			length += 1
		avgReward.append(disReward)
	return np.mean(avgReward)

part = sys.argv[1]
size = 0

if(part == 'A2b'):
	discountRates = [0.01, 0.1, 0.5, 0.8, 0.99]
	y = []
	x = []
	for i in range(5):
		states = generateStates()
		maxNormList = valueIteration(states, 10**-10, discountRates[i])
		y.append(maxNormList)
		x.append([i + 1 for i in range(len(maxNormList))])
	for i in range(5):
		plt.plot(x[i], y[i], label = discountRates[i])
	plt.xlabel('Iterations')
	plt.ylabel('Max Norm')
	plt.title('Max Norm vs Iterations for varying discount factors')
	plt.legend()
	plt.savefig('ValueIteration.png')

elif(part == 'A2c'):
	discountRates = [0.1, 0.99]
	passPos = (0, 0)
	passDest = (4, 4)
	taxiPos = (4, 0)
	for i in range(2):
		print("Discount Rate: " + str(discountRates[i]))
		states = generateStates()
		valueIteration(states, 10**-10, discountRates[i])
		curState = states[(taxiPos, passPos, passDest)]
		for j in range(20):
			print(curState.taxiPos, curState.passPos, curState.passDest, curState.action)
			curState, reward = simulator(curState, curState.action)
			if(curState.isExit):
				break

elif(part == 'A3a'):
	states = generateStates()
	print("Iterative Method:")
	policyIteration(states, 0.9)
	states = generateStates()
	print("\nLinear Algebra Method:")
	policyIterationAlgebra(states, 0.9)

elif(part == 'A3b'):
	discountRates = [0.01, 0.1, 0.5, 0.8, 0.99]
	y = []
	x = []
	for i in range(5):
		temp = []
		states = generateStates()
		utilities = policyIteration(states, discountRates[i])
		for i in range(len(utilities)):
			temp.append(policyLoss(utilities[i], utilities[-1]))
		y.append(temp)
		x.append([i + 1 for i in range(len(temp))])
	for i in range(5):
		plt.plot(x[i], y[i], label = discountRates[i])
	plt.xlabel('Iterations')
	plt.ylabel('Policy Loss')
	plt.title('Policy Loss vs Iterations for varying discount factors')
	plt.legend()
	plt.savefig('PolicyIteration.png')

elif(part == 'B2'):
	passDest = depots[np.random.choice([i for i in range(4)])]
	states = generateStates()
	iterNum, evaluations = qLearning(states, 0.25, 0.99, 0.1, False, passDest, 1000, depots, 5, True)
	df = pd.DataFrame({"y":evaluations})
	df = df.rolling(10, min_periods = 1).mean()
	evaluations = df['y'].tolist()
	plt.plot(iterNum, evaluations, label = 'Q-Learning')
	states = generateStates()
	iterNum, evaluations = qLearning(states, 0.25, 0.99, 0.1, True, passDest, 1000, depots, 5, True)
	df = pd.DataFrame({"y":evaluations})
	df = df.rolling(10, min_periods = 1).mean()
	evaluations = df['y'].tolist()
	plt.plot(iterNum, evaluations, label = 'Q-Learning(Decay)')
	states = generateStates()
	iterNum, evaluations = sarsaLearning(states, 0.25, 0.99, 0.1, False, passDest, 1000, depots, 5, True)
	df = pd.DataFrame({"y":evaluations})
	df = df.rolling(10, min_periods = 1).mean()
	evaluations = df['y'].tolist()
	plt.plot(iterNum, evaluations, label = 'Sarsa Learning')
	states = generateStates()
	iterNum, evaluations = sarsaLearning(states, 0.25, 0.99, 0.1, True, passDest, 1000, depots, 5, True)
	df = pd.DataFrame({"y":evaluations})
	df = df.rolling(10, min_periods = 1).mean()
	evaluations = df['y'].tolist()
	plt.plot(iterNum, evaluations, label = 'Sarsa Learning(Decay)')
	plt.xlabel('Iteration Number')
	plt.ylabel('Average Discounted Reward')
	plt.title('Average Discounted Reward vs Iterations for different Algorithms')
	plt.legend()
	plt.savefig('Convergence.png')

elif(part == 'B3'):
	passSources = [(4, 4), (0, 0), (0, 3), (4, 0), (0, 3)]
	passDestinations = [(0, 3), (0, 3), (0, 3), (0, 3), (0, 3)]
	taxiPositions = [(1, 3), (3, 2), (4, 1), (0, 4), (2, 0)]
	rewards = []
	for i in range(5):
		passDest = passDestinations[i]
		taxiPos = taxiPositions[i]
		passPos = passSources[i]
		print("Instance - " + str(i))
		print((taxiPos, passPos, passDest))
		states = generateStates()
		qLearning(states, 0.25, 0.99, 0.1, False, passDest, 2000, depots, 5, False)
		avgReward = []
		for j in range(100):
			curState = states[(taxiPos, passPos, passDest)]
			disReward = 0
			length = 0
			while(curState.isExit == False):
				nextState, reward = simulator(curState, curState.action)
				disReward += (0.99**length)*reward
				curState = nextState
				length += 1
			avgReward.append(disReward)
		rewards.append(np.mean(avgReward))
	print("Rewards for 5 instances:" , rewards)
	print("Average Reward", np.mean(rewards))

elif(part == 'B4'):
	passDest = depots[np.random.choice([i for i in range(4)])]
	explorationRates = [0, 0.05, 0.1, 0.5, 0.9]
	learningRates = [0.1, 0.2, 0.3, 0.4, 0.5]
	for i in range(5):
		states = generateStates()
		iterNum, evaluations = qLearning(states, 0.1, 0.99, explorationRates[i], False, passDest, 1000, depots, 5, True)
		df = pd.DataFrame({"y":evaluations})
		df = df.rolling(10, min_periods = 1).mean()
		evaluations = df['y'].tolist()
		plt.plot(iterNum, evaluations, label = 'Exploration Rate - ' + str(explorationRates[i]))
	plt.xlabel('Iteration Number')
	plt.ylabel('Average Discounted Reward')
	plt.title('Average Discounted Reward vs Iterations for varying Exploration Rates')
	plt.legend()
	plt.savefig('VaryingExplorationRate.png')

	for i in range(5):
		states = generateStates()
		iterNum, evaluations = qLearning(states, learningRates[i], 0.99, 0.1, False, passDest, 1000, depots, 5, True)
		df = pd.DataFrame({"y":evaluations})
		df = df.rolling(10, min_periods = 1).mean()
		evaluations = df['y'].tolist()
		plt.plot(iterNum, evaluations, label = 'Learning Rate - ' + str(learningRates[i]))
	plt.xlabel('Iteration Number')
	plt.ylabel('Average Discounted Reward')
	plt.title('Average Discounted Reward vs Iterations for varying Learning Rates')
	plt.legend()
	plt.savefig('VaryingLearningRate.png')

elif(part == 'B5'):
	passSources = [(9, 5), (6, 3), (0, 4), (9, 0), (0, 9)]
	passDestinations = [(9, 0), (9, 8), (0, 9), (5, 6), (1, 0)]
	taxiPositions = [(1, 3), (3, 2), (4, 1), (0, 4), (2, 0)]
	rewards = []
	for i in range(5):
		passDest = passDestinations[i]
		taxiPos = taxiPositions[i]
		passPos = passSources[i]
		print("Instance - " + str(i))
		print((taxiPos, passPos, passDest))
		states = generateStates10X10()
		qLearning(states, 0.25, 0.99, 0.1, False, passDest, 10000, depots10X10, 10, False)
		avgReward = []
		for j in range(100):
			curState = states[(taxiPos, passPos, passDest)]
			disReward = 0
			length = 0
			while(curState.isExit == False):
				nextState, reward = simulator(curState, curState.action)
				disReward += (0.99**length)*reward
				curState = nextState
				length += 1
			avgReward.append(disReward)
		rewards.append(np.mean(avgReward))
	print("Rewards for 5 instances:" , rewards)
	print("Average Reward", np.mean(rewards))

if(part == 'Anim5'):
	size = 5

if(part == 'Anim10'):
	size = 10
	
if(part == 'Anim5' or part == 'Anim10'):
	if(size == 5):
		states = generateStates()
		d = depots
		maxIter = 2000
	else:
		states = generateStates10X10()
		d = depots10X10
		maxIter = 10000

	taxiPos = (np.random.choice([i for i in range(size)]), np.random.choice([i for i in range(size)]))
	passDest = d[np.random.choice([i for i in range(int(size*4/5))])]
	passPos = d[np.random.choice([i for i in range(int(size*4/5))])]
	while(passPos == passDest):
		passPos = d[np.random.choice([i for i in range(int(size*4/5))])]

	# valueIteration(states, 10**-10, 0.9)
	# policyIteration(states, 0.9)
	# policyIterationAlgebra(states, 0.9)
	qLearning(states, 0.25, 0.99, 0.1, False, passDest, maxIter, d, size, False)
	# sarsaLearning(states, 0.25, 0.99, 0.1, False, passDest, maxIter, d, size, False)

	curState = states[(taxiPos, passPos, passDest)]
	print("Starting state:", (taxiPos, passPos, passDest))


	plt.xlim(0,size)
	plt.ylim(0,size)
	plt.xticks([i for i in range(size + 1)])
	plt.yticks([i for i in range(size + 1)])
	ax = plt.gca()
	ax.axes.xaxis.set_ticklabels([])
	ax.axes.yaxis.set_ticklabels([])
	plt.scatter(curState.taxiPos[1] + 0.5, curState.taxiPos[0] + 0.5, color = "yellow", s = 100, label = 'taxi')
	plt.scatter(curState.passPos[1] + 0.5, curState.passPos[0] + 0.5, color = "red", s = 100, label = 'passenger source')
	plt.scatter(curState.passDest[1] + 0.5, curState.passDest[0] + 0.5, color = "green", s = 100, label = 'passenger destination')
	plt.legend(loc='upper center', bbox_to_anchor = (0.5, -0.05), fancybox = True, shadow = True, ncol = 5)
	if(size == 5):
		plt.plot([1, 1], [0, 2], linewidth = 3, color = 'black')
		plt.plot([3, 3], [0, 2], linewidth = 3, color = 'black')
		plt.plot([2, 2], [3, 5], linewidth = 3, color = 'black')
	else:
		plt.plot([1, 1], [0, 4], linewidth = 3, color = 'black')
		plt.plot([3, 3], [6, 10], linewidth = 3, color = 'black')
		plt.plot([4, 4], [0, 4], linewidth = 3, color = 'black')
		plt.plot([6, 6], [4, 8], linewidth = 3, color = 'black')
		plt.plot([8, 8], [0, 4], linewidth = 3, color = 'black')
		plt.plot([8, 8], [6, 10], linewidth = 3, color = 'black')
	for i in range(size):
		plt.text(i + 0.5, -0.5, str(i))
	for i in range(size):
		plt.text(-0.25, i + 0.25, str(i))
	plt.grid(True)
	prevX = taxiPos[1]
	prevY = taxiPos[0]
	for i in range(100):
		plt.scatter(prevX + 0.5, prevY + 0.5, color = "white", s = 200)
		plt.scatter(curState.passDest[1] + 0.5, curState.passDest[0] + 0.5, color = "green", s = 100, label = 'passenger destination')
		plt.scatter(curState.taxiPos[1] + 0.5, curState.taxiPos[0] + 0.5, color = "yellow", s = 100, label = 'taxi')
		plt.pause(0.5)
		prevX = curState.taxiPos[1]
		prevY = curState.taxiPos[0]
		if(curState.isExit):
			break
		print(curState.taxiPos, curState.action)
		if(curState.action == 'NORTH'):
			curState = curState.northState
		elif(curState.action == 'SOUTH'):
			curState = curState.southState
		elif(curState.action == 'EAST'):
			curState = curState.eastState
		elif(curState.action == 'WEST'):
			curState = curState.westState
		elif(curState.action == 'PICK_UP'):
			curState = curState.pickUpState
		else:
			curState = curState.putDownState
		# curState, reward = simulator(curState, curState.action)
