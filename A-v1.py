# Name Classes as CamelCase format and functions/variables as camelCase (first letter small). 
# Constants are named as FIRST_LAST

# TODO:
# 1) Linear algebra method for policy iteration

import numpy as np
import random
import matplotlib.pyplot as plt

class State:
	def __init__(self):
		self.gridIdx = (-1, -1, -1)
		self.value = 0
		self.qValues = [0, 0, 0, 0, 0, 0]

		self.northState = None
		self.southState = None
		self.eastState = None
		self.westState = None
		self.pickUpState = None
		self.putDownState = None

		self.isTaxiSource = False
		self.isPassSource = False
		self.isPassDest = False
		self.isExit = False

		self.action = None

	def applyAction(self):
		if(self.action == 'NORTH'):
			nextState = np.random.choice([self.northState, self.southState, self.eastState, self.westState], p = [0.85, 0.05, 0.05, 0.05])
			reward = -1
		elif(self.action == 'SOUTH'):
			nextState = np.random.choice([self.northState, self.southState, self.eastState, self.westState], p = [0.05, 0.85, 0.05, 0.05])
			reward = -1
		elif(self.action == 'EAST'):
			nextState = np.random.choice([self.northState, self.southState, self.eastState, self.westState], p = [0.05, 0.05, 0.85, 0.05])
			reward = -1
		elif(self.action == 'WEST'):
			nextState = np.random.choice([self.northState, self.southState, self.eastState, self.westState], p = [0.05, 0.05, 0.05, 0.85])
			reward = -1
		elif(self.action == 'PICK_UP'):
			if(self.isPassSource):
				nextState = self.pickUpState
				reward = -1
			else:
				nextState = self
				reward = -10
		else:
			nextState = self.putDownState
			reward = -10
			if(self.isPassDest):
				reward = 20
		return nextState, reward

def generateInstance():
	depots = [[0, 0], [4, 0], [0, 3], [4, 4]]
	taxiSource = [-1, -1]
	taxiSource[0] = np.random.choice([0, 1, 2, 3, 4])
	taxiSource[1] = np.random.choice([0, 1, 2, 3, 4])
	if(taxiSource in depots):
		depots.remove(taxiSource)
	passSource, passDest = random.sample(depots, 2)
	grid = [[[State() for i in range(5)] for i in range(5)] for i in range(2)]
	print(taxiSource, passSource, passDest)

	# Border Walls
	for k in range(2):
		for i in range(5):
			for j in range(5):
				grid[k][i][j].gridIdx = (k, i, j)
				if(i == 4):
					grid[k][i][j].northState = grid[k][i][j]
				else:
					grid[k][i][j].northState = grid[k][i + 1][j]
				if(i == 0):
					grid[k][i][j].southState = grid[k][i][j]
				else:
					grid[k][i][j].southState = grid[k][i - 1][j]
				if(j == 4):
					grid[k][i][j].eastState = grid[k][i][j]
				else:
					grid[k][i][j].eastState = grid[k][i][j + 1]
				if(j == 0):
					grid[k][i][j].westState = grid[k][i][j]
				else:
					grid[k][i][j].westState = grid[k][i][j - 1]
				if(k == 0):
					grid[k][i][j].pickUpState = grid[k + 1][i][j]
				else:
					grid[k][i][j].putDownState = grid[k - 1][i][j]

	# Walls in grid
	for k in range(2):
		grid[k][0][0].eastState = grid[k][0][0]
		grid[k][1][0].eastState = grid[k][1][0]
		grid[k][0][1].westState = grid[k][0][1]
		grid[k][1][1].westState = grid[k][1][1]

		grid[k][0][2].eastState = grid[k][0][2]
		grid[k][1][2].eastState = grid[k][1][2]
		grid[k][0][3].westState = grid[k][0][3]
		grid[k][1][3].westState = grid[k][1][3]

		grid[k][3][1].eastState = grid[k][3][1]
		grid[k][4][1].eastState = grid[k][4][1]
		grid[k][3][2].westState = grid[k][3][2]
		grid[k][4][2].westState = grid[k][4][2]

	grid[0][taxiSource[0]][taxiSource[1]].isTaxiSource = True
	grid[0][passSource[0]][passSource[1]].isPassSource = True
	grid[1][passDest[0]][passDest[1]].isPassDest = True

	exitState = State()
	exitState.isExit = True
	grid[1][passDest[0]][passDest[1]].putDownState = exitState

	return grid, taxiSource, passSource, passDest

def simulatorV1(grid, taxiSource):
	curState = grid[0][taxiSource[0]][taxiSource[1]]
	totalReward = 0
	i = 0

	while(curState.isExit == False):
		print(curState.gridIdx)
		if(curState.action == 'PICK_UP' and curState.isPassSource):
			curState.isPassSource = False

		if(curState.action == 'PUT_DOWN'):
			curState.isPassSource = True

		curState, reward = curState.applyAction()
		totalReward += reward
		i += 1
		if(i > 20):
			break

def simulatorV2(state, action):
	(k, i, j) = state.gridIdx
	if(action == 'NORTH'):
		nextState = np.random.choice([state.northState, state.southState, state.eastState, state.westState], p = [0.85, 0.05, 0.05, 0.05])
		reward = -1
	elif(action == 'SOUTH'):
		nextState = np.random.choice([state.northState, state.southState, state.eastState, state.westState], p = [0.05, 0.85, 0.05, 0.05])
		reward = -1
	elif(action == 'EAST'):
		nextState = np.random.choice([state.northState, state.southState, state.eastState, state.westState], p = [0.05, 0.05, 0.85, 0.05])
		reward = -1
	elif(action == 'WEST'):
		nextState = np.random.choice([state.northState, state.southState, state.eastState, state.westState], p = [0.05, 0.05, 0.05, 0.85])
		reward = -1
	elif(action == 'PICK_UP'):
		if(k == 0 and state.isPassSource):
			state.isPassSource = False       ## Check This
			nextState = state.pickUpState
			reward = -1
		elif(k == 0 and state.isPassSource == False):
			nextState = state
			reward = -10
		else:
			nextState = state
			reward = -1
	else:
		if(k == 0 and state.isPassSource):
			nextState = state
			reward = -1
		elif(k == 0 and state.isPassSource == False):
			nextState = state
			reward = -10
		elif(k == 1 and state.isPassDest):
			nextState = state.putDownState
			reward = 20
		else:
			nextState = state.putDownState
			nextState.isPassSource = True   ## Check This
			reward = -1
	return nextState, reward

def isConverged(grid, newValues, epsilon):
	maxValue = 0
	for k in range(2):
		for i in range(5):
			for j in range(5):
				maxValue = max(maxValue, abs(grid[k][i][j].value - newValues[k][i][j]))
	return maxValue < epsilon

def bellmanUpdate(grid, gamma, getValues, getPolicy, getVPi):
	newValues = [[[0 for i in range(5)] for i in range(5)] for i in range(2)]
	validActions = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'PICK_UP', 'PUT_DOWN']
	for k in range(2):
		for i in range(5):
			for j in range(5):
				curState = grid[k][i][j]
				northVal = 0.85*(-1 + gamma*curState.northState.value) + 0.05*(-1 + gamma*curState.southState.value) + 0.05*(-1 + gamma*curState.eastState.value) + 0.05*(-1 + gamma*curState.westState.value)
				southVal = 0.05*(-1 + gamma*curState.northState.value) + 0.85*(-1 + gamma*curState.southState.value) + 0.05*(-1 + gamma*curState.eastState.value) + 0.05*(-1 + gamma*curState.westState.value)
				eastVal = 0.05*(-1 + gamma*curState.northState.value) + 0.05*(-1 + gamma*curState.southState.value) + 0.85*(-1 + gamma*curState.eastState.value) + 0.05*(-1 + gamma*curState.westState.value)
				westVal = 0.05*(-1 + gamma*curState.northState.value) + 0.05*(-1 + gamma*curState.southState.value) + 0.05*(-1 + gamma*curState.eastState.value) + 0.85*(-1 + gamma*curState.westState.value)
				if(k == 0):
					if(curState.isPassSource):
						pickUpVal = (-1 + gamma*curState.pickUpState.value)
						putDownVal = (-1 + gamma*curState.value)
					else:
						pickUpVal = (-10 + gamma*curState.value)
						putDownVal = (-10 + gamma*curState.value)
				else:
					if(curState.isPassDest):
						pickUpVal = (-1 + gamma*curState.value)
						putDownVal = (20 + gamma*curState.putDownState.value)
					else:
						pickUpVal = (-1 + gamma*curState.value)
						putDownVal = (-1 + gamma*curState.putDownState.value)
				qValues = [northVal, southVal, eastVal, westVal, pickUpVal, putDownVal]
				if(getValues):
					newValues[k][i][j] = max(qValues)
				if(getPolicy):
					newValues[k][i][j] = np.argmax(qValues)
				if(getVPi):
					newValues[k][i][j] = qValues[validActions.index(curState.action)]
	return newValues

def valueIteration(grid, taxiSource, epsilon):
	newValues = [[[0 for i in range(5)] for i in range(5)] for i in range(2)]
	gamma = 0.9
	itr = 0
	while(itr == 0 or isConverged(grid, newValues, epsilon) == False):
		for k in range(2):
			for i in range(5):
				for j in range(5):
					grid[k][i][j].value = newValues[k][i][j]
		newValues = bellmanUpdate(grid, gamma, True, False, False)
		itr += 1
	validActions = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'PICK_UP', 'PUT_DOWN']
	optPolicy = bellmanUpdate(grid, gamma, False, True, False)
	for k in range(2):
		for i in range(5):
			for j in range(5):
				grid[k][i][j].action = validActions[optPolicy[k][i][j]]
	print(itr)

def isSamePolicy(grid, newPolicy):
	validActions = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'PICK_UP', 'PUT_DOWN']
	for k in range(2):
		for i in range(5):
			for j in range(5):
				if(grid[k][i][j].action != validActions[newPolicy[k][i][j]]):
					return False
	return True

def policyIteration(grid, taxiSource):
	gamma = 1
	validActions = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'PICK_UP', 'PUT_DOWN']
	newPolicy = [[[0 for i in range(5)] for i in range(5)] for i in range(2)]
	itr = 0
	while(itr == 0 or isSamePolicy(grid, newPolicy) == False):
		for k in range(2):
			for i in range(5):
				for j in range(5):
					grid[k][i][j].action = validActions[newPolicy[k][i][j]]
		newValues = bellmanUpdate(grid, gamma, False, False, True)
		for k in range(2):
			for i in range(5):
				for j in range(5):
					grid[k][i][j].value = newValues[k][i][j]
		newPolicy = bellmanUpdate(grid, gamma, False, True, False)
		itr += 1
	print(itr)

def qLearning(grid, taxiSource, passSource, passDest, alpha, gamma, epsilon):
	validActions = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'PICK_UP', 'PUT_DOWN']
	depots = [[0, 0], [4, 0], [0, 3], [4, 4]]
	for itr in range(5000):
		epsilon = epsilon/(1 + itr)
		startState = [-1, -1, -1]
		startState[0] = 0 #np.random.choice([0, 1])
		startState[1] = np.random.choice([0, 1, 2, 3, 4])
		startState[2] = np.random.choice([0, 1, 2, 3, 4])
		# passSource = random.sample(depots, 1)[0]
		# print(passSource)
		if(startState == [0] + passSource):
			itr -= 1
			continue
		curState = grid[startState[0]][startState[1]][startState[2]]
		newQValues = [[[[None for i in range(6)] for i in range(5)] for i in range(5)] for i in range(2)]
		if(startState[0] == 0):
			grid[0][passSource[0]][passSource[1]].isPassSource = True
		length = 0
		while(curState.isExit == False and length < 500): #2000 good results
			(k, i, j) = curState.gridIdx
			choices = [0, 1, 2, 3, 4, 5]
			optAction = np.argmax(curState.qValues)
			randomAction = np.random.choice(choices)
			randomAction = np.random.choice([0, 1, 2, 3, 4, 5])
			curAction =  np.random.choice([optAction, randomAction], p = [1 - epsilon, epsilon])
			# print(curState.gridIdx, validActions[curAction])
			nextState, reward = simulatorV2(curState, validActions[curAction])
			curState.qValues[curAction] = (1 - alpha)*curState.qValues[curAction] + alpha*(reward + gamma*max(nextState.qValues))
			curState = nextState
			length += 1
		print(itr)

		for k in range(2):
			for i in range(5):
				for j in range(5):
					grid[k][i][j].isPassSource = False
		# grid[0][taxiSource[0]][taxiSource[1]].isTaxiSource = True
	grid[0][passSource[0]][passSource[1]].isPassSource = True
		# grid[1][passDest[0]][passDest[1]].isPassDest = True
		
	for k in range(2):
		for i in range(5):
			for j in range(5):
				print((k, i, j), grid[k][i][j].qValues)
				grid[k][i][j].action = validActions[np.argmax(grid[k][i][j].qValues)]
		

grid, taxiSource, passSource, passDest = generateInstance()
# valueIteration(grid, taxiSource, 10**-6)
# policyIteration(grid, taxiSource)
curState = grid[0][taxiSource[0]][taxiSource[1]]
qLearning(grid, taxiSource, passSource, passDest, 0.25, 0.99, 0.1)
plt.xlim(0,5)
plt.ylim(0,5)
points = []
plt.scatter(taxiSource[1] + 0.5, taxiSource[0] + 0.5, color = "yellow", s = 100, label = 'taxi')
plt.scatter(passSource[1] + 0.5, passSource[0] + 0.5, color = "red", s = 100, label = 'passenger source')
plt.scatter(passDest[1] + 0.5, passDest[0] + 0.5, color = "green", s = 100, label = 'passenger destination')
plt.plot([1, 1], [0, 2], linewidth = 3, color = 'black')
plt.plot([3, 3], [0, 2], linewidth = 3, color = 'black')
plt.plot([2, 2], [3, 5], linewidth = 3, color = 'black')
plt.grid()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
prevX = taxiSource[0]
prevY = taxiSource[1]
for i in range(100):
	(k, i, j) = curState.gridIdx
	print((k, i, j), curState.action)
	plt.scatter(prevX + 0.5, prevY + 0.5, color = "white", s = 200)
	plt.scatter(j + 0.5, i + 0.5, color = "yellow", s = 100)
	plt.pause(0.2)
	prevX = j
	prevY = i
	if(curState.isExit):
		break
	curState, reward = simulatorV2(curState, curState.action)


