import numpy as np
import random

'''
    This class represents an agent that uses Sarsa to learn
'''
class SarsaAgent():
    def __init__(self, gamma, alpha, epsilon, numOfAction, stateDRange):
        self.gamma = gamma # discount factor
        self.alpha = alpha # learning rate
        self.epsilon = epsilon # the threshold for random action selection
        self.numOfAction = numOfAction
        self.stateDRange = stateDRange
        self.totalReward = 0
        self.values = self.initializeValueTable()

    # initialize an empty q-table
    def initializeValueTable(self):
        table = {}
        for i in range(self.stateDRange):
            for j in range(self.stateDRange):
                for k in range(self.stateDRange):
                    for l in range(self.stateDRange):
                        for m in range(self.stateDRange):
                            for a in range(self.numOfAction):
                                table[((i, j, k, l, m), a)] = 0.0
        return table

    # compute the new q-value for the current (state, action) pair and update it in self.qValues
    def update(self, state, action, nextState, reward, nextAction):
        self.values[(state, action)] += self.alpha * (reward + self.gamma * (self.values[(nextState, nextAction)] - self.values[(state, action)]))

    # choose a legal action based on a particular gamma value
    def getAction(self, state):
        action = None
        r = random.random()  # rand 0, 1, 2
        # if less than epsilon, go with random action
        if r < self.epsilon:
            action = random.randint(0, self.numOfAction - 1)
        # otherwise, stick with our own policy (i.e., choose the action with the highest q-value)
        else:
            # action_values looks like [q(s, 0),  q(s, 1), q(s, 2)]
            action_values = [self.values[(state, action)] for action in range(self.numOfAction)]
            action = np.argmax(action_values)
        return action # return 0, 1, or 2

    # set alpha to a particular value
    def setAlpha(self, alpha):
        self.alpha = alpha

    # get epsilon
    def getEpsilon(self):
        return self.epsilon

    # set epsilon to a particular value
    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    # linearly decrease epsilon as training number increases
    def updateEpsilon(self, minEpsilon, decayFactor):
        # if self.epsilon > minEpsilon:
        #     self.epsilon -= self.numOfAction / decayFactor
        self.epsilon = self.epsilon * decayFactor

    # return the total reward
    def getTotalReward(self):
        return self.totalReward

    # update the total reward
    def updateTotalReward(self, reward):
        self.totalReward += reward

    # reset the total reward to 0
    def resetTotalReward(self):
        self.totalReward = 0

