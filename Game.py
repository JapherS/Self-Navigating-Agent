import sys
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import statistics

import gym
import gym_race
from QLearning import QLearningAgent
from Sarsa import SarsaAgent

class Game():
    def __init__(self):
        self.env = gym.make("Pyrace-v0")
        self.numOfTraining = 9999999
        self.numOfStep = 2000
        self.numOfAction = self.env.action_space.n
        self.minEpsilon = 0.005
        self.stateDRange = 11

    def playQLearning(self, alpha, epsilon, gamma, decayFactor, trainingThreshold):
        algoTitle = "Q-Learning"
        trainingRewards = []
        meanRewardArray = []
        reachedGoalFirst = False # flag for the status that the agent has beat the game for the first time
        reachedGoalFirstENum = 0 # the the episode number the agent first reached the goal
        reachedGoalStable = False # flag for the status that the agent has learned to continuously beat the game

        # initialize UI window
        self.env.set_view(True)

        # initialize a q-learning agent
        qLearningAgent = QLearningAgent(gamma, alpha, epsilon, self.numOfAction, self.stateDRange)

        for episode in range(self.numOfTraining):
            trainingRewards, meanRewardArray, reachedGoalStable = self.updateRewardPlot(trainingRewards, meanRewardArray,
                                                                 qLearningAgent.getTotalReward(), reachedGoalStable)

            # agent has completed training and plot the second graph
            if reachedGoalStable:
                self.plotGraph(trainingRewards, episode-1, algoTitle)
                print("=======================================================")
                print(algoTitle + " training completed!")
                print("First time reaching the goal: Episode " + str(reachedGoalFirstENum))
                print("Time point of training completion: Episode " + str(episode-1))
                return

            # initialize a starting state for each new episode
            state = tuple(self.env.reset())

            # reset totalReward to 0 for each new training episode
            qLearningAgent.resetTotalReward()

            for t in range(self.numOfStep):
                # get the next action
                action = qLearningAgent.getAction(state)
                observation, reward, trainingCompleted, garbage = self.env.step(action)
                nextState = tuple(observation) # format nextState

                # update total reward in the current training episode
                qLearningAgent.updateTotalReward(reward)

                # update q-value with (s,a,r,s')
                qLearningAgent.update(state, action, nextState, reward)

                # update state for the next iteration
                state = nextState

                # hide UI during initial training
                if episode >= trainingThreshold:
                    self.env.render()

                # end the episode if the agent crashed or has exceeded the max number of steps allowed
                if trainingCompleted or t == self.numOfStep - 1:
                    self.displayEpisodeSummary(episode, t, qLearningAgent.getTotalReward())
                    break

            # plot the 1st graph - if it is the first time the agent has reached the goal
            if qLearningAgent.getTotalReward() == 10000 and not reachedGoalFirst:
                self.plotGraph(trainingRewards, episode, algoTitle)
                reachedGoalFirst = True
                reachedGoalFirstENum = episode

            # update epsilon for the next episode
            # qLearningAgent.updateEpsilon(self.minEpsilon, episode)
            qLearningAgent.updateEpsilon(self.minEpsilon, decayFactor)



    def playSarsa(self, alpha, epsilon, gamma, decayFactor, trainingThreshold):
        algoTitle = "Sarsa"
        trainingRewards = []
        meanRewardArray = []
        reachedGoalFirst = False  # flag for the status that the agent has beat the game for the first time
        reachedGoalFirstENum = 0  # the the episode number the agent first reached the goal
        reachedGoalStable = False  # flag for the status that the agent has learned to continuously beat the game

        # initialize UI window
        self.env.set_view(True)

        # initialize a q-learning agent
        sarsaAgent = SarsaAgent(gamma, alpha, epsilon, self.numOfAction, self.stateDRange)

        for episode in range(self.numOfTraining):
            trainingRewards, meanRewardArray, reachedGoalStable = self.updateRewardPlot(trainingRewards, meanRewardArray,
                                                                 sarsaAgent.getTotalReward(), reachedGoalStable)

            # agent has completed training
            if reachedGoalStable:
                self.plotGraph(trainingRewards, episode-1, algoTitle)
                print("=======================================================")
                print(algoTitle + " training completed!")
                print("First time reaching the goal: Episode " + str(reachedGoalFirstENum))
                print("Time point of training completion: Episode " + str(episode-1))
                return

            # initialize a starting state for each new episode
            state = tuple(self.env.reset())

            # reset totalReward to 0 for each new training episode
            sarsaAgent.resetTotalReward()

            action = sarsaAgent.getAction(state)
            for t in range(self.numOfStep):
                # get the next action
                observation, reward, trainingCompleted, garbage = self.env.step(action)
                nextState = tuple(observation) # format nextState

                # update total reward in the current training episode
                sarsaAgent.updateTotalReward(reward)

                # get the next Action
                nextAction = sarsaAgent.getAction(nextState)

                # update q-value with (s,a,r,s')
                sarsaAgent.update(state, action, nextState, reward, nextAction)

                # update state and action for the next iteration
                state = nextState
                action = nextAction

                # hide UI during initial training
                if episode >= trainingThreshold:
                    self.env.render()

                # end the episode if the agent crashed or has exceeded the max number of steps allowed
                if trainingCompleted or t == self.numOfStep - 1:
                    self.displayEpisodeSummary(episode, t, sarsaAgent.getTotalReward())
                    break

            # plot the 1st graph - if it is the first time the agent has reached the goal
            if sarsaAgent.getTotalReward() == 10000 and not reachedGoalFirst:
                self.plotGraph(trainingRewards, episode, algoTitle)
                reachedGoalFirst = True
                reachedGoalFirstENum = episode

            # update epsilon for the next episode
            sarsaAgent.updateEpsilon(self.minEpsilon, decayFactor)

    # def saveTable(self, q_table):
    #     self.q_values = q_table

    # plot graph
    def plotGraph(self, rewards, episode, algoTitle):
        plt.plot(rewards)
        plt.ylabel('Episode Rewards')
        plt.xlabel('Episode Units (x10)')
        plt.title(algoTitle + " - " + str(episode) + " Episode")
        plt.show()

    # display the summarized result of a episode
    def displayEpisodeSummary(self, episode, t, r):
        print("Episode %d || Time steps: %i | Total reward: %f" % (episode, t, round(r, 2)))

    # update overall training reward for graph plotting
    def updateRewardPlot(self, trainingRewards, meanRewardArray, r, reachedGoalStable):
        meanRewardArray.append(r)
        if len(meanRewardArray) == 10:
            mean = statistics.mean(meanRewardArray)
            if mean == 10000:
                reachedGoalStable = True
            trainingRewards.append(mean)
            meanRewardArray = []
        return (trainingRewards, meanRewardArray, reachedGoalStable)


if __name__ == "__main__":
    alpha = 0.15
    epsilon = 1
    gamma = 0.99
    decayFactor = 0.99
    trainingThreshold = 999999 # initial training to be completed fast (without UI)

    game = Game()
    game.playQLearning(alpha, epsilon, gamma, decayFactor, trainingThreshold)
    # game.playSarsa(alpha, epsilon, gamma, decayFactor, trainingThreshold)

