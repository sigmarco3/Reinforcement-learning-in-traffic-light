import numpy as np
import csv
from sumo_rl.exploration import EpsilonGreedy
import pandas as pd
from ast import literal_eval
class QLAgent:
    def readTable(self):
        dict={}
        inputFile="tabelle/tabelleSingole 2x2-"+self.id+".csv"
        with open(inputFile,'r',newline='') as file_name:
            reader=csv.reader(file_name,delimiter=';')
            for row in reader:

                    k, v = row
                    k=literal_eval(k)
                    v=literal_eval(v)
                    dict[k] = v



        return dict

    def __init__(self, starting_state, state_space, action_space,id, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.id = id
        self.q_table = self.readTable()
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self):
        #self.q_table = self.readTable()
        
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        self.state = s1
        self.acc_reward += reward
