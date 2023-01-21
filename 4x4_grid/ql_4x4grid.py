import argparse
import os
import sys
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
#sys.path.insert(1,"D:\programmi\sumo\sumo-rl-master\sumo_rl_master\environment")  # uso un altro file env.py in modo da non intaccare la libreria
from environment.env import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

def plotResult(file):
    #with open(file + '_conn0_run4.csv', newline="", encoding="ISO-8859-1") as filecsv:
        # lettore = csv.reader(filecsv, delimiter=",")
        # header = next(lettore)
        #
        # t = [(linea[0], linea[3]) for linea in lettore]
        # t = np.array(t)
        # dati = t[:, 1]
        #
        # time = t[:, 0]
        # times = np.array(time)
        # dati = [float(s) for s in dati]
        #
        # # print(datis.shape)
        #
        # # fig = plt.figure()
        # # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # plt.xlabel('secondi')
        # plt.ylabel('system total waiting time')
        # max = float(time[len(time)-1])
        #
        # plt.xticks(time)
        # plt.plot(time, dati, color='blue')
        #
        # plt.show()
        df = pd.read_csv(file + '_conn0_run4.csv')
        # definiamo le dimensioni della finestra in pollici ed il dpi
        from matplotlib.pyplot import figure
        figure(figsize=(18, 10), dpi=80)
        x = df['total_vehicle']
        y = df['system_total_stopped']
        plt.xlabel("auto")
        plt.ylabel("sistem total stopped")
        plt.plot(x, y)
        plt.show()
if __name__ == '__main__':

    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 4

    env = SumoEnvironment(net_file='nets/4x4-Lucas/4x4.net.xml',
                          route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                          use_gui=True,
                          num_seconds=2000,
                          min_green=5,
                          delta_time=5)

    initial_states = env.reset()
    ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),  #qui assegno un codice per la funzione
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=alpha,
                                 gamma=gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)) for ts in env.ts_ids}
    for run in range(1, runs+1):
        if run != 1:
            initial_states = env.reset()
            for ts in initial_states.keys():
                ql_agents[ts].state = env.encode(initial_states[ts], ts)

        infos = []
        done = {'__all__': False}
        while not done['__all__']:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(action=actions)
            
            for agent_id in s.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

        env.save_csv('outputs/4x4/ql-4x4grid-variReward-2000sec(e auto)', run)
        env.close()
        if run==4:
            plotResult('outputs/4x4/ql-4x4grid-variReward-2000sec(e auto)')

