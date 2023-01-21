import sys
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
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
        df = pd.read_csv(file + '_conn0_run1.csv')
        # definiamo le dimensioni della finestra in pollici ed il dpi
        from matplotlib.pyplot import figure
        figure(figsize=(18, 10), dpi=80)
        x = df['step']
        t =df['total_vehicle']
        y = df['system_total_waiting_time']
        z = df['system_total_stopped']
        u = df['system_mean_waiting_time']
        v = df['system_mean_speed']


        df1 = pd.read_csv(file + '_conn0_run2.csv')
        y1 = df['system_total_waiting_time']


        t1 = df1['total_vehicle']
        z1 = df1['system_total_stopped']
        u1 = df1['system_mean_waiting_time']
        v1 = df1['system_mean_speed']
        df2 = pd.read_csv(file + '_conn0_run3.csv')
        y2 = df2['system_total_waiting_time']
        z2 = df2['system_total_stopped']
        u2 = df2['system_mean_waiting_time']
        v2 = df2['system_mean_speed']


        t2 = df2['total_vehicle']
        df3 = pd.read_csv(file + '_conn0_run4.csv')
        y3 = df3['system_total_waiting_time']
        z3 = df3['system_total_stopped']
        u3 = df3['system_mean_waiting_time']
        v3 = df3['system_mean_speed']


        t3 = df3['total_vehicle']
        ym = (y + y1 + y2 + y3) / 4
        zm = (z + z1 + z2 + z3) / 4
        tm = (t + t1 +t2 +t3)/4
        um = (u + u1 + u2 + u3) / 4
        vm = (v + v1 + v2 + v3) / 4


        #plt.ylabel("system_total_waiting time")


        #plt.xlabel("step")
        #plt.title("rete 4x4 reward diversi tra colonne ")
       # plt.plot(x, ym)


        mu1 = ym.mean()
        sigma1 = ym.std()


        std_error = np.std(ym, ddof=1) / np.sqrt(len(ym))
        # create chart

        # fig ,ax = plt.subplots(1)
        # ax.plot(x, mu1, lw=2, label='total waiting time', color='blue')
        #
        # ax.fill_between(x, mu1 + sigma1, mu1 - sigma1, facecolor='blue', alpha=0.5)
        plt.bar(x=tm,  # x-coordinates of bars
        height=ym,yerr=std_error)
        # ax.set_ylabel("total waiting time")
        # ax.grid()
        plt.title('2x2 wait (alpha 0.1 gamma 0.99) misura total waiting time')
        plt.xlabel('step')
        plt.ylabel('system total waiting time(seconds)')
        plt.show()

        #plt.plot(x,zm)


        std_error = np.std(zm, ddof=1) / np.sqrt(len(zm))
        # create chart
        plt.bar(x=tm,  # x-coordinates of bars
        height=zm,  # height of bars
        yerr=std_error,  # error bar width
        capsize=4)

        plt.xlabel("step")
        plt.title("2x2 wait (alpha 0.1 gamma 0.99) misura total stopped ")
        plt.ylabel("system total stopped (vehicles)")
        plt.show()
        #plt.plot(x, um)


        std_error = np.std(um, ddof=1) / np.sqrt(len(um))
        # create chart
        plt.bar(x=x,  # x-coordinates of bars
        height=um,  # height of bars
        yerr=std_error,  # error bar width
        capsize=4)
        plt.xlabel("step")
        plt.title("2x2 wait(alpha 0.1 gamma 0.99) misura mean waiting time ")
        plt.ylabel("system mean waiting time(seconds)")
        plt.show()
        #plt.plot(x, vm)


        std_error = np.std(vm, ddof=1) / np.sqrt(len(vm))
        # create chart
        plt.bar(x=x,  # x-coordinates of bars
        height=vm,  # height of bars
        yerr=std_error,  # error bar width
        capsize=4)
        plt.xlabel("step")
        plt.title("2x2  wait (alpha 0.1 gamma 0.99) misura mean speed")
        plt.ylabel("system mean speed(Km/h)")
        plt.show()
if __name__ == '__main__':
    plotResult('D:/programmi/sumo/esperimenti semafori/Reinforcement-learning-in-traffic-light/outputs/2x2/result-alpha0.1-gamma0.99_trainingCopiato_50v')