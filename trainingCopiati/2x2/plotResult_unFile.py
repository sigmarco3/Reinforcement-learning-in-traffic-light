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
        ym = df['system_total_waiting_time']
        zm = df['system_total_stopped']
        um = df['system_mean_waiting_time']
        vm = df['system_mean_speed']

        #plt.ylabel("system_total_waiting time")


        #plt.xlabel("step")
        #plt.title("rete 4x4 reward diversi tra colonne ")
       # plt.plot(x, ym)


        mu1 = ym.mean()
        sigma1 = ym.std()


        std_error = np.std(ym, ddof=1) / np.sqrt(len(ym))
        # create chart
        plt.plot(x,  # x-coordinates of bars
        ym)
        plt.title('singolo incrocio esperimento da libreria (alpha 0.1 gamma 0.99) misura total waiting time')
        plt.xlabel('step')
        plt.ylabel('system total waiting time(seconds)')
        plt.show()

        #plt.plot(x,zm)


        std_error = np.std(zm, ddof=1) / np.sqrt(len(zm))
        # create chart
        plt.plot(x,zm  # height of bars
        )

        plt.xlabel("step")
        plt.title("singolo incrocio esperimento da libreria (alpha 0.1 gamma 0.99 misura total stopped ")
        plt.ylabel("system total stopped (vehicles)")
        plt.show()
        #plt.plot(x, um)


        std_error = np.std(um, ddof=1) / np.sqrt(len(um))
        # create chart
        plt.plot(x,um)
        plt.xlabel("step")
        plt.title("singolo incrocio esperimento da libreria (alpha 0.1 gamma 0.99 misura mean waiting time ")
        plt.ylabel("system mean waiting time(seconds)")
        plt.show()
        #plt.plot(x, vm)


        std_error = np.std(vm, ddof=1) / np.sqrt(len(vm))
        # create chart
        plt.plot(x,vm)
        plt.xlabel("step")
        plt.title("singolo incrocio esperimento da libreria (alpha 0.1 gamma 0.99) misura mean speed")
        plt.ylabel("system mean speed(Km/h)")
        plt.show()
if __name__ == '__main__':
    plotResult('D:/programmi/sumo/esperimenti semafori/outputs/2way-single-intersection/result-alpha0.1-gamma0.99-queue')
    #plotResult('D:/programmi/sumo/esperimenti semafori/outputs/2way-single-intersection/result-alpha0.1-gamma0.99')