import matplotlib.pyplot as plt
import os
import re

log_dir = '/home/wnpan/feddf/sst_01/'
list = os.listdir(log_dir)


# pattern = '.*validation performance = (.*)\n'
# '\{\'loss\': ([0-9][.][0-9]*), \'top1\': ([0-9][.][0-9]*).*'
# pattern = re.compile(pattern)
# name_pattern = '(.*)\\.txt'
def stat_movement():
    target_comm = [5, 10, 20, 50, 100]
    target_acc = 75.0
    import numpy as np
    import json
    for entry in list:
        data_files = os.listdir(os.path.join(log_dir, entry))
        y = []
        comm_rounds = []
        name = entry
        for i, data_file in enumerate(data_files):
            yi = []
            comm_round = 101
            # data_file = os.path.join(log_dir, entry, data_file, '0', 'log.txt')
            data_file = os.path.join(log_dir, entry, data_file, '0', 'log-1.json')
            datas = json.load(open(data_file))
            for i, data in enumerate(datas):
                if data['comm_round'] in target_comm and (i % 2) == 0:
                    yi.append(data['top1'])

            if yi != []:
                y.append(yi)
            comm_rounds.append(comm_round)
        y = np.array(y)
        mean_acc = np.mean(y, axis=0)
        std_acc = np.std(y, axis=0)
        print(f"method:{name}")
        info = ""
        for i in range(5):
            info += f"{mean_acc[i]:.2f} {std_acc[i]:.2f} "
        print(info)
        comm_rounds = np.array(comm_rounds)
        mean_rounds = np.mean(comm_rounds)
        std_rounds = np.std(comm_rounds)
        print(f"reach target:{mean_rounds},std:{std_rounds}")


def stat_best_acc():
    import json
    import numpy as np
    for entry in list:
        data_files = os.listdir(os.path.join(log_dir, entry))
        y = []
        final_y = []
        name = entry
        for i, data_file in enumerate(data_files):

            data_file = os.path.join(log_dir, entry, data_file, '0', 'log-1.json')
            try:
                datas = json.load(open(data_file))
                final_y.append(datas[-2]['top1'])
                y.append(datas[-1]['best_perfs']['top1'][0])
            except:
                print(f"failed on {name}")
        final_y = np.array(final_y)
        y = np.array(y)
        mean_acc = np.mean(y)
        std_acc = np.std(y,ddof = 1)
        print(f"method:{name} \nBest:")
        print(f"mean acc:{mean_acc:.2f}$\\pm${std_acc:.2f}")
        mean_acc = np.mean(final_y)
        std_acc = np.std(final_y,ddof = 1)
        print("Final:")
        print(f"mean acc:{mean_acc:.2f}$\\pm${std_acc:.2f}")


def draw_plot():
    import json
    import numpy as np

    for entry in list:
        data_files = os.listdir(os.path.join(log_dir, entry))
        y = []
        name = entry
        x = range(100)
        if 'FedGKD-Local' != name and "FedAvg" not in name:
            continue

        for i, data_file in enumerate(data_files):
            yi = []

            data_file = os.path.join(log_dir, entry, data_file, '0', 'log-1.json')
            datas = json.load(open(data_file))
            for i, data in enumerate(datas):
                if (i % 2) == 0:
                    yi.append(data['top1'])
            y.append(yi)
        y = np.array(y)
        y = np.mean(y, axis=0)
        plt.plot(x, y, label=name)

    # plt.savefig('pictures/cifar10_alpha1.pdf')
    plt.legend()
    plt.xlabel('#Communication rounds', fontsize=20)
    plt.ylabel('top-1 accuracy(%)', fontsize=20)
    plt.show()


stat_best_acc()
# stat_movement()
# draw_plot()
