import os
import torch
from pylab import *


def draw_acc_three(All, GR, model_type, attr, xticks, aug_type, x_label=None):
    for i in range(len(GR)):
        width = 12
        height = 8
        plt.figure(figsize=(width, height))
        fontsize = 28
        if x_label is None:
            x_label = "Gaussian Noise Variance"
        y_label = "Acc"
        marker_size = 10
        line_width = 3

        var = np.arange(len(All[0][i]))

        plt.grid()

        plt.plot(var, torch.mean(All[0][i], dim=-1).numpy(), label="Overall Acc", marker='v', markersize=marker_size, linewidth=line_width,
                 color="green")
        plt.fill_between(var, torch.mean(All[0][i], dim=-1).numpy() - torch.std(All[0][i], dim=-1).numpy(), torch.mean(All[0][i], dim=-1).numpy() + torch.std(All[0][i], dim=-1).numpy(), color="green", alpha=0.2)

        plt.plot(var, torch.mean(All[1][i], dim=-1).numpy(), label="Minority Acc", marker='s', markersize=marker_size, linewidth=line_width,
                 color="red")
        plt.fill_between(var, torch.mean(All[1][i], dim=-1).numpy() - torch.std(All[1][i], dim=-1).numpy(), torch.mean(All[1][i], dim=-1).numpy() + torch.std(All[1][i], dim=-1).numpy(), color="red", alpha=0.2)

        plt.plot(var, torch.mean(All[2][i], dim=-1).numpy(), label="Majority Acc", marker='o', markersize=marker_size, linewidth=line_width,
                 color="orange")
        plt.fill_between(var, torch.mean(All[2][i], dim=-1).numpy() - torch.std(All[2][i], dim=-1).numpy(), torch.mean(All[2][i], dim=-1).numpy() + torch.std(All[2][i], dim=-1).numpy(), color="orange", alpha=0.2)

        plt.legend(fontsize=fontsize-3, loc=0, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
        plt.xticks(var, xticks, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # plt.title(f"CelebA {model_type} {attr} Gaussian Rate: {GR[i]}", fontsize=fontsize)

        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(f"graphs/acc_{model_type}_{attr}_{aug_type}_gr{GR[i]}.png")
        plt.show()
        plt.close()


if __name__ == "__main__":
    model = "resnet18"
    attr = "BlondHair"
    aug_type = "Crop"
    res_all = {"2022": {}, "2021": {}, "2020": {}}
    res_minor = {"2022": {}, "2021": {}, "2020": {}}
    res_major = {"2022": {}, "2021": {}, "2020": {}}
    # res_all = {"0.1": {}, "0.3": {}, "0.5": {}}
    # res_minor = {"0.1": {}, "0.3": {}, "0.5": {}}
    # res_major = {"0.1": {}, "0.3": {}, "0.5": {}}

    res_all = [[], [], []]
    res_major = [[], [], []]
    res_minor = [[], [], []]

    GR_list = ["0.1", "0.3", "0.5"]
    # GV_list = ["0.000001", "0.00001", "0.0001", "0.001", "0.01", "0.1", "1", "10"]
    GV_list = ["100", "120", "140", "160", "180", "200", "220"]
    # GV_list = ["0", "30", "60", "90", "120", "150", "180"]
    for i, gr in enumerate(GR_list):
        res_all[i] = [[] for j in range(len(GV_list))]
        res_minor[i] = [[] for j in range(len(GV_list))]
        res_major[i] = [[] for j in range(len(GV_list))]
        for j, gv in enumerate(GV_list):
            for seed in ["2020", "2021", "2022"]:
                # file_name = model + "_" + attr + "_gv" + gv + "_gr" + gr + "_seed" + seed + ".log"
                file_name = model + "_" + attr + "_" + aug_type + "_gv" + gv + "_gr" + gr + "_seed" + seed + ".log"
                with open(file_name, "r") as file:
                    for line in file:
                        pass
                    try:
                        last_line = line.split(",")
                        acc_all = float(last_line[0][-6:])
                        acc_minor = float(last_line[1])
                        acc_major = float(last_line[2])
                    except Exception:
                        print(file_name)
                        sys.exit()

                    res_all[i][j].append(acc_all)
                    res_minor[i][j].append(acc_minor)
                    res_major[i][j].append(acc_major)

                    print(acc_all, acc_minor, acc_major)

    res_all = torch.tensor(res_all)
    res_minor = torch.tensor(res_minor)
    res_major = torch.tensor(res_major)

    # draw_acc_three([res_all, res_minor, res_major], GR_list, model, attr, ["1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1", "10"])
    draw_acc_three([res_all, res_minor, res_major],
                   GR_list, model, attr,
                   ["100", "120", "140", "160", "180", "200", "220"],
                   aug_type, "Random Crop Size")



