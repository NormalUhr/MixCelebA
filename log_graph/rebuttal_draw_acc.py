from pylab import *

if __name__ == "__main__":

    # Canvas setting
    width = 14
    height = 12
    markersize = 20
    linewidth = 4
    markevery = 1
    fontsize = 50
    alpha = 0.7

    plt.rcParams['font.sans-serif'] = 'Times New Roman'

    import seaborn as sns
    sns.set_theme()

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)

    plt.grid(visible=True, which='major', linestyle='-', linewidth=4)
    plt.minorticks_on()
    plt.grid(visible=True, which='minor')
    plt.rcParams['font.sans-serif'] = 'Times New Roman'

    p_name = np.array(["0.01", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35"])

    x_label = "Minority Ratio"
    y_label = "Test Accuracy"

    # plt.ylim(94.0, 95.5)
    x = np.array(range(8))
    loss_1 = np.array([0.8882, 0.8882, 0.8882, 0.8903, 0.8903, 0.8989, 0.9032, 0.8989])
    loss_2 = np.array([0.8628, 0.8651, 0.8651, 0.8721, 0.8814, 0.8837, 0.8953, 0.8953])
    loss_3 = np.array([0.9100, 0.9080, 0.9080, 0.9060, 0.8980, 0.9120, 0.9100, 0.9020])

    plt.plot(x, loss_1, color="orange", marker='o', markevery=1, linestyle='-',
             linewidth=linewidth,
             markersize=markersize, label="Overall Acc")
    plt.plot(x, loss_2, color="green", marker='o', markevery=1, linestyle='-',
             linewidth=linewidth,
             markersize=markersize, label="Minority Acc")
    plt.plot(x, loss_3, color="red", marker='o', markevery=1, linestyle='-',
             linewidth=linewidth,
             markersize=markersize, label="Majority Acc")

    plt.legend(fontsize=fontsize - 6, loc=0, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(x, p_name, rotation=0, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.title("", fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(f"graphs/rebuttal_acc_study.pdf")
    plt.show()
