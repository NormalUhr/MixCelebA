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
    y_label = "Test Loss"

    # plt.ylim(94.0, 95.5)
    x = np.array(range(8))
    loss_1 = np.array([0.2845, 0.2924, 0.2900, 0.2756, 0.2681, 0.2558, 0.2521, 0.2616])
    loss_2 = np.array([0.3430, 0.3562, 0.3475, 0.3216, 0.3086, 0.2911, 0.2832, 0.2922])
    loss_3 = np.array([0.2341, 0.2376, 0.2406, 0.23609, 0.2333, 0.2254, 0.2253, 0.2353])

    plt.plot(x, loss_1, color="orange", marker='o', markevery=1, linestyle='-',
             linewidth=linewidth,
             markersize=markersize, label="Overall Loss")
    plt.plot(x, loss_2, color="green", marker='o', markevery=1, linestyle='-',
             linewidth=linewidth,
             markersize=markersize, label="Minority Loss")
    plt.plot(x, loss_3, color="red", marker='o', markevery=1, linestyle='-',
             linewidth=linewidth,
             markersize=markersize, label="Majority Loss")

    plt.legend(fontsize=fontsize - 6, loc=0, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(x, p_name, rotation=0, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.title("", fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(f"graphs/rebuttal_loss_study.pdf")
    plt.show()
