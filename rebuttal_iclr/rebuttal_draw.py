from matplotlib import pyplot as plt
import torch
import seaborn as sns


if __name__ == "__main__":
    width = 14
    height = 12
    markersize = 45
    linewidth = 5
    markevery = 1
    fontsize = 55
    alpha = 0.5
    marker = "."
    colors = ['green', 'red', 'orange']

    male_data = torch.load("singular_male.pth", map_location="cpu")
    female_data = torch.load("singular_female.pth", map_location="cpu")
    total_data = torch.load("singular_total.pth", map_location="cpu")

    male_data = torch.sqrt(male_data)
    female_data = torch.sqrt(female_data)
    total_data = torch.sqrt(total_data)

    plt.figure(figsize=(width, height))
    sns.set_theme()
    plt.grid(visible=True, which='major', linestyle='-', linewidth=4)
    plt.grid(visible=True, which='minor')
    plt.minorticks_on()
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.plot(male_data.numpy(), label="Majority", linewidth=4, color="red", linestyle="--", alpha=1)

    plt.xlabel("Order", fontsize=fontsize)
    plt.ylabel("Singular Value", fontsize=fontsize)
    plt.legend(loc="upper right", fontsize=fontsize, shadow=True, fancybox=True)
    # plt.xscale("log")
    # plt.xlim([0, 256])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("graphs/male_singular_value.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(width, height))
    sns.set_theme()
    plt.grid(visible=True, which='major', linestyle='-', linewidth=4)
    plt.grid(visible=True, which='minor')
    plt.minorticks_on()
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.plot(total_data.numpy(), label="Total", linewidth=4, color="blue", linestyle="--", alpha=1)

    plt.xlabel("Order", fontsize=fontsize)
    plt.ylabel("Singular Value", fontsize=fontsize)
    plt.legend(loc="upper right", fontsize=fontsize, shadow=True, fancybox=True)
    # plt.xscale("log")
    # plt.xlim([0, 256])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("graphs/total_singular_value.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(width, height))
    sns.set_theme()
    plt.grid(visible=True, which='major', linestyle='-', linewidth=4)
    plt.grid(visible=True, which='minor')
    plt.minorticks_on()
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.plot(female_data.numpy(), label="Minority", linewidth=4, color="green", linestyle="--", alpha=1.0)

    plt.xlabel("Order", fontsize=fontsize)
    plt.ylabel("Singular Value", fontsize=fontsize)
    plt.legend(loc="upper right", fontsize=fontsize, shadow=True, fancybox=True)
    # plt.xscale("log")
    # plt.xlim([0, 256])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("graphs/female_singular_value.png")
    plt.show()
    plt.close()

