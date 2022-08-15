import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

COLORS = list(mcolors.TABLEAU_COLORS)


def plot_data(title: str, data: list, labels: list, path="images/"):
    print(path + title)
    epochs = range(1, len(data[0]) + 1)
    for i, data_group in enumerate(data):
        plt.plot(epochs, data_group, COLORS[i], label=labels[i])
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    plt.savefig(f"{path+title}.png", bbox_inches="tight")
    plt.close()


def plot_hist(title: str, data: list, path="images/"):
    print(path + title)
    plt.hist(data, bins=243)
    plt.title(title)
    plt.xlabel("Result")
    plt.ylabel("Count")
    plt.legend()
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    plt.savefig(f"{path+title}.png", bbox_inches="tight")
    plt.close()


def plot_q_values(title: str, data: list, action_labels: list, path="images/"):
    print(path + title)
    print(f"data dimensions: {data.shape}")
    print(f"action_labels: {len(action_labels)}")
    M = data.shape[0]
    amount = M
    epochs = range(amount)
    fig, axs = plt.subplots(len(action_labels))
    barWidth = 1
    for i in range(len(action_labels)):
        axs[i].plot(
            epochs,
            data[:, i],
            color=COLORS[i],
            label=action_labels[i],
            # width=barWidth,
        )
        axs[i].grid(True)
        axs[i].set_title(f"{action_labels[i]}")
        # axs[i].set_xlabel('Epochs')
        # axs[i].set_ylabel('Frequency')
        axs[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(hspace=1)
    fig.savefig(f"{path+title}.png", bbox_inches="tight")
    # plt.title(title)
    # plt.xlabel('Epochs')
    # plt.ylabel('Frequency')
    # plt.legend()
    plt.close()


def plot_frequencies(title: str, data: list, action_labels: list, path="images/"):
    print(path + title)
    print(f"data dimensions: {data.shape}")
    print(f"action_labels: {len(action_labels)}")
    M = data.shape[0]
    amount = M
    epochs = range(amount)
    fig, axs = plt.subplots(1)
    fig.suptitle("Frequencies")
    barWidth = 1
    if len(action_labels) == 5:
        axs.bar(
            epochs,
            data[:, 0][:amount],
            color=COLORS[0],
            label=action_labels[0],
            width=barWidth,
        )
        axs.bar(
            epochs,
            data[:, 1][:amount],
            bottom=data[:, 0][:amount],
            color=COLORS[1],
            label=action_labels[1],
            width=barWidth,
        )
        axs.bar(
            epochs,
            data[:, 2][:amount],
            bottom=[i + j for i, j in zip(data[:, 0][:amount], data[:, 1][:amount])],
            color=COLORS[2],
            label=action_labels[2],
            width=barWidth,
        )
        axs.bar(
            epochs,
            data[:, 3][:amount],
            bottom=[
                i + j + k
                for i, j, k in zip(
                    data[:, 0][:amount], data[:, 1][:amount], data[:, 2][:amount]
                )
            ],
            color=COLORS[3],
            label=action_labels[3],
            width=barWidth,
        )
        axs.bar(
            epochs,
            data[:, 4][:amount],
            bottom=[
                i + j + k + l
                for i, j, k, l in zip(
                    data[:, 0][:amount],
                    data[:, 1][:amount],
                    data[:, 2][:amount],
                    data[:, 3][:amount],
                )
            ],
            color=COLORS[4],
            label=action_labels[4],
            width=barWidth,
        )
    else:
        raise ValueError(f"{len(action_labels)} Number of actions not supported")
    axs.grid(True)
    axs.set_title(f"word values")
    # axs.set_xlabel('Epochs')
    # axs.set_ylabel('Frequency')
    axs.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #   fancybox=True, shadow=True, ncol=5)
    # axs.legend()
    fig.subplots_adjust(hspace=1)
    fig.savefig(f"{path+title}.png", bbox_inches="tight")
    # plt.title(title)
    # plt.xlabel('Epochs')
    # plt.ylabel('Frequency')
    # plt.legend()
    plt.close()
