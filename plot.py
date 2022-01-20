import matplotlib.pyplot as plt
import os

COLORS = ["g", "b", "m", "r", "y"]


def plot_data(title: str, data: list, labels: list, path="assets/"):
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


def plot_frequencies(
    title: str, data: list, hand_labels: list, action_labels: list, path="assets/"
):
    print(path + title)
    print(f"data dimensions: {len(data)}, {len(data[0])}, {len(data[0][0])}")
    M = len(data[0][0])
    amount = M
    epochs = range(amount)  # range(1,len(data[0][0])+1)
    fig, axs = plt.subplots(len(data))
    fig.suptitle("Frequencies")
    barWidth = 1
    for i, hand in enumerate(data):
        if len(action_labels) == 5:
            axs[i].bar(
                epochs,
                hand[0][:amount],
                color=COLORS[0],
                label=action_labels[0],
                width=barWidth,
            )
            axs[i].bar(
                epochs,
                hand[1][:amount],
                bottom=hand[0][:amount],
                color=COLORS[1],
                label=action_labels[1],
                width=barWidth,
            )
            axs[i].bar(
                epochs,
                hand[2][:amount],
                bottom=[i + j for i, j in zip(hand[0][:amount], hand[1][:amount])],
                color=COLORS[2],
                label=action_labels[2],
                width=barWidth,
            )
            axs[i].bar(
                epochs,
                hand[3][:amount],
                bottom=[
                    i + j + k
                    for i, j, k in zip(
                        hand[0][:amount], hand[1][:amount], hand[2][:amount]
                    )
                ],
                color=COLORS[3],
                label=action_labels[3],
                width=barWidth,
            )
            axs[i].bar(
                epochs,
                hand[4][:amount],
                bottom=[
                    i + j + k + l
                    for i, j, k, l in zip(
                        hand[0][:amount],
                        hand[1][:amount],
                        hand[2][:amount],
                        hand[3][:amount],
                    )
                ],
                color=COLORS[4],
                label=action_labels[4],
                width=barWidth,
            )
        elif len(action_labels) == 4:
            axs[i].bar(
                epochs,
                hand[0][:amount],
                color=COLORS[0],
                label=action_labels[0],
                width=barWidth,
            )
            axs[i].bar(
                epochs,
                hand[1][:amount],
                bottom=hand[0][:amount],
                color=COLORS[1],
                label=action_labels[1],
                width=barWidth,
            )
            axs[i].bar(
                epochs,
                hand[2][:amount],
                bottom=[i + j for i, j in zip(hand[0][:amount], hand[1][:amount])],
                color=COLORS[2],
                label=action_labels[2],
                width=barWidth,
            )
            axs[i].bar(
                epochs,
                hand[3][:amount],
                bottom=[
                    i + j + k
                    for i, j, k in zip(
                        hand[0][:amount], hand[1][:amount], hand[2][:amount]
                    )
                ],
                color=COLORS[3],
                label=action_labels[3],
                width=barWidth,
            )
        elif len(action_labels) == 3:
            axs[i].bar(
                epochs,
                hand[0][:amount],
                color=COLORS[0],
                label=action_labels[0],
                width=barWidth,
            )
            axs[i].bar(
                epochs,
                hand[1][:amount],
                bottom=hand[0][:amount],
                color=COLORS[1],
                label=action_labels[1],
                width=barWidth,
            )
            axs[i].bar(
                epochs,
                hand[2][:amount],
                bottom=[i + j for i, j in zip(hand[0][:amount], hand[1][:amount])],
                color=COLORS[2],
                label=action_labels[2],
                width=barWidth,
            )
        elif len(action_labels) == 2:
            axs[i].bar(
                epochs,
                hand[0][:amount],
                color=COLORS[0],
                label=action_labels[0],
                width=barWidth,
            )
            axs[i].bar(
                epochs,
                hand[1][:amount],
                bottom=hand[0][:amount],
                color=COLORS[1],
                label=action_labels[1],
                width=barWidth,
            )
        else:
            raise ValueError(f"{len(action_labels)} Number of actions not supported")
        axs[i].grid(True)
        axs[i].set_title(f"word {hand_labels[i]}")
        # axs[i].set_xlabel('Epochs')
        # axs[i].set_ylabel('Frequency')
        axs[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #   fancybox=True, shadow=True, ncol=5)
    # axs.legend()
    fig.subplots_adjust(hspace=1)
    fig.savefig(f"{path+title}.png", bbox_inches="tight")
    # plt.title(title)
    # plt.xlabel('Epochs')
    # plt.ylabel('Frequency')
    # plt.legend()
    plt.close()
