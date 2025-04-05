from IPython.core.display import clear_output
from matplotlib import pyplot as plt


def plot(
    total_rewards_per_episode,
    total_loss_V,
    total_loss_Q1,
    total_loss_Q2,
    total_loss_PI,
    winning=None,
    plot_type=0,
):
    """
    Can be used to live-plot the rewards, losses, winning rate during training instead of e.g. Tensorboard.
    total_rewards_per_episode:  array containing the rewards per episode
    total_loss_V:               array containing the losses of the Value function
    total_loss_Q1:              array containing the losses of the Q1 function
    total_loss_Q2:              array containing the losses of the Q2 function
    total_loss_PI:              array containing the losses of the policy network
    winning:                    array containing the percentage of wins so far
    plot_type:      = 0:        only plot total rewards
                    = 1:        plot total_rewards and losses
                    = 2:        plot total rewards and winning fraction
                    = 3:        only plot winning fraction

    """
    clear_output(True)
    if plot_type == 0:
        plt.plot(range(len(total_rewards_per_episode)), total_rewards_per_episode)
    elif plot_type == 1:
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].plot(range(len(total_rewards_per_episode)), total_rewards_per_episode)
        axes[0, 0].set_title("Reward")
        axes[0, 1].plot(range(len(total_loss_V)), total_loss_V)
        axes[0, 1].set_title("V loss")

        axes[1, 0].plot(range(len(total_loss_Q1)), total_loss_Q1, c="r")
        axes[1, 0].plot(range(len(total_loss_Q2)), total_loss_Q2, c="b")
        axes[1, 0].set_title("Q losses")
        axes[1, 1].plot(range(len(total_loss_PI)), total_loss_PI)
        axes[1, 1].set_title("PI Loss")
    elif plot_type == 2:
        if winning is None:
            winning = []
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(range(len(total_rewards_per_episode)), total_rewards_per_episode)
        axes[0].set_title("Reward")
        axes[1].plot(range(len(winning)), winning)
        axes[1].set_title("Win fraction")
    elif plot_type == 3:
        plt.plot(range(len(winning)), winning)

    plt.show()
