import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import clear_output


class TrainingHistory:
    def __init__(self):
        self.episode_rewards = []
        self.total_loss_V = []
        self.total_loss_Q1 = []
        self.total_loss_Q2 = []
        self.total_loss_PI = []
        self.winning = []

    def update(
        self,
        *,
        episode_reward: float = None,
        loss_v: float = None,
        loss_q1: float = None,
        loss_q2: float = None,
        loss_pi: float = None,
        won_episode: int = None,
    ):
        if episode_reward:
            self.episode_rewards.append(episode_reward)
        if loss_v:
            self.total_loss_V.append(loss_v)
        if loss_q1:
            self.total_loss_Q1.append(loss_q1)
        if loss_q2:
            self.total_loss_Q2.append(loss_q2)
        if loss_pi:
            self.total_loss_PI.append(loss_pi)
        if won_episode:
            self.winning.append(won_episode)

    def plot(self):
        """
        Can be used to live-plot the rewards, losses, winning rate during training instead of e.g. Tensorboard.
        """
        clear_output(True)
        mosaic = """
                AABB
                CCDD
                """
        if self.winning:
            mosaic = """
                    AABB
                    CCDD
                    EEEE
                    """

        fig, axd = plt.subplot_mosaic(mosaic, figsize=(8, 4))

        axd["A"].plot(range(len(self.episode_rewards)), self.episode_rewards)
        axd["A"].set_title("Reward")
        axd["B"].plot(range(len(self.total_loss_V)), self.total_loss_V)
        axd["B"].set_title("V loss")
        axd["C"].plot(range(len(self.total_loss_Q1)), self.total_loss_Q1, c="r")
        axd["C"].plot(range(len(self.total_loss_Q2)), self.total_loss_Q2, c="b")
        axd["C"].set_title("Q losses")
        axd["D"].plot(range(len(self.total_loss_PI)), self.total_loss_PI)
        axd["D"].set_title("PI Loss")

        if self.winning:
            axd["E"].plot(range(len(self.winning)), self.winning)
            axd["E"].set_title("Win fraction")
        plt.tight_layout()
        plt.show()

    def stdout(self):
        out_string = ""
        if self.episode_rewards:
            out_string += "Reward: " + str(self.episode_rewards[-1]) + " | "
        if self.total_loss_V:
            out_string += "V-loss: " + str(self.total_loss_V[-1]) + " | "
        if self.total_loss_Q1:
            out_string += "Q1-loss " + str(self.total_loss_Q1[-1]) + " | "
        if self.total_loss_Q2:
            out_string += "Q2-loss: " + str(self.total_loss_Q2[-1]) + " | "
        if self.total_loss_PI:
            out_string += "PI-loss: " + str(self.total_loss_PI[-1]) + " | "
        if self.winning:
            out_string += "Winning: " + str(self.winning[-1]) + " | "
        print(out_string)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action
