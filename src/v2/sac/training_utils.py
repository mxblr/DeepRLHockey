import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


class TrainingHistory:
    def __init__(self, log_target: str = "plot"):
        """
        Simple training history for writing or plotting training statistics

        Receives losses, rewards and wins.
        :param log_target: Options are "plot" for plotting the loss curves as a matplotlib.pyplot object or "stdout" for
                           writing training statistics to stdout.
        """
        self.episode_rewards = []
        self.total_loss_V = []
        self.total_loss_Q1 = []
        self.total_loss_Q2 = []
        self.total_loss_PI = []
        self.total_loss_alpha = []
        self.winning = []

        log_options = ["plot", "stdout"]
        if log_target not in log_options:
            raise ValueError(f"TrainingHistory does not support {log_target=} - choose one of {log_options}.")
        self.log_target = log_target

    def update(
        self,
        *,  # only allow keyword arguments
        episode_reward: float = None,
        loss_v: float = None,
        loss_q1: float = None,
        loss_q2: float = None,
        loss_pi: float = None,
        loss_alpha: float = None,
        won_episode: int = None,
    ):
        """
        Saves the current training statistics

        :param episode_reward: Reward of the current training step.
        :param loss_v: loss_v of the current training step.
        :param loss_q1: loss_q1 of the current training step.
        :param loss_q2: loss_q2 of the current training step.
        :param loss_pi: loss_pi of the current training step.
        :param loss_alpha: loss_alpha of the current training step.
        :param won_episode: Information if the current game was won.
        """
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
        if loss_alpha:
            self.total_loss_alpha.append(loss_alpha)
        if won_episode:
            self.winning.append(won_episode)

    def __call__(self):
        """Plot or write the current training statistic to stdout or pyplot."""
        if self.log_target == "plot":
            self.plot()
        elif self.log_target == "stdout":
            self.stdout()
        else:
            raise NotImplementedError(f"Output {self.log_target!r} is not implemented.")

    def plot(self):
        """
        Can be used to live-plot the rewards, losses, winning rate during training instead of e.g. Tensorboard.
        """
        clear_output(True)
        mosaic = """
                AABB
                CCDD
                """
        if self.winning or self.total_loss_alpha:
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
            axd["E"].plot(range(len(self.winning)), self.winning, c="g")
            axd["E"].set_title("Win fraction")
        elif self.total_loss_alpha:
            axd["E"].plot(range(len(self.total_loss_alpha)), self.total_loss_alpha, c="b")
            axd["E"].set_title("Alpha Loss")

        plt.tight_layout()
        plt.show()

    def stdout(self):
        """
        Can be used to live-plot the rewards, losses, winning rate during training instead of e.g. Tensorboard.
        """

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
    """Translates between tanh activated actions (from SAC) and the actual action space"""

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Called by env.step, turns actions from range [-1, 1] into actions in range [action_space.low, action_space.high]

        :param action: Action to be performed in range [-1, 1]
        :return: action in action_space range of environment
        """
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """
        Turns actions from range [action_space.low, action_space.high] into actions in range [-1, 1]

        :param action: Action in environments action-space range
        :return: Action in range [-1, 1]
        """

        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action
