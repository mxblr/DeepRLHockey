"""Script to train a SAC agent on the Laser Hockey Environment - Work in Progress"""

import typing

import gymnasium as gym
import hockey.hockey_env as h_env
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.v2.sac.sac import (
    NormalPolicyFunction,
    NormalPolicyFunctionConfig,
    SoftActorCritic,
    SoftActorCriticConfig,
    ValueFunction,
    ValueFunctionConfig,
)


def eval_agent_against_component(
    training_env: h_env.HockeyEnv,
    agent: typing.Type[nn.Module],
    opponent_agent: typing.Union[typing.Type[nn.Module], typing.Any],
    max_steps: int,
    render_env: bool = False,
) -> typing.Tuple[int, float]:
    """Evaluate the agent on one episode against an opponent

    Returns information about the winner (Winner == 0: draw, Winner == 1: sac_agent (left player), Winner == -1:
    opponent wins (right player)) and the average reward of the episode.

    :param training_env: The environment we evaluate the agent on
    :param agent: Our agent. Must expose function "act"
    :param opponent_agent: Opponent. Must expose function "act"
    :param max_steps: Maximum number of steps to take in the environment
    :param render_env: If True renders the game for humans
    :return: Returning an indicator about the winner (-1: opponent, 0: draw, 1: sac_agent) and the average reward
    across the episode
    """
    obs, info = training_env.reset()
    obs_agent2 = training_env.obs_agent_two()
    total_rewards: list = []
    for _ in range(max_steps):
        if render_env:
            training_env.render("human")
        a1 = agent.act(obs)
        a2 = opponent_agent.act(obs_agent2)
        obs, r, d, t, info = training_env.step(np.hstack([a1, a2]))
        obs_agent2 = training_env.obs_agent_two()
        total_rewards.append(r)
        if d or t:
            break
    winner: int = info["winner"]
    return winner, np.mean(total_rewards)


class RandomAgent:
    """An agent performing random steps"""

    def act(self, observation):  # noqa
        return np.random.uniform(-1, 1, 4)


def eval_agent(
    game_env: h_env.HockeyEnv, eval_agent: typing.Type[nn.Module], episodes_per_opponent: int = 5, max_steps: int = 250
) -> pd.DataFrame:
    """
    Evaluates the agent against different available opponents

    Opponents are:
    - random opponent
    - handcrafted weak
    - handcrafted strong
    - self play

    :param game_env: The hockey environment
    :param eval_agent: The agent to evaluate
    :param episodes_per_opponent:How often to play against an opponent
    :param max_steps: Maximum environment steps to take
    :return df_results: A pandas dataframe containing the results (columns: "episode", "opponent", "won", "reward")
    """
    results = []
    # random agent
    random_opponent = RandomAgent()
    for episode in range(episodes_per_opponent):
        episode_won, episode_reward = eval_agent_against_component(
            game_env, agent=eval_agent, opponent_agent=random_opponent, max_steps=max_steps
        )
        results.append((episode, "random_opponent", episode_won, episode_reward))

    # weak handcrafted
    weak_handcrafted_opponent = h_env.BasicOpponent(weak=True)
    for episode in range(episodes_per_opponent):
        episode_won, episode_reward = eval_agent_against_component(
            game_env, agent=eval_agent, opponent_agent=weak_handcrafted_opponent, max_steps=max_steps
        )
        results.append((episode, "weak_handcrafted_opponent", episode_won, episode_reward))

    # strong handcrafted
    strong_handcrafted_opponent = h_env.BasicOpponent(weak=False)
    for episode in range(episodes_per_opponent):
        episode_won, episode_reward = eval_agent_against_component(
            game_env, agent=eval_agent, opponent_agent=strong_handcrafted_opponent, max_steps=max_steps
        )
        results.append((episode, "strong_handcrafted_opponent", episode_won, episode_reward))

    # self play
    sac_agent_opponent = eval_agent
    for episode in range(episodes_per_opponent):
        episode_won, episode_reward = eval_agent_against_component(
            game_env, agent=eval_agent, opponent_agent=sac_agent_opponent, max_steps=max_steps
        )
        results.append((episode, "sac_agent_opponent", episode_won, episode_reward))

    df_results = pd.DataFrame(results, columns=["episode", "opponent", "won", "reward"])
    return df_results


class HockeyAdjustedActionSpace(h_env.HockeyEnv):
    """Custom environment

    Environment with action space split in half - to be aligned with SAC implementation

    Optionally, we can add further rewards (see get_reward and get_reward_agent_two)
    """

    def __init__(self, mode=h_env.Mode.NORMAL):
        super().__init__(mode=mode, keep_mode=True)
        self.action_space = gym.spaces.Box(-1, +1, (4,), dtype=np.float32)
        self.touched_puck = False
        self.agent_2_touched_puck = False

    def reset(self, one_starting=None, mode=None, seed=None, options=None):
        self.touched_puck = False
        return super(HockeyAdjustedActionSpace, self).reset(
            one_starting=one_starting, mode=mode, seed=seed, options=options
        )

    def get_reward(self, info):
        # additional rewards
        # - "reward_touch_puck"
        # - "reward_puck_direction"
        # - direct goal path blocked
        # - direct goal path
        # - puck was touched at all
        self.touched_puck = self.touched_puck or bool(info["reward_touch_puck"])

        reward = 0

        if self.done:
            if self.winner == 0:  # tie
                reward += 0
            elif self.winner == 1:  # you won
                reward += 10
            else:  # opponent won
                reward -= 10
        # reward = super(HockeyAdjustedActionSpace, self).get_reward(info)
        reward += 5 * info["reward_closeness_to_puck"]
        reward -= 1 - self.touched_puck * 0.1
        reward += info["reward_touch_puck"]
        reward += info["reward_puck_direction"]

        # reward closeness to goal of opponent
        dist_puck_to_goal = h_env.dist_positions(self.goal_player_2.position, self.puck.position)
        max_dist = 500.0 / h_env.SCALE
        max_reward = -10.0  # max (negative) reward through this proxy
        factor = max_reward / (max_dist * self.max_timesteps / 2)
        reward += dist_puck_to_goal * factor
        return reward

    def get_reward_agent_two(self, info_two):
        # reward = super(HockeyAdjustedActionSpace, self).get_reward_agent_two(info_two)
        self.agent_2_touched_puck = self.agent_2_touched_puck or bool(info_two["reward_touch_puck"])

        reward = 0

        if self.done:
            if self.winner == 0:  # tie
                reward += 0
            elif self.winner == 1:  # you won
                reward -= 10
            else:  # opponent won
                reward += 10
        # reward = super(HockeyAdjustedActionSpace, self).get_reward(info)
        reward += 5 * info_two["reward_closeness_to_puck"]
        reward -= 1 - self.touched_puck * 0.1
        reward += info_two["reward_touch_puck"]
        reward += info_two["reward_puck_direction"]

        # reward closeness to goal of opponent
        dist_puck_to_goal = h_env.dist_positions(self.goal_player_1.position, self.puck.position)
        max_dist = 500.0 / h_env.SCALE
        max_reward = -10.0  # max (negative) reward through this proxy
        factor = max_reward / (max_dist * self.max_timesteps / 2)
        reward += dist_puck_to_goal * factor
        return reward


def fill_buffer_with_agent(sac_agent_entity, observation, opponent_agent, self_play_agent):
    """
    Filling the buffer based on an agent and optionally an opponent

    :param sac_agent_entity: The SAC trainer (containing the buffer)
    :param observation: The current observation of the environment
    :param opponent_agent: The opponent - performing actions of agent 2
    :param self_play_agent: The player - performing actions of agent 1
    :return ob_new: the newly observed environment state
    :return env_done: indicator - is the env done?
    :return reward: reward of current environment step
    """
    a = self_play_agent.act(observation)

    # if we play a game with an opponent, sample the opponent action
    store_a = a
    ob_opponent_agent = sac_agent_entity.env.obs_agent_two()
    a_opponent_agent = opponent_agent.act(ob_opponent_agent)
    a = np.hstack([a, a_opponent_agent])

    # execute the action and receive updated observations and reward
    ob_new, reward, terminated, truncated, *_info = sac_agent_entity.env.step(a)
    env_done = terminated or truncated
    # store the action and observations
    sac_agent_entity.buffer.store(observation, store_a, float(reward), ob_new, env_done)
    return ob_new, env_done, reward


if __name__ == "__main__":
    env = HockeyAdjustedActionSpace()
    config = SoftActorCriticConfig(
        q_fct_config=ValueFunctionConfig(hidden_layers=[256, 256]),
        v_fct_config=ValueFunctionConfig(hidden_layers=[256, 256]),
        pi_fct_config=NormalPolicyFunctionConfig(
            hidden_layers=[256, 256], output_activation_function_log_std=torch.tanh
        ),
        discount=0.96,
        tau=0.005,
        batch_size=128,
        alpha="auto",
        learning_rate_v=1e-4,
        learning_rate_pi=1e-4,
        learning_rate_q=1e-4,
        learning_rate_alpha=1e-5,
        buffer_size=int(1e5),
        max_grad_norm=None,
        normalize_actions=False,
    )

    # set up the trainer
    sac_agent = SoftActorCritic(config, policy_function=NormalPolicyFunction, value_function=ValueFunction, env=env)

    # fill buffer with actions from strong basic opponent
    ob, _info = sac_agent.env.reset()
    episode_reward = 0
    for _step in range(int(1e5)):
        ob_new, env_done, reward = fill_buffer_with_agent(
            sac_agent_entity=sac_agent,
            observation=ob,
            opponent_agent=h_env.BasicOpponent(weak=True),
            self_play_agent=h_env.BasicOpponent(weak=False),
        )
        episode_reward += reward
        if env_done:
            print(f"Burn in episode reward {episode_reward}")
            episode_reward = 0
            ob, _info = sac_agent.env.reset()

    # train networks
    history = sac_agent.train(
        epochs=10_000,
        max_steps=1,
        env_steps=500,
        grad_steps=32,
        n_burn_in_steps=0,
        n_log_epochs=1,
        log_output="stdout",
        opponent_agent=h_env.BasicOpponent(weak=True),
    )

    df_eval = eval_agent(game_env=env, eval_agent=sac_agent, episodes_per_opponent=10, max_steps=1000)

    _ = eval_agent_against_component(
        training_env=env,
        agent=sac_agent,
        opponent_agent=h_env.BasicOpponent(weak=True),
        max_steps=1000,
        render_env=True,
    )
