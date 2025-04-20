"""
This script contains code to evaluate gemini models on solving Pendulum based on images.
"""

import json
import os
import typing
from enum import Enum
from getpass import getpass
from time import sleep, time

import gymnasium as gym
import numpy as np
import torch
from google import genai
from google.genai import types
from gymnasium.wrappers import RecordVideo
from pydantic import BaseModel, Field

from src.v2.sac.sac import SoftActorCritic  # noqa

# set up environment
if os.getenv("GOOGLE_API_KEY") is None:
    os.environ["GOOGLE_API_KEY"] = getpass("Please provide your: 'Google API Key': ")


class TeacherModel(Enum):
    RANDOM = "random"
    SAC = "Soft Actor Critic"


class GeminiConfig:
    def __init__(
        self,
        *,
        llm_system_prompt: str,
        llm_model: str = "gemini-2.0-flash",
        llm_temperature: float = 0.0,
        max_environment_steps: int = 100,
        sleep_between_steps: int = 0,
        burn_in_steps_per_episode: int = 0,
        teacher: TeacherModel = TeacherModel.RANDOM,
        burn_in_episodes: int = 1,
        llm_retry_steps: int = 3,
        thinking_budget: typing.Union[int, None] = None,
    ):
        self.llm_system_prompt = llm_system_prompt
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.max_environment_steps = max_environment_steps
        self.sleep_between_steps = sleep_between_steps
        self.burn_in_steps_per_episode = burn_in_steps_per_episode
        self.teacher = teacher
        self.burn_in_episodes = burn_in_episodes
        self.llm_retry_steps = llm_retry_steps
        self.thinking_budget = thinking_budget


class ActionSchema(BaseModel):
    torque: float = Field(None, ge=-100, le=100)


class LLMEnvironmentObserver:
    def __init__(
        self,
        llm_obs_range: typing.Tuple[typing.List[int], typing.List[int], type, typing.List[str]],
        llm_action_range: typing.Tuple[int, int, type, str],
        environment,
    ):
        # action
        self.llm_action_space_low = llm_action_range[0]
        self.llm_action_space_high = llm_action_range[1]
        self.llm_action_space_type = llm_action_range[2]
        self.llm_action_space_diff = self.llm_action_space_high - self.llm_action_space_low
        self.llm_action_name = llm_action_range[3]
        # observation
        self.llm_obs_space_low = llm_obs_range[0]
        self.llm_obs_space_high = llm_obs_range[1]
        self.llm_obs_space_type = llm_obs_range[2]
        self.llm_obs_space_diff = [
            high - low for (low, high) in zip(self.llm_obs_space_low, self.llm_obs_space_high, strict=True)
        ]
        self.llm_obs_names = llm_obs_range[3]
        # original spaces
        self.env = environment
        # dtype_mapper = {np.dtypes.Float32DType: float}
        # action
        self.original_action_space_low = self.env.action_space.low
        self.original_action_space_high = self.env.action_space.high
        self.original_action_space_diff = self.original_action_space_high - self.original_action_space_low
        self.original_action_space_type = float  # TODO
        # observation
        self.original_obs_space = self.env.observation_space
        self.original_obs_space_low = self.env.observation_space.low
        self.original_obs_space_high = self.env.observation_space.high
        self.original_obs_space_diff = [
            high - low for (low, high) in zip(self.original_obs_space_low, self.original_obs_space_high, strict=True)
        ]
        self.original_obs_space_type = float  # TODO

    @staticmethod
    def _transform(value, current_low, current_range, target_range, target_low, target_high, target_type):
        value_normalized = (value - current_low) / current_range
        value_target = value_normalized * target_range + target_low
        value_target = np.clip(value_target, target_low, target_high)
        return target_type(value_target)

    def to_llm_action(self, act):
        """From original range of the env to the action range used for the llm"""  #
        return self._transform(
            value=act,
            current_low=self.original_action_space_low,
            current_range=self.original_action_space_diff,
            target_high=self.llm_action_space_high,
            target_low=self.llm_action_space_low,
            target_range=self.llm_action_space_diff,
            target_type=self.llm_action_space_type,
        )

    def from_llm_action(self, act):
        """From action ranged used by llm to the original range of the env"""
        return self._transform(
            value=act,
            current_low=self.llm_action_space_low,
            current_range=self.llm_action_space_diff,
            target_high=self.original_action_space_high,
            target_low=self.original_action_space_low,
            target_range=self.original_action_space_diff,
            target_type=self.original_action_space_type,
        )

    def to_llm_observation(self, observation):
        """From original range of the env to the observations range used for the llm"""
        llm_observation = []
        for value, current_low, current_diff, target_high, target_low, target_diff in zip(
            observation,
            self.original_obs_space_low,
            self.original_obs_space_diff,
            self.llm_obs_space_high,
            self.llm_obs_space_low,
            self.llm_obs_space_diff,
            strict=True,
        ):
            llm_observation.append(
                self._transform(
                    value=value,
                    current_low=current_low,
                    current_range=current_diff,
                    target_high=target_high,
                    target_low=target_low,
                    target_range=target_diff,
                    target_type=self.llm_obs_space_type,
                )
            )
        return llm_observation

    def from_llm_observation(self, observation):
        """From observations ranged used by llm to the original range of the env"""
        env_observation = []
        for value, current_low, current_diff, target_high, target_low, target_diff in zip(
            observation,
            self.llm_obs_space_low,
            self.llm_obs_space_diff,
            self.original_obs_space_high,
            self.original_obs_space_low,
            self.original_obs_space_diff,
            strict=True,
        ):
            env_observation.append(
                self._transform(
                    value=value,
                    current_low=current_low,
                    current_range=current_diff,
                    target_high=target_high,
                    target_low=target_low,
                    target_range=target_diff,
                    target_type=self.original_obs_space_type,
                )
            )
        return env_observation

    def get_observation_string(self, obs):
        obs_llm = self.to_llm_observation(obs)
        observation_string = ""
        for observation, name in zip(obs_llm, self.llm_obs_names, strict=True):
            observation_string += f" {name}: {observation}"
        return observation_string

    def get_action_string(self, act, from_llm: bool = True):
        if not from_llm:
            act = self.to_llm_action(act)
        return "{" + f"{self.llm_action_name}: {act}" + "}"

    def get_reward_string(self, reward):
        return f"Episode reward: {int(reward)}"


if __name__ == "__main__":
    config = GeminiConfig(
        llm_system_prompt="""Your task is to solve the inverted pendulum swingup problem.  
        The system consists of a pendulum attached at one end to a fixed point, and the other end being free. 
        The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into an 
        upright position, with its center of gravity right above the fixed point. The upright position has values 
        x = 100, y= 0 and angular velocity = 0. The torque is an integer in the range of 0 to 100. 
        After each action, you will see the current state of the pendulum with the x-y coordinates of the pendulum’s 
        free end x in range [-100, 100] and y in range [-100, 100], and its angular velocity in range [-100, 100].
        """,
        max_environment_steps=50,
        sleep_between_steps=5,
        llm_temperature=0.0,
        burn_in_steps_per_episode=75,
        llm_model="gemini-2.0-flash",
        teacher=TeacherModel.SAC,
        burn_in_episodes=5,
        thinking_budget=None,
    )

    # set up google AI client
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    genai_config = genai.types.GenerateContentConfig(
        system_instruction=config.llm_system_prompt,
        temperature=config.llm_temperature,
        safety_settings=[
            genai.types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH")
        ],
        response_mime_type="application/json",
        response_schema=ActionSchema,
        thinking_config=types.ThinkingConfig(thinking_budget=config.thinking_budget)
        if config.thinking_budget is not None
        else None,
    )

    env = gym.make("Pendulum-v1", render_mode="rgb_array").unwrapped
    normalizer = LLMEnvironmentObserver(
        llm_action_range=(0, 100, int, "torque"),
        llm_obs_range=([-100, -100, -100], [100, 100, 100], int, ["x", "y", "angular velocity"]),
        environment=env,
    )

    # we will save the video to the directory video_evals
    name_prefix = f"llm-gemini-{str(int(time()))}-{config.llm_model}"
    env = RecordVideo(env, video_folder="video_evals", name_prefix=name_prefix, episode_trigger=lambda x: True)

    teacher_model = None
    if config.teacher == TeacherModel.SAC:
        teacher_model = torch.load("examples/models/sac_agent.pt", weights_only=False)

    contents = []

    # Burn in phase
    # Teach Gemini how to solve Pendulum, by putting some (observation, action, rewards) in its context window
    for episode in range(config.burn_in_episodes):
        ob, _ = env.reset()
        text_part = types.Part(text=normalizer.get_observation_string(obs=ob))

        episode_reward = 0
        reward = None
        for _episode_step in range(config.burn_in_steps_per_episode):
            # append observation and img to the context window
            contents.extend([types.Content(role="user", parts=[text_part])])

            if teacher_model:
                # sample an action from SAC
                action = teacher_model.act_greedy(torch.as_tensor(ob).view(1, teacher_model.dim_obs))
            else:
                # sample a random action
                action = env.action_space.sample()

            # append the action to the context window
            contents.append(
                types.Content(
                    role="model", parts=[types.Part(text=normalizer.get_action_string(action, from_llm=False))]
                )
            )

            # take step
            ob, reward, terminated, truncated, *_ = env.step(np.array(action))
            episode_reward += reward
            # append environment state to the context window
            text_part = types.Part(text=normalizer.get_observation_string(obs=ob))

            # end episode if we won
            if env_done := (terminated or truncated):
                break

        contents.append(
            types.Content(role="user", parts=[types.Part(text=normalizer.get_reward_string(episode_reward))])
        )
        print(f"Reward at end of burn-in episode {episode} is {reward}")

    # time to test gemini
    ob, _ = env.reset()
    text_part = types.Part(text=normalizer.get_observation_string(obs=ob))
    total_reward = 0
    llm_call_failure_count = 0
    for step in range(config.max_environment_steps):
        # potentially sleep a few seconds (cause I am LLM API poor and do not want to run into quota limits)
        sleep(config.sleep_between_steps)

        # append observation and img to the context window
        contents.extend([types.Content(role="user", parts=[text_part])])

        # get outputs for contents
        try:
            response = client.models.generate_content(model=config.llm_model, config=genai_config, contents=contents)

        except genai.errors.APIError as e:
            if e.code == 429:  # resource exhausted
                if llm_call_failure_count < config.llm_retry_steps:
                    sleep(e["details"][0].get("retryDelay"))
                    llm_call_failure_count += 1
                    continue
                else:
                    break
            else:
                print(e)

                if llm_call_failure_count < config.llm_retry_steps:
                    # if we have retries left, continue
                    llm_call_failure_count += 1
                    continue
                else:
                    break

        except Exception as e:
            print(e)

            if llm_call_failure_count < config.llm_retry_steps:
                # if we have retries left, continue
                llm_call_failure_count += 1
                continue
            else:
                break

        # take action
        action = response.parsed
        if action is None:
            print("Action was None")
            if llm_call_failure_count < config.llm_retry_steps:
                # if we have retries left, continue
                llm_call_failure_count += 1
                continue
            else:
                break
        action = int(action.torque)
        # append response to contents
        # contents.append(types.Content(role="model", parts=[types.Part(text=response.text)]))
        contents.append(
            types.Content(role="model", parts=[types.Part(text=normalizer.get_action_string(action, from_llm=True))])
        )
        print(f"model reponse = {action}")
        action = normalizer.from_llm_action(action)

        ob, reward, terminated, truncated, *_info = env.step(np.array([action]))

        # append environment state to the
        text_part = types.Part(text=normalizer.get_observation_string(obs=ob))

        total_reward += reward
        print(f"{step}: Performed {action=} and received {reward=}. {total_reward=}")
        if env_done := (terminated or truncated):
            print(f"Episode won: {total_reward=}, {reward=}")
            break
    else:
        print(f"Episode lost: {total_reward=}")

    # save config file to disk alongside video
    env.close()
    with open(os.path.join("video_evals", name_prefix + ".json"), "w") as f:
        experiment_data = config.__dict__
        experiment_data["total_reward"] = total_reward
        experiment_data["teacher"] = str(config.teacher.value)
        json.dump(config.__dict__, f)
