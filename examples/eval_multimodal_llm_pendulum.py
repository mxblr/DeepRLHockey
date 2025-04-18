"""
This script contains code to evaluate gemini models on solving Pendulum based on images.
"""

import json
import os
from enum import Enum
from getpass import getpass
from time import sleep, time

import gymnasium as gym
import numpy as np
import torch
from google import genai
from google.genai import types
from gymnasium.wrappers import RecordVideo
from PIL import Image
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
        burn_in_steps: int = 0,
        teacher: TeacherModel = TeacherModel.RANDOM,
        burn_in_episodes: int = 1,
        llm_retry_steps: int = 3,
    ):
        self.llm_system_prompt = llm_system_prompt
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.max_environment_steps = max_environment_steps
        self.sleep_between_steps = sleep_between_steps
        self.burn_in_steps = burn_in_steps
        self.teacher = teacher
        self.burn_in_episodes = burn_in_episodes
        self.burn_in_steps_per_episode = self.burn_in_steps // self.burn_in_episodes if self.burn_in_episodes else 0
        self.llm_retry_steps = llm_retry_steps


class ActionSchema(BaseModel):
    torque: float = Field(None, ge=-2, le=2)


if __name__ == "__main__":
    config = GeminiConfig(
        llm_system_prompt="""Your task is to solve the inverted pendulum swingup problem.  
        The system consists of a pendulum attached at one end to a fixed point, and the other end being free. 
        The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into an 
        upright position, with its center of gravity right above the fixed point. The torque is a float in the range of
        -2 to 2. You will receive a reward after each action, your goal is to maximize the reward.
        Additionally, you receive the current state of the pendulum with the  x-y coordinates of the pendulum’s free 
        end and its angular velocity: x in range [-1, 1], y in range [-1, 1], angular velocity in range [-8, 8].
        """,
        max_environment_steps=50,
        sleep_between_steps=5,
        llm_temperature=0.0,
        burn_in_steps=50,
        llm_model="gemini-2.5-flash-preview-04-17",
        teacher=TeacherModel.SAC,
        burn_in_episodes=0,
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
    )

    env = gym.make("Pendulum-v1", render_mode="rgb_array").unwrapped

    # we will save the video to the directory video_evals
    name_prefix = f"gemini-{str(int(time()))}-{config.llm_model}"
    env = RecordVideo(env, video_folder="video_evals", name_prefix=name_prefix, episode_trigger=lambda x: True)

    teacher_model = None
    if config.teacher == TeacherModel.SAC:
        teacher_model = torch.load("examples/models/sac_agent.pt", weights_only=False)

    contents = []

    # Burn in phase
    # Teach Gemini how to solve Pendulum, by putting some (observation, action, rewards) in its context window
    for episode in range(config.burn_in_episodes):
        ob, _ = env.reset()
        text_part = types.Part(text=f"reward: NaN, x: {ob[0]}, y: {ob[1]}, angular velocity: {ob[2]}")

        reward = None
        for _episode_step in range(config.burn_in_steps_per_episode):
            # append observation and img to the context window
            contents.extend([types.Content(role="user", parts=[text_part]), Image.fromarray(env.render())])

            if teacher_model:
                # sample an action from SAC
                action = teacher_model.act_greedy(torch.as_tensor(ob).view(1, teacher_model.dim_obs))
            else:
                # sample a random action
                action = env.action_space.sample()

            # append the action to the context window
            contents.append(
                types.Content(role="model", parts=[types.Part(text=json.dumps({"torque": float(action[0])}))])
            )

            # take step
            ob, reward, terminated, truncated, *_ = env.step(np.array(action))

            # append environment state to the context window
            text_part = types.Part(text=f"reward: {reward}, x: {ob[0]}, y: {ob[1]}, angular velocity: {ob[2]}")

            # end episode if we won
            if env_done := (terminated or truncated):
                break
        print(f"Reward at end of burn-in episode {episode} is {reward}")

    # time to test gemini
    ob, _ = env.reset()
    text_part = types.Part(text=f"reward: NaN, x: {ob[0]}, y: {ob[1]}, angular velocity: {ob[2]}")
    total_reward = 0
    llm_call_failure_count = 0
    for step in range(config.max_environment_steps):
        # potentially sleep a few seconds (cause I am LLM API poor and do not want to run into quota limits)
        sleep(config.sleep_between_steps)

        # append observation and img to the context window
        contents.extend([types.Content(role="user", parts=[text_part]), Image.fromarray(env.render())])

        # get outputs for contents
        try:
            response = client.models.generate_content(model=config.llm_model, config=genai_config, contents=contents)
        except Exception as e:
            print(e)

            if llm_call_failure_count < config.llm_retry_steps:
                # if we have retries left, continue
                llm_call_failure_count += 1
                continue
            else:
                break
        # append response to contents
        contents.append(types.Content(role="model", parts=[types.Part(text=response.text)]))

        # take action
        action = float(response.parsed.torque)
        ob, reward, terminated, truncated, *_info = env.step(np.array([action]))

        # append environment state to the
        text_part = types.Part(text=f"reward: {reward}, x: {ob[0]}, y: {ob[1]}, angular velocity: {ob[2]}")

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
        experiment_data["teacher_model"] = str(experiment_data["teacher"].value)
        json.dump(config.__dict__, f)
