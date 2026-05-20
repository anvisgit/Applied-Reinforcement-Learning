import os 
import gymnasium as gym
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from stable_baselines3 import ppo
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy 


env_name='CartPole-v0'
env=gym.make("CartPole-v1", render_mode="rgb_array")

eps=5
for ep in range(1,eps+1):
    state=env.reset()
    done=False
    score=0

    while not done:

        frame = env.render()
        clear_output(wait=True)
        plt.imshow(frame)
        plt.axis("off")
        display(plt.gcf())
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()

