import numpy as np

# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
from stable_baselines3.common.vec_env import VecVideoRecorder

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.workerRushAI for _ in range(1)],
    map_paths=["maps/8x8/basesWorkers8x8.xml"],
    reward_weight=np.array([10.0, 2.0, 2.0, 0.2, 2.0, 6.0]),
)
envs = VecVideoRecorder(envs, "videos", record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)

class CustomEnv(gym.Env):
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self, envs):
        super().__init__()
        self._envs = envs
        self._envs.reset()

        self._envs.action_space.seed(0)
        self.action_space = self._envs.action_space
        self.observation_space = self._envs.observation_space

    def step(self, action):
     
        action_mask = self._envs.get_action_mask()
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])
        action_mask[action_mask == 0] = -9e8

        return self._envs.step(action)

    def reset(self):
        return self._envs.reset()

    def render(self):
        self._envs.render()

    def close(self):
        self._envs.close()


#Life is easy no. Let at tweake koden bagom. 

env = CustomEnv(envs)
model = PPO("MlpPolicy", env, verbose=2,device = "cpu")
model.learn(total_timesteps=2500000)

'''Outcommented er til at gemme og loade models no use lige nu'''
model.save("testModelRts") #Gem model
del model
model = PPO.load("testModelRts-PPO", device = "cpu")

obs = env.reset()

for i in range(100000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
