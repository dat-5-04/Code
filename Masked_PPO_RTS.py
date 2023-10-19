import gym
import numpy as np
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


#File and map management
map_size = "8x8"
folder_path = f"{map_size}_maskedPPO"



def mask_fn(env: gym.Env) -> np.ndarray:
    action_mask = env._envs.get_action_mask()
    return action_mask

envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2048,
    render_theme=2,
    ai2s=[microrts_ai.passiveAI for _ in range(1)],
    map_paths=[ f"maps/{map_size}/basesWorkers{map_size}.xml"],
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
        next_obs, reward, done, info = envs.step(action)
        info_dict = {k: v for d in info for k, v in d.items()} #convert list of dicts to a dict to match gym API

        return next_obs, reward, done, info_dict

    def reset(self):
        return self._envs.reset()

    def render(self):
        self._envs.render()

    def close(self):
        self._envs.close()

def initTrainedModels(env):
    files = os.listdir(folder_path)
    valid_files = [f for f in files if f.endswith(".zip") and f[:-4].isdigit()]

    if valid_files:
        print("Loading and continuing from previous sessions")
        iteration = max(map(lambda x: int(x[:-4]), valid_files))
        return MaskablePPO.load(f"./{folder_path}/{iteration}", env, device="cpu"), iteration + 1 #+1 for next in line
    else:
        print("No previous sessions found")
        return MaskablePPO("MlpPolicy", env, verbose=2, device="cpu"), 0


#env creation with action mask wrapper as well
env = CustomEnv(envs)
env = ActionMasker(env, mask_fn)

#training
model, iteration = initTrainedModels(env)
while True:
    model.learn(total_timesteps=4*2048)
    model.save(f"./{folder_path}/{iteration}")
