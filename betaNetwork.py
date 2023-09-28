import numpy as np

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
import tensorflow as tf

# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
# from stable_baselines3.common.vec_env import VecVideoRecorder


def best_model(map_x, map_y, attack_range):
  model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1, 4, 4, 27))
  ,tf.keras.layers.Dense(24, activation='relu')
  ,tf.keras.layers.Dense(map_x*map_y*6*4*4*4*4*7*(attack_range**2), activation='relu')
#   ,tf.keras.layers.Softmax(map_x*map_y*6*4*4*4*4*7*(attack_range**2))
  ])
  return model

def action_map(map_x, map_y, attack_range):
    attack_targets = attack_range**2
    i = 0;
    result = [map_x*map_y*6*4*4*4*4*7*attack_targets]
    
    for src_unit in range(0, map_x*map_y-1):
        for action_type in range(0,5):
            for move_param in range(0,3):
                for harvest_param in range(0,3):
                    for return_param in range(0,3):
                        for produce_dir_param in range(0,3):
                            for produce_type_param in range(0,6):
                                for rel_attack_position in range(0, attack_targets-1):
                                    result.append([src_unit, action_type, move_param, harvest_param, return_param, produce_dir_param, produce_type_param, rel_attack_position])
                                    # result[i] = [src_unit, action_type, move_param, harvest_param, return_param, produce_dir_param, produce_type_param, rel_attack_position]
                                    i = i + 1
    return result
                      
map_x = 4
map_y = 4
attack_range = 3
model = best_model(map_x, map_y, attack_range)
a_map = action_map(map_x, map_y, attack_range)

envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.passiveAI for _ in range(1)],
    map_paths=["maps/4x4/base4x4.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
# envs = VecVideoRecorder(envs, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def sample(logits):
    # https://stackoverflow.com/a/40475357/6611317
    p = softmax(logits, axis=1)
    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    return choices.reshape(-1, 1)


envs.action_space.seed(0)
obs = envs.reset()
nvec = envs.action_space.nvec
done = False

for i in range(1000000):
    # envs.render()
    action_mask = envs.get_action_mask()
    action_mask = action_mask.reshape(-1, action_mask.shape[-1])
    action_mask[action_mask == 0] = -9e8
    
    # sample valid actions
    while not done:
        choices = model(100,obs)
        print(f'choices, shape: {choices.shape}')
        choices = choices.numpy()
        print(f'choices, shape: {choices.shape}')
        exit(0)
        action = a_map[max(range(len(a_)), key=lambda i: my_list[i])]
        print(f'action: {action}')
        next_obs, reward, done, info = envs.step(action)

envs.close()
