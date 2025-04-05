"""
D4RL references:
- https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/maze/point_maze.py
- https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/maze/maze_v4.py
"""
import cv2
import time

import gymnasium as gym
import numpy as np

from collections import deque
from gymnasium.spaces import Box
from gymnasium_robotics.envs.maze.maps import R, G, C
from rl_suite.envs import Observation
from gymnasium.core import Env


OPEN_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1],
    [1, C, C, C, C, 1],
    [1, C, C, C, C, 1],
    [1, C, C, C, C, 1],
    [1, 1, 1, 1, 1, 1]
]

U_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1],
    [1, C, C, C, C, 1],
    [1, 1, 1, 1, C, 1],
    [1, C, C, C, C, 1],
    [1, 1, 1, 1, 1, 1]
]

SMALL_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1],
    [1, C, C, C, C, 1],
    [1, C, 1, 1, C, 1],
    [1, C, 1, C, C, 1],
    [1, C, 1, 1, C, 1],
    [1, C, C, C, C, 1],
    [1, 1, 1, 1, 1, 1]
]

T_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, C, C, C, C, C, 1],
    [1, C, 1, 1, 1, C, 1],
    [1, C, C, 1, C, C, 1],
    [1, C, C, C, C, C, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

PLUS_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, C, C, C, C, C, 1],
    [1, C, C, 1, C, C, 1],
    [1, C, 1, 1, 1, C, 1],
    [1, C, C, 1, C, C, 1],
    [1, C, C, C, C, C, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

MEDIUM_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, C, C, 1, 1, C, C, 1],
    [1, C, C, 1, C, C, C, 1],
    [1, 1, C, C, C, 1, 1, 1],
    [1, C, C, 1, C, C, C, 1],
    [1, C, 1, C, C, 1, C, 1],
    [1, C, C, C, 1, C, C, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

LARGE_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, C, 0, 0, 0, 1, C, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, C, 0, 1, 0, 0, C, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, C, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, C, 0, C, 1, 0, C, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# MIN_TIME_MAP = [
#     [1, 1, 1, 1, 1, 1, 1],
#     [1, 0, R, 0, 0, R, 1],
#     [1, R, 1, 1, 1, 0, 1],
#     [1, G, 1, R, 0, R, 1],
#     [1, 1, R, 1, 1, 0, 1],
#     [1, R, 0, 0, 0, R, 1],
#     [1, 1, 1, 1, 1, 1, 1],
# ]

MIN_TIME_MAP = [
    [1, 1, 1, 1, 1, 1],
    [1, 0, R, R, R, 1],
    [1, R, 0, 1, 0, 1],
    [1, G, 1, 0, R, 1],
    [1, 1, R, 1, 0, 1],
    [1, R, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
]


class PointMaze(Env):
    all_maps = {"open": OPEN_DIVERSE_GR, "small": SMALL_MAZE_DIVERSE_GR, "medium": MEDIUM_MAZE_DIVERSE_GR,
                "large": LARGE_MAZE_DIVERSE_GR, "min_time": MIN_TIME_MAP, "U": U_DIVERSE_GR,
                "T": T_MAZE_DIVERSE_GR, "plus": PLUS_MAZE_DIVERSE_GR}
    """
    PointMaze_UMazeDense=v3 uses np.exp(-np.linalg.norm(a-b)) as reward 
    """
    def __init__(self, map_type="small", reward_type="sparse", reward=-1, 
                 use_image=False, img_history=3, render_mode=None) -> None:
        assert reward_type in ["sparse", "dense"], print(reward_type)
        assert map_type in self.all_maps, print(map_type)
        assert render_mode in ["human", "rgb_array", None], print(render_mode)
        
        self.timeout = 10000
        self.reward_type = reward_type
        self.use_image = use_image
        self.img_history = img_history
        if self.use_image:
            self._image_buffer = deque([], maxlen=img_history)
            if render_mode is None:
                render_mode = "rgb_array"
        print(f"point_maze_{map_type} with {reward_type} rewards. Visual task: {use_image}")

        self.render_mode = render_mode
        self.reward = reward
        
        self.env = gym.make("PointMaze_UMaze-v3", maze_map=self.all_maps[map_type], render_mode=render_mode)
        self._maze_map = self.all_maps[map_type]

        self._obs_dim = 4 if use_image else 8
        self._action_dim = 2
        self._img_dim = (160, 160)
        self._prev_action = np.zeros(2)
        self.steps = 0
    
    def set_seeds(self, seed):
        self.env.reset(seed=seed) # This is just to set the seed
        self.env.action_space.seed(seed)
        
    def make_obs(self, x):
        if self.use_image:
            obs = Observation()

            # Get new image
            new_img = self.get_image()
            self._image_buffer.append(new_img)
            obs.images = np.concatenate(self._image_buffer, axis=0)

            # N.B: Use only velocity as a part of proprioception obs
            obs.proprioception = np.zeros(self._obs_dim, dtype=np.float32)
            obs.proprioception[:2] = x['observation'].astype(np.float32)[2:]
            obs.proprioception[2:] = self._prev_action
        else:
            obs = np.zeros(self._obs_dim, dtype=np.float32)
            obs[:4] = x['observation'].astype(np.float32)
            obs[4:6] = x['desired_goal'].astype(np.float32)
            obs[6:] = self._prev_action
        return obs
    
    def get_image(self):
        assert self.use_image or self.render_mode=="rgb_array"
        img = self.render()
        img = cv2.resize(img, self._img_dim, interpolation = cv2.INTER_AREA)
        img = img[40:, :, :]
        img = np.transpose(img, [2, 0, 1])  # c, h, w
        return img

    def reset(self, randomize_target=True, **kwargs):
        options = {}
        if not randomize_target:
            # The goal is unchanged, only the agent spawns at a new location.
            goal_cell = self.env.maze.cell_xy_to_rowcol(self._goal_cell)
            if self._maze_map[goal_cell[1]][goal_cell[0]] != 1:
                options = {"goal_cell": goal_cell}
        
        x = self.env.reset(options=options)[0]
        self._goal_cell = x['desired_goal']

        if self.use_image:
            new_img = self.get_image()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)
        
        self._prev_action = np.zeros(self._action_dim)

        return self.make_obs(x), {}

    def step(self, action):
        next_x, r, _, _, info = self.env.step(action)
        self.steps += 1
        next_obs = self.make_obs(next_x)
        done = r
        truncated = self.steps == self.timeout
        self._prev_action = action[:]

        if self.reward_type == "sparse":
            reward = self.reward
        else:
            reward = -0.25 * np.linalg.norm(next_x['achieved_goal'] - next_x['desired_goal'], axis=-1)

        return next_obs, reward, done, truncated, info
    
    def render(self):
        if self.render_mode is not None:
            return self.env.render()

    @property
    def observation_space(self):
        return Box(shape=(self._obs_dim,), high=10, low=-10)

    @property
    def image_space(self):
        if not self.use_image:
            raise AttributeError(f'use_image={self.use_image}')

        image_shape = (3 * self.img_history, 120, 160)
        return Box(low=0, high=255, shape=image_shape)

    @property
    def proprioception_space(self):
        if not self.use_image:
            raise AttributeError(f'use_image={self._use_image}')
        
        return self.observation_space

    @property
    def action_space(self):
        return self.env.action_space
    
    def close(self):
        self.env.close()

def main():    
    seed = 42
    n_episodes = 100
    map_type = "open"
    reward_type = "sparse"
    use_image = False
    render_mode = None  # "human", "rgb_array"
    np.random.seed(seed)
    env = PointMaze(reward_type=reward_type, map_type=map_type, render_mode=render_mode, use_image=use_image)
    env.set_seeds(seed)
    # env = gym.make('PointMaze_UMaze-v3', maze_map=MIN_TIME_MAP, render_mode = "human")

    obs = env.reset(randomize_target=True)
    for i_ep in range(n_episodes):
        done = False
        ret = 0
        step = 0
        # First episode. Cannot use previous goal if it was never initialized.
        tic = time.time()
        while (not done and step < timeout):
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            ret += reward
            step += 1
            obs = next_obs

            if use_image:
                cv_img = obs.images[6:9, :, :]
                cv_img = np.moveaxis(cv_img, 0, -1).astype(np.uint8)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                # cv2.imshow(f"Point Maze: {map_type}", cv_img, )
                time.sleep(0.05)

                # if cv2.waitKey(1) == ord('q'):                
                #     break

            # if step % 100 == 0:
                # print(f"Obs: {obs[:4]}, action: {action}, reward: {reward}")
            env.render()

        if not done:
            obs = env.reset(randomize_target=False)
        else:
            obs = env.reset(randomize_target=True)
            
        print("Episode {} ended in {} steps with return {:.2f}. Done: {}. Render time: {:.2f}".format(
            i_ep+1, step, ret, done, time.time()-tic))
    
    env.close()


if __name__ == "__main__":
    main()
    