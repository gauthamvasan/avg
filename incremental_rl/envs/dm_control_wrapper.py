import dm_control
import torch
import numpy as np

from dm_control import suite
from gymnasium.spaces import Box
from gymnasium.core import Env


class DMControl(Env):
    def __init__(self, domain, task, **kwargs):
        self.env = suite.load(domain_name=domain, task_name=task)
        self.domain = domain
        self.task = task
        
        self.rgb_array = kwargs.get('render_mode', '') == "rgb_array"
            
        # Observation space
        self._obs_dim = 0
        for key, val in self.env.observation_spec().items():
            if val.shape:
                self._obs_dim += val.shape[0]
            else:
                self._obs_dim += 1
        
        # Action space
        self._action_dim = self.env.action_spec().shape[0]    
    
    def make_obs(self, x):
        obs = []
        for _, val in x.items():
            obs.append(val.ravel())
        return np.concatenate(obs)

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            self.env = suite.load(domain_name=self.domain, task_name=self.task, task_kwargs={'random': kwargs['seed']})
        time_step = self.env.reset()
        obs = self.make_obs(time_step.observation)
        return obs, {}
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        x = self.env.step(action)

        reward = x.reward
        terminated = x.last()
        truncated = False
        info = {}
        next_obs = self.make_obs(x.observation)

        return next_obs, reward, terminated, truncated, info

    @property
    def observation_space(self):
        return Box(shape=(self._obs_dim,), high=10, low=-10)

    @property
    def image_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')

        image_shape = (3 * self._image_buffer.maxlen, 100, 120)
        return Box(low=0, high=255, shape=image_shape)

    @property
    def proprioception_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')
        
        return self.observation_space

    @property
    def action_space(self):
        return Box(shape=(self._action_dim,), high=1, low=-1)

    def render(self):
        if self.rgb_array:
            rgb_array = self.env.physics.render(camera_id=0)
            return rgb_array
        
        self.env.render()
    

if __name__ == "__main__":
    seed = 42    
    EP = 10 
    env = DMControl(domain="finger", task="turn_easy")   
    #### Reproducibility
    np.random.seed(seed)
    env.reset(seed=seed)
    ####

    rets = []
    ep_lens = []
    for i in range(EP):
        obs = env.reset()
        terminated = False
        steps = 0
        ret = 0
        while not terminated:
            action = np.random.uniform(-1, 1, size=env._action_dim)
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
            # print(next_obs, reward, terminated)
            steps += 1 
            ret += reward
        rets.append(ret)
        ep_lens.append(steps)
        print("Episode: {} ended in {} steps with return: {}".format(i+1, steps, ret))

    # Random policy stats
    rets = np.array(rets)
    ep_lens = np.array(ep_lens)
    print("Mean: {:.2f}".format(np.mean(rets)))
    print("Standard Error: {:.2f}".format(np.std(rets) / np.sqrt(len(rets) - 1)))
    print("Median: {:.2f}".format(np.median(rets)))
    print("Max length:", max(ep_lens))
    print("Min length:", min(ep_lens))
