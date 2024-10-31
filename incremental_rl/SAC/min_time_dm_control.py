"""
DeepMind control suite tasks implemented in minimum-time formulation
- https://github.com/deepmind/dm_control/tree/main/dm_control
"""

import torch
import gymnasium as gym
import dm_env
import os, cv2

import numpy as np

from dm_control.suite.utils import randomizers
from dm_control.suite.reacher import Reacher, Physics
from dm_control.rl.control import flatten_observation
from dm_control import suite
from dm_control.rl import control
from dm_control.suite import common
from dm_control.utils import io as resources
from gym.spaces import Box
from collections import deque
from tqdm import tqdm


class Observation:
    def __init__(self):
        self.images = None
        self.metadata = None
        self.proprioception = None
        


class DMControlBaseEnv:
    def __init__(self):
        pass
    
    def reset(self):
        raise NotImplemented
    
    def step(self, action):
        raise NotImplemented
    
    @property
    def observation_space(self):
        return Box(shape=(self._obs_dim,), high=10, low=-10)

    @property
    def proprioception_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')
        
        return self.observation_space

    @property
    def action_space(self):
        return Box(shape=(self._action_dim,), high=1, low=-1)

    def render(self):
        self.env.render()
    
    

class BallInCupWrapper(DMControlBaseEnv):
    """ Minimum-time variant of Ball in a cup env """
    def __init__(self, seed, timeout, reward=-1, use_image=False, img_history=3):
        """ Outputs state transition data as torch arrays """
        self.env = suite.load(domain_name="ball_in_cup", task_name="catch", task_kwargs={'random': seed, 'time_limit': float('inf')})
        self.timeout = timeout
        self.reward = reward
        self._obs_dim = 8 if not use_image else 4
        self._action_dim = 2
        self.steps = 0
        self.img_dim = [84, 84]

        self._use_image = use_image
        
        if use_image:
            self._image_buffer = deque([], maxlen=img_history)
            print("Visual ball in cup")
        else:
            print('Non visual ball in cup')

    def make_obs(self, x):
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        if not self._use_image:
            obs[:4] = x.observation['position'].astype(np.float32)
            obs[4:8] = x.observation['velocity'].astype(np.float32)
        else:
            obs[:4] = x.observation['velocity'].astype(np.float32)
        return obs

    def _get_new_img(self):
        img = self.env.physics.render()
        img = cv2.resize(img, self.img_dim, interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, [2, 0, 1])  # c, h, w
        return img

    def reset(self):
        self.steps = 0
        if self._use_image:
            obs = Observation()
            obs.proprioception = self.make_obs(self.env.reset())

            new_img = self._get_new_img()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            obs = self.make_obs(self.env.reset())

        return obs, {}

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        x = self.env.step(action)
        self.steps += 1

        reward = self.reward
        done = x.reward
        truncated = self.steps == self.timeout
        info = {}

        if self._use_image:
            next_obs = Observation()
            next_obs.proprioception = self.make_obs(x)
            new_img = self._get_new_img()
            self._image_buffer.append(new_img)
            next_obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            next_obs = self.make_obs(x)        
        
        return next_obs, reward, done, truncated, info

    @property
    def image_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')

        image_shape = tuple([3 * self._image_buffer.maxlen] + self.img_dim)
        return Box(low=0, high=255, shape=image_shape)


class ReacherWrapper(DMControlBaseEnv):
    """ Minimum-time variant of reacher env with 3 modes: Easy, Hard,  """
    def __init__(self, seed, timeout, reward=-1, mode="easy", use_image=False, img_history=3):
        """ Outputs state transition data as torch arrays """
        assert mode in ["easy", "hard", "torture"]

        if mode == "torture":
            physics = Physics.from_xml_string(*ReacherWrapper.get_modified_model_and_assets())
            task = Reacher(target_size=.001, random=seed)
            self.env = control.Environment(physics, task, time_limit=float('inf'), **{})
        else:
            self.env = suite.load(domain_name="reacher", task_name=mode, task_kwargs={'random': seed, 'time_limit': float('inf')})

        self._obs_dim = 4 if use_image else 6
        self._action_dim = 2
        self.img_dim = [84, 84]
        
        self.timeout = timeout
        self.reward = reward
        self._use_image = use_image
        self.steps = 0
        
        if use_image:
            self._image_buffer = deque([], maxlen=img_history)
            print(f'Visual dm reacher {mode}')
        else:
            print(f'Non visual dm reacher {mode}')

    def make_obs(self, x):
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        obs[:2] = x.observation['position'].astype(np.float32)
        obs[2:4] = x.observation['velocity'].astype(np.float32)

        if not self._use_image: # this should be inferred from image
            obs[4:6] = x.observation['to_target'].astype(np.float32)
        
        return obs

    @staticmethod
    def get_modified_model_and_assets():
        """Returns a tuple containing the model XML string and a dict of assets."""
        PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
        return resources.GetResource(os.path.join(PARENT_DIR, 'envs/reacher_small_finger.xml')), common.ASSETS

    def _get_new_img(self):
        img = self.env.physics.render()
        img = cv2.resize(img, self.img_dim, interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, [2, 0, 1])  # c, h, w
        return img

    def _initialize_episode_random_agent_only(self):
        """Sets the state of the environment at the start of each episode."""
        self.env.physics.named.model.geom_size['target', 0] = self.env.task._target_size
        randomizers.randomize_limited_and_rotational_joints(self.env.physics, self.env.task.random)

        super(Reacher, self.env.task).initialize_episode(self.env.physics)

    def _reset_agent_only(self):
        """Starts a new episode and returns the first `TimeStep`."""
        self.env._reset_next_step = False
        self.env._step_count = 0
        with self.env.physics.reset_context():
            self._initialize_episode_random_agent_only()

        observation = self.env.task.get_observation(self.env.physics)
        if self.env._flat_observation:
            observation = flatten_observation(observation)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=observation)

    def reset(self, randomize_target=True):
        self.steps = 0
        if self._use_image:
            obs = Observation()
            if randomize_target:
                obs.proprioception = self.make_obs(self.env.reset())
            else:
                obs.proprioception = self.make_obs(self._reset_agent_only())

            new_img = self._get_new_img()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            if randomize_target:
                obs = self.make_obs(self.env.reset())
            else:
                obs = self.make_obs(self._reset_agent_only())

        return obs, {}

    def step(self, action):
        self.steps += 1
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        x = self.env.step(action)        

        reward = self.reward
        done = x.reward
        truncated = self.steps == self.timeout
        info = {}

        if self._use_image:
            next_obs = Observation()
            next_obs.proprioception = self.make_obs(x)
            new_img = self._get_new_img()
            self._image_buffer.append(new_img)
            next_obs.images = np.concatenate(self._image_buffer, axis=0)
        else:
            next_obs = self.make_obs(x)
            
        return next_obs, reward, done, truncated, info

    @property
    def image_space(self):
        if not self._use_image:
            raise AttributeError(f'use_image={self._use_image}')

        image_shape = tuple([3 * self._image_buffer.maxlen] + self.img_dim)
        return Box(low=0, high=255, shape=image_shape)
   

def random_policy_hits_vs_timeout():
    total_steps = 20000
    timeouts = [1, 2, 5, 10, 25, 50, 100, 500, 1000, 5000]

    # Env
    envs = ['dm reacher easy', 'dm reacher hard', 'ball in cup']
    envs = ['dm reacher hard']
    for env_s in envs:
        steps_record = open(f"{env_s}_steps_record.txt", 'w')
        hits_record = open(f"{env_s}_random_stat.txt", 'w')

        for timeout in tqdm(timeouts):
            for seed in range(30):
                torch.manual_seed(seed)
                np.random.seed(seed)

                if env_s == 'ball in cup':
                    env = BallInCupWrapper(seed)
                elif env_s == 'dm reacher torture':
                    env = ReacherWrapper(seed=seed, mode="torture")
                elif env_s == 'dm reacher hard':
                    env = ReacherWrapper(seed=seed, mode="hard")
                elif env_s == 'dm reacher easy':
                    env = ReacherWrapper(seed=seed, mode="easy")
                else:
                    raise NotImplementedError()

                if not hasattr(env, "_action_dim"):
                    env._action_dim = env.action_spec().shape[0]

                steps_record.write(f"timeout={timeout}, seed={seed}: ")
                # Experiment
                hits = 0
                steps = 0
                epi_steps = 0
                env.reset()
                while steps < total_steps:
                    action = np.random.normal(size=env.action_space.shape)

                    # Receive reward and next state            
                    _, _, done, _ = env.step(action)
                    
                    # print("Step: {}, Next Obs: {}, reward: {}, done: {}".format(steps, next_obs, reward, done))

                    # Log
                    steps += 1
                    epi_steps += 1

                    # Termination
                    if done or epi_steps == timeout:
                        if 'dm reacher' in env_s:
                            env.reset(randomize_target=done)
                        else:
                            env.reset()
                            
                        epi_steps = 0

                        if done:
                            hits += 1
                        else:
                            steps += 20
                            
                        steps_record.write(str(steps)+', ')

                steps_record.write('\n')
                hits_record.write(f"timeout={timeout}, seed={seed}: {hits}\n")
        
        steps_record.close()
        hits_record.close()

if __name__ == '__main__':
    # for domain_name, task_name in suite.ALL_TASKS: # suite.BENCHMARKING
    #     print(domain_name, texit()ask_name)
    #     env = suite.load(domain_name, task_name)

    # m = ManipulatorWrapper(task_name="insert_ball", seed=3)

    # interaction("manipulator", "bring_ball")
    # interaction("acrobot", "swingup_sparse")

    # visualize_behavior("manipulator", "bring_ball")
    # visualize_behavior("ball_in_cup", "catch")
    # visualize_behavior("reacher", "hard")
    # visualize_behavior("acrobot", "swingup_sparse")

    # r = ReacherWrapper(seed=1)
    # random_policy_stats()
    # ranndom_policy_hits_vs_timeout()
    # env = BallInCupWrapper(1, 1000, use_image=True)
    # env = ReacherWrapper(seed=1, timeout=50, use_image=True)
    # obs = env.reset()
    # img = obs.images

    # print(img.shape)
    # img_to_show = np.transpose(img, [1, 2, 0])
    # img_to_show = img_to_show[:,:,-3:]
    # cv2.imshow('', img_to_show)
    # cv2.waitKey(0)

    # for t in range(1000):
    #     randomize_target = t % 100 == 0
            
    #     next_obs = env.reset(randomize_target=randomize_target)
    #     next_img = next_obs.images
    #     img_to_show = np.transpose(next_img, [1, 2, 0])
    #     img_to_show = img_to_show[:,:,-3:]
    #     cv2.imshow('', img_to_show)
    #     cv2.waitKey(50)

    env = ReacherWrapper(seed=3)
