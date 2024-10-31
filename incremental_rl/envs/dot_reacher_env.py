import cv2
import torch
import time

import numpy as np

from gymnasium.core import Env
from gymnasium.spaces import Box


class DotReacherEnv(Env):
    def __init__(self, pos_tol=0.25, vel_tol=0.1, timeout=20000, penalty=-1, **kwargs):
        """ Continuous Action Space; Acceleration Control

        Args:
            pos_tol (float): Position tolerance - how close should the agent be to the the target.
            vel_tol (float): Velocity tolerance - how fast should the agent be moving when it's close to the target
            dt (float): Action cycle time
            timeout (int): Maximum episode length
        """
        self._pos_tol = pos_tol
        self._vel_tol = vel_tol
        self._timeout = timeout
        self.reward = penalty
        self.dt = 0.2

        super(DotReacherEnv, self).__init__()

        self._action_low = torch.tensor([[-1., -1.]])
        self._action_high = torch.tensor([[1., 1.]])
        self._pos_low = torch.tensor([[-1., -1.]])
        self._pos_high = torch.tensor([[1., 1.]])
        self.steps = 0

    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(2,))

    @property
    def observation_space(self):
        # TODO: Verify that min/max velocity are always within these bounds
        return Box(low=-10, high=10, shape=(4,))

    def reset(self, **kwargs):
        self.steps = 0
        self.pos = torch.rand((1, 2)) * (self._pos_high - self._pos_low) + self._pos_low
        self.vel = torch.zeros((1, 2))
        obs = torch.cat((self.pos, self.vel), 1)
        return obs.flatten().numpy(), {}

    def step(self, action):
        """

        Args:
            action: 2-D Tensor (vals between [-1, 1])

        Returns:

        """
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action.astype(np.float32)).view((1, -1))

        self.steps += 1

        # Clamp the action
        action = torch.clamp(action, min=self._action_low, max=self._action_high)

        # Acceleration control for smoothness
        self.pos = self.pos + self.vel * self.dt + 0.5 * action * self.dt ** 2
        self.vel[self.pos < self._pos_low] = -0.1 * self.vel[self.pos < self._pos_low]
        self.vel[self.pos > self._pos_high] = -0.1 * self.vel[self.pos > self._pos_high]
        self.pos = torch.clamp(self.pos, self._pos_low, self._pos_high)
        self.vel += action * self.dt

        # Observation
        next_obs = torch.cat((self.pos, self.vel), 1)

        # Reward
        reward = self.reward

        # Done
        reached_goal = torch.allclose(self.pos, torch.zeros(2), atol=self._pos_tol) and \
               torch.allclose(self.vel, torch.zeros(2), atol=self._vel_tol)
        truncated = self.steps == self._timeout
        terminated = reached_goal or truncated

        # Metadata
        info = {}

        return next_obs.flatten().numpy(), reward, terminated, truncated, info

    def render(self):
        pass

class DotReacherEasy(DotReacherEnv):
    def __init__(self, **kwargs):
        super().__init__(pos_tol=0.25, vel_tol=0.1, timeout=20000, penalty=-1)


class DotReacherHard(DotReacherEnv):
    def __init__(self, **kwargs):
        super().__init__(pos_tol=0.1, vel_tol=0.05, timeout=20000, penalty=-1)


def random_pi_dot_reacher():
    # Problem
    torch.manual_seed(3)

    # Env
    timeout = 20000
    env = DotReacherEnv(pos_tol=0.1, vel_tol=0.05, timeout=timeout)

    # Experiment
    EP = 50
    rets = []
    ep_lens = []
    # Slogs = []
    steps = 0
    for ep in range(EP):
        # Slogs.append([])
        obs = env.reset()
        # Slogs[-1].append(obs)
        ret = 0
        ep_steps = 0
        while True:
            # Take action
            A = torch.rand((1, 2))
            A = A * (env._action_high - env._action_low) + env._action_low
            # print(A)

            # Receive reward and next state
            next_obs, R, terminated, truncated, _ = env.step(A)
            # print("Step: {}, Obs: {}, Action: {}".format(steps, obs, A))

            # Log
            # Slogs[-1].append(next_obs)
            ret += R
            steps += 1
            ep_steps += 1

            # Termination
            if terminated or truncated:
                rets.append(ret)
                ep_lens.append(ep_steps)
                print("Episode: {}: # steps = {}, return = {:.2f}. Total Steps: {}".format(ep, ep_steps, ret, steps))
                break

            obs = next_obs

    # Random policy stats
    rets = np.array(rets)
    ep_lens = np.array(ep_lens)
    print("Mean:", np.mean(ep_lens))
    print("Standard Error:", np.std(ep_lens)/np.sqrt(len(ep_lens)-1))
    print("Median:", np.median(ep_lens))
    inds = np.where(ep_lens == env._timeout)
    print("Success Rate (%):", (1 - len(inds[0])/len(ep_lens))*100.)
    print("Max length:", max(ep_lens))
    print("Min length:", min(ep_lens))

    # Plotting
    # plt.plot(-100 * torch.tensor(rets))
    # plt.figure()
    #
    # colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:red", "tab:brown"]
    # for i in range(-min(30, EP), 0):
    #     color = colors[i % len(colors)]
    #     Slog = torch.cat(Slogs[i])
    #     for i in range(Slog.shape[0] - 1):
    #         plt.plot(Slog[i:i + 2, 0], Slog[i:i + 2, 1], alpha=(i + 1) / Slog.shape[0], color=color, marker='.')
    # plt.xlim([env._pos_low[0, 0], env._pos_high[0, 0]])
    # plt.ylim([env._pos_low[0, 1], env._pos_high[0, 1]])
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid()
    # plt.show()


if __name__ == '__main__':
    random_pi_dot_reacher()
    