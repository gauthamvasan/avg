import torch
import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from threading import Lock


class SACReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, capacity, batch_size):
        self.batch_size = batch_size
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, capacity
        self.lock = Lock()

        size_of_buffer = ((((self.observations.size * self.observations.itemsize) + \
                            (self.next_observations.size * self.next_observations.itemsize) + \
                            (self.actions.size * self.actions.itemsize) + (8 * capacity)) / 1024) / 1024)
        print("Size of replay buffer: {:.2f}MB".format(size_of_buffer))

    def add(self, obs, action, next_obs, reward, done):
        with self.lock:
            self.observations[self.ptr] = obs
            self.next_observations[self.ptr] = next_obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.ptr = (self.ptr+1) % self.max_size
            self.size = min(self.size+1, self.max_size)

    def sample(self):
        with self.lock:
            idxs = np.random.randint(0, self.size, size=self.batch_size)
            observations = torch.from_numpy(self.observations[idxs, :])
            next_observations = torch.from_numpy(self.next_observations[idxs, :])
            actions = torch.from_numpy(self.actions[idxs])
            rewards = torch.from_numpy(self.rewards[idxs])
            dones = torch.from_numpy(self.dones[idxs])
            return (observations, actions, next_observations, rewards, dones)

    def __len__(self):
        return self.size


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class SACPixelBuffer:
    """ Buffer to store environment transitions. """
    def __init__(self, image_shape, action_shape, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size

        self.images = np.zeros((capacity, *image_shape), dtype=np.uint8)            
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.count = 0

        size_of_buffer = (((((self.images.size * self.images.itemsize) + \
            (self.actions.size * self.actions.itemsize) + (8 * capacity)) / 1024) / 1024)/ 1024)
        print("Size of replay buffer: {:.2f}GB".format(size_of_buffer))

    def add(self, image, action, next_image, reward, done):
        # N.B: next_image is unused here; 
        self.images[self.idx] = image
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.count = self.capacity if self.full else self.idx

    def sample(self):
        idxs = np.random.randint(0, self.count-1, size=min(self.count-1, self.batch_size))

        images = torch.as_tensor(self.images[idxs]).float()
        next_images = torch.as_tensor(self.images[idxs+1]).float()
 
        actions = torch.as_tensor(self.actions[idxs])
        rewards = torch.as_tensor(self.rewards[idxs])
        dones = torch.as_tensor(self.dones[idxs])

        return images, actions, next_images, rewards, dones


class SACRADBuffer(object):
    """ Buffer to store environment transitions. """
    def __init__(self, image_shape, proprioception_shape, action_shape, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size

        self.images = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.propris = np.zeros((capacity, *proprioception_shape), dtype=np.float32)       

        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.count = 0

        size_of_buffer = (((((self.images.size * self.images.itemsize) + (self.propris.size * self.propris.itemsize) + \
            (self.actions.size * self.actions.itemsize) + (8 * capacity)) / 1024) / 1024)/ 1024)
        logging.info("Size of replay buffer: {:.2f}GB".format(size_of_buffer))

    def add(self, image, propri, action, reward, done):        
        self.images[self.idx] = image        
        self.propris[self.idx] = propri

        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.count = self.capacity if self.full else self.idx

    def sample(self):
        idxs = np.random.randint(0, self.count-1, size=min(self.count-1, self.batch_size))
        images = torch.as_tensor(self.images[idxs]).float()
        next_images = torch.as_tensor(self.images[idxs+1]).float()
        propris = torch.as_tensor(self.propris[idxs]).float()
        next_propris = torch.as_tensor(self.propris[idxs+1]).float()        
        actions = torch.as_tensor(self.actions[idxs])
        rewards = torch.as_tensor(self.rewards[idxs])
        dones = torch.as_tensor(self.dones[idxs])

        return images, propris, actions, next_images, next_propris, rewards, dones
    