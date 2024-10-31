import torch
import time, cv2

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
from incremental_rl.utils import orthogonal_weight_init


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        self.temperature = 1.   # This can be learned too

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.contiguous().view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints
    

class ImageEncoder(nn.Module):
    def __init__(self, image_shape, device):
        super().__init__()
        self.device = device        
        assert len(image_shape) == 3
        self.cnn_phi_dim = 32 * 35 * 35
        self.repr_dim = 64

        self.conv = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU()
        )

        # self.trunk = nn.Sequential(
        #     nn.Linear(self.cnn_phi_dim, self.repr_dim),
        #     nn.LayerNorm(self.repr_dim),
        #     nn.Tanh()
        # )
        self.ss = SpatialSoftmax(height=35, width=35, channel=32, data_format='NCHW')

        self.apply(orthogonal_weight_init)
        self.to(device)

    def forward(self, image, detach=False):               
        # Transpose the tensor to the required shape (batch_size, channels, height, width)        
        # image = image.permute(0, 3, 1, 2).to(self.device)

        image = image / 255.0 - 0.5
        h = self.conv(image)
        h = torch.reshape(h, (h.shape[0], -1))

        
        h = self.ss(h)

        if detach:
            h = h.detach()

        # h = self.trunk(h)
        
        return h


class Actor(nn.Module):
    """ Continous MLP Actor for Soft Actor-Critic """
    def __init__(self, encoder, proprioception_dim, action_dim, n_hid, device):
        super(Actor, self).__init__()        
        self.device = device
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        self.encoder = encoder

        # Two hidden layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.encoder.repr_dim + proprioception_dim, n_hid),
            nn.LeakyReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(n_hid, action_dim)
        self.log_std = nn.Linear(n_hid, action_dim)

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)


    def forward(self, image, proprioception, detach_encoder=True):
        image_feat = self.encoder(image, detach=detach_encoder)
        fc1 = self.fc1(torch.cat((image_feat, proprioception), axis=-1))
        phi = self.fc2(fc1)        
        phi_norm = torch.norm(phi, p=2, dim=1, keepdim=True)
        phi = phi/phi_norm

        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
   
        try:
            dist = MultivariateNormal(mu, torch.diag_embed(log_std.exp()))
        except Exception as e:
            print("Mean: {}, Sigma: {}".format(mu, torch.exp(log_std[0])))
            raise e
        
        
        action_pre = dist.rsample()
        lprob = dist.log_prob(action_pre)
        lprob -= (2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))).sum(axis=1)
        
        # N.B: Tanh must be applied _only_ after lprob estimation of dist sampled action!! 
        #   A mistake here can break learning :/ 
        action = torch.tanh(action_pre)
        action_info = {'mu': mu, 'log_std': log_std, 'dist': dist, 'lprob': lprob, 'action_pre': action_pre}

        return action, action_info


class Q(nn.Module):
    def __init__(self, obs_dim, action_dim, n_hid, device):
        super(Q, self).__init__()        
        self.device = device

        # Two hidden layers
        self.fc1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, n_hid),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),            
        )
        self.q = nn.Linear(n_hid, 1)
        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs, action):
        action = action.to(self.device)
        fc1 = self.fc1(torch.cat((obs, action), axis=-1))
        phi = self.fc2(fc1)
        phi_norm = torch.norm(phi, p=2, dim=1, keepdim=True)
        phi = phi/phi_norm
        return self.q(phi).view(-1)


class DoubleQ(nn.Module):
    def __init__(self, encoder, proprioception_dim, action_dim, n_hid, device):
        super(DoubleQ, self).__init__()
        self.encoder = encoder
        self.device = device

        # build value functions
        self.Q1 = Q(encoder.repr_dim + proprioception_dim, action_dim, n_hid, device)
        self.Q2 = Q(encoder.repr_dim + proprioception_dim, action_dim, n_hid, device)
        self.to(device)

    def forward(self, image, proprioception, action, detach_encoder=False):  
        img_phi = self.encoder(image, detach=detach_encoder)
        phi = torch.cat((img_phi, proprioception), axis=-1)      
        q1 = self.Q1(phi, action)
        q2 = self.Q2(phi, action)
        return q1, q2


class PixelQ(nn.Module):
    def __init__(self, encoder, action_dim, n_hid, device):
        super(PixelQ, self).__init__()
        self.encoder = encoder
        self.device = device

        # build value functions
        self.Q = Q(encoder.repr_dim, action_dim, n_hid, device)
        self.to(device)

    def forward(self, obs, action, detach_encoder=False):  
        img_phi = self.encoder(obs, detach=detach_encoder)      
        return self.Q(img_phi, action)
    