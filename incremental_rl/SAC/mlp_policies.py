import torch

import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.distributions import MultivariateNormal

# dict to enable loading activations based on a string
nn_activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
}


def mlp_hidden_layers(input_dim, hidden_sizes, activation="relu"):
    """ Helper function to create hidden MLP layers.
    N.B: The same activation is applied after every layer

    Args:
        input_dim: An int denoting the input size of the mlp
        hidden_sizes: A list with ints containing hidden sizes
        activation: A str specifying the activation function

    Returns:

    """
    activation = nn_activations[activation]
    dims = [input_dim] + hidden_sizes
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation)
    return layers


def orthogonal_weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, nn_params, device, use_pnorm=False):
        super(GaussianActor, self).__init__()
        self.device = device
        self.use_pnorm = use_pnorm
        
        # Initialize Actor Network
        nhid_actor = nn_params["mlp"]["hidden_sizes"][-1]
        ab_layers = mlp_hidden_layers(input_dim=obs_dim, hidden_sizes=nn_params["mlp"]["hidden_sizes"],
                                      activation=nn_params["mlp"]["activation"])
        self.actor_body = torch.nn.Sequential(*ab_layers)
        self.actor_mean = torch.nn.Sequential(torch.nn.Linear(nhid_actor, action_dim))
        self.actor_log_std = torch.nn.Sequential(torch.nn.Linear(nhid_actor, action_dim))

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def get_features(self, x):
        with torch.no_grad():
            phi = self.actor_body(x)
        return phi

    def forward(self, obs, rp):
        obs = obs.to(self.device)
        phi = self.actor_body(obs)
        if self.use_pnorm:
            phi_norm = torch.norm(phi, dim=1).view((-1, 1))
            phi = phi/phi_norm

        mu = self.actor_mean(phi)
        log_std = self.actor_log_std(phi)

        try:
            dist = MultivariateNormal(mu, torch.diag_embed(log_std.exp()))
        except Exception as e:
            print("Mean: {}, Sigma: {}".format(mu, torch.exp(log_std[0])))
            raise e

        action = dist.rsample() if rp else dist.sample()
        action_info = {'mu': mu, 'log_std': log_std, 'dist': dist, 'lprob': dist.log_prob(action)}
        
        return action, action_info

    def get_lprob(self, dist, action):
        return dist.log_prob(action)
    

class MeanZeroActor(GaussianActor):
    def __init__(self, obs_dim, action_dim, nn_params, device):
        super(MeanZeroActor, self).__init__(obs_dim, action_dim, nn_params, device)
        nhid_actor = nn_params["mlp"]["hidden_sizes"][-1]

        # Mean zero centering
        self.actor_mean[-1].weight.data[:] = 0
        self.actor_mean[-1].bias.data[:] = 0

        # Sigma σ=0.01 
        self.actor_log_std[-1].weight.data[:] = 0
        self.actor_log_std[-1].bias.data[:] = torch.log(torch.as_tensor(0.01, dtype=torch.float32))

        self.to(device=device)
        

class SquashedGaussianMLPActor(nn.Module):
    """ Continous MLP Actor for Soft Actor-Critic """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, obs_dim, action_dim, nn_params, device, use_pnorm=False):
        super(SquashedGaussianMLPActor, self).__init__()
        self.device = device
        self.use_pnorm = use_pnorm

        layers = mlp_hidden_layers(input_dim=obs_dim, hidden_sizes=nn_params["mlp"]["hidden_sizes"],
                                   activation=nn_params["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)

        self.mu = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], action_dim)
        self.log_std = nn.Linear(nn_params["mlp"]["hidden_sizes"][-1], action_dim)

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def get_features(self, x):
        with torch.no_grad():
            phi = self.phi(x)
            if self.use_pnorm:
                phi_norm = torch.norm(phi, dim=1).view((-1, 1))
                phi = phi/phi_norm
        return phi

    def forward(self, obs, rp):
        """ Sample an action

        Args:
            obs (tensor): Observation vector
            rp (bool): Draw an action using reparametrization trick or not

        Returns:
            tensor: Action vector
            dict: Contains relevant metadata about sampling the action
        """
        obs = obs.to(self.device)
        phi = self.phi(obs)
        if self.use_pnorm:
            phi_norm = torch.norm(phi, dim=1).view((-1, 1))
            phi = phi/phi_norm

        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
   
        try:
            dist = MultivariateNormal(mu, torch.diag_embed(log_std.exp()))
        except Exception as e:
            print("Mean: {}, Sigma: {}".format(mu, torch.exp(log_std[0])))
            raise e
        
        action_pre = dist.rsample() if rp else dist.sample()
        lprob = self.get_lprob(dist, action_pre)      
        
        # N.B: Tanh must be applied _only_ after lprob estimation of dist sampled action!! 
        #   A mistake here can break learning :/ 
        action = torch.tanh(action_pre)        
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        mod_action = torch.mean(torch.abs(action)).item()
        action_info = {
            'mu': mu, 'log_std': log_std, 'dist': dist, 'lprob': lprob, 
            'action_pre': action_pre, 'mod_action': mod_action, 'entropy': entropy.mean().item()
        }
        
        
        return action, action_info

    def get_lprob(self, dist, action):
        lprob = dist.log_prob(action)
        lprob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        return lprob


class Critic(nn.Module):
    def __init__(self, obs_dim, nn_params, device, use_pnorm=False):
        super(Critic, self).__init__()
        self.device = device
        self.use_pnorm = use_pnorm

        # Initialize Critic Network
        nhid_critic = nn_params["mlp"]["hidden_sizes"][-1]
        c_layers = mlp_hidden_layers(input_dim=obs_dim, hidden_sizes=nn_params["mlp"]["hidden_sizes"],
                                     activation=nn_params["mlp"]["activation"])
        self.phi = torch.nn.Sequential(*c_layers)
        self.value = torch.nn.Linear(nhid_critic, 1)
        
        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, x):
        x = x.to(self.device)
        phi = self.phi(x)
        if self.use_pnorm:
            phi_norm = torch.norm(phi, dim=1).view((-1, 1))
            phi = phi/phi_norm
        return self.value(phi).view(-1)


class DoubleQ(nn.Module):
    def __init__(self, obs_dim, action_dim, nn_params, device):
        super(DoubleQ, self).__init__()
        self.device = device

        # build value functions
        self.Q1 = Critic(obs_dim+action_dim, nn_params, device)
        self.Q2 = Critic(obs_dim+action_dim, nn_params, device)
        self.to(device)

    def forward(self, obs, action):
        x = torch.cat((obs, action), -1).to(self.device)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    cfg = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg.obs_dim = 4
    cfg.action_dim = 2
    cfg.actor_nn_params = {
        'mlp': {
            'hidden_sizes': [256, 256],
            'activation': "relu",
        }
    }
    cfg.device = torch.device('cpu')
    actor = SquashedGaussianMLPActor(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, 
                                     nn_params=cfg.actor_nn_params, device=cfg.device)         

    obs = np.random.uniform(-0.1, 0.1, size=(2, cfg.obs_dim)).astype(np.float32)
    obs = torch.tensor(obs)
    action, info = actor(obs, rp=True)
    print(action, info['lprob'])

    critic = DoubleQ(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, 
                     nn_params=cfg.actor_nn_params, device=cfg.device)
    q1, q2 = critic(obs, action)
    print(q1, q2)         

