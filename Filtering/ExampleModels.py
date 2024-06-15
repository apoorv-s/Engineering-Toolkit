import numpy as np
from scipy.stats import gamma, norm


class SimpleModel():
    ## Based on the example from the UPF, RVD Merwe, et.al.
    def __init__(self, omega, phi, gamma_shape, gamma_scale, obs_noise_std) -> None:
        self.omega = omega
        self.phi = phi
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.obs_noise_std = obs_noise_std
        self.forward_noise_distribution = gamma(a=self.gamma_shape, scale=self.gamma_scale)
        self.obs_noise_distribution = norm(loc=0, scale=obs_noise_std)
        
        self.forward_noise_mean = self.forward_noise_distribution.mean()
        self.obs_noise_mean = self.obs_noise_distribution.mean()
        
    def forward_model(self, current_state, current_time, with_noise = True):
        if with_noise:
            return 1 + np.sin(self.omega*np.pi*current_time) + self.phi*current_state + self.forward_noise_distribution.rvs()
        else:
            ## makes the noise zero mean when dealing with UKF
            return 1 + np.sin(self.omega*np.pi*current_time) + self.phi*current_state + self.forward_noise_mean
    
    def transition_probability(self, current_state, current_time, next_state):
        # Mean included when with_noise = False
        noise = next_state - self.forward_model(current_state, current_time, with_noise=False) + self.forward_noise_mean
        return noise, self.forward_noise_distribution.pdf(noise)
    
    def observation_operator(self, current_state, current_time, with_noise):
        if(current_time <= 30):
            obs = self.phi*(current_state**2)
        else:
            obs = self.phi*(current_state**2) - 2
            
        if with_noise:
            obs = obs + self.obs_noise_distribution.rvs()
        else:
            ## makes the noise zero mean when dealing with UKF
            obs = obs + self.obs_noise_mean
        return obs
            
    def observation_likelihood(self, current_state, current_time, true_observation):
        predicted_observation = self.observation_operator(current_state, current_time,
                                                          with_noise=False)
        # Mean included when with_noise = False
        obs_diff = true_observation - predicted_observation + self.obs_noise_mean
        return obs_diff, self.obs_noise_distribution.pdf(obs_diff)
        
        
    
    