import numpy as np
from Filtering.ExampleModels import SimpleModel
from Filtering.KalmanFilters import UnscentedKF

from scipy.stats import multivariate_normal

class BootstrapPF():
    def __init__(self, n_particles:int, state_dim:int, obs_dim:int, model:SimpleModel) -> None:
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.model = model
        
    def propagation(self, current_states, current_time, aux_data):
        propagated_states = np.zeros_like(current_states)
        # TODO: multiprocessing
        for i_par in range(self.n_particles):
            propagated_states[i_par, ...] = self.model.forward_model(current_states[i_par, ...],
                                                                    current_time, aux_data)
        return propagated_states
    
    def update(self, propagated_states, propagated_weights, propagated_time, true_observation):
        obs_likelihood_vec = np.zeros_like(propagated_weights)
        # TODO: multiprocessing
        for i_par in range(self.n_particles):
            _, obs_likelihood_vec[i_par] = self.model.observation_likelihood(propagated_states[i_par, ...],
                                                                          propagated_time, true_observation)
        updated_weight_vec = propagated_weights*obs_likelihood_vec
        updated_weight_vec = updated_weight_vec/np.sum(updated_weight_vec)
        updated_states = propagated_states # No update since the states are sampled from the transition PDF
        return [updated_states, updated_weight_vec]

    def resampling(self, updated_states, updated_weight_vec):
        # assimilated_states : [n_particles, ...])
        cumulative_prob = np.cumsum(updated_weight_vec)
        resampled_freq_vec = np.zeros(self.n_particles)
        resampled_states = np.zeros_like(updated_states)
        u_seed = np.random.uniform()/self.n_particles
        k = 0
        for i_par in range(self.n_particles):
            temp_u = u_seed + (i_par/self.n_particles)
            while(temp_u > cumulative_prob[k]):
                k = k + 1
            resampled_states[i_par, ...] = updated_states[k, ...]
            resampled_freq_vec[k] = resampled_freq_vec[k] + 1        
        resampled_weight_vec = np.ones(self.n_particles)/self.n_particles
        return [resampled_states, resampled_weight_vec, resampled_freq_vec]

    def get_weighted_mean(self, state_ensemble, weight_ensemble):
        ## state_ensemble: n_steps, n_particles, dim
        ## weight_ensemble: n_steps, n_particles
        return np.sum(state_ensemble*weight_ensemble[:, :, np.newaxis], axis = 1)
    
class UnscentedPF():
    def __init__(self, n_particles:int, state_dim:int, obs_dim:int, model:SimpleModel) -> None:
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.model = model
        
        alpha = 0.001
        beta = 2
        kappa = 0
        
        self.ukf_obj = UnscentedKF(state_dim, obs_dim, alpha, beta, kappa, model)
        
    def importance_sampling(self, current_state_ensemble, current_time, next_time, true_observation):
        ## current_state_ensemble: n_particle x state_dim
        ## Also calculates transition probability
        imp_sampling_ensemble = np.zeros((self.n_particles, self.state_dim))
        imp_density_ensemble = np.zeros(self.n_particles)
        for i_par in range(self.n_particles):
            [imp_sampling_ensemble[i_par, :],
             imp_density_ensemble[i_par]] = self.importance_sampling_helper(current_state_ensemble[i_par, :],
                                                                            current_time, next_time, true_observation)
        return [imp_sampling_ensemble, imp_density_ensemble]
    
    def importance_sampling_helper(self, current_state, current_time, next_time, true_observation):
        ## current_state: state_dim, current_time: float, true_observation: obs_dim
        ## better way to define this cov?
        current_state_cov = np.diag(np.ones(self.state_dim)*0.5)
        
        [propagated_sigma_states,
         propagated_sigma_obs] = self.ukf_obj.propagation(current_state, current_state_cov, current_time, next_time)
        [_prop_state_mean, _prop_state_cov,
         _prop_obs_mean, _prop_obs_cov, 
         updated_state_mean, 
         updated_state_cov] = self.ukf_obj.update(propagated_sigma_states, propagated_sigma_obs, 
                                                  true_observation)
        ukf_posterior = multivariate_normal(updated_state_mean, updated_state_cov)
        imp_sample = ukf_posterior.rvs()
        imp_density = ukf_posterior.pdf(imp_sample)
        return imp_sample, imp_density
    
    def weight_update(self, current_state_ensemble, imp_sampling_ensemble, imp_density_ensemble, prior_weights,
                      current_time, next_time, true_observation):
        updated_weights = np.zeros(self.n_particles)
        obs_likelihoods = np.zeros(self.n_particles)
        transition_probs = np.zeros(self.n_particles)
        for i_par in range(self.n_particles):
            _, obs_likelihoods[i_par] = self.model.observation_likelihood(imp_sampling_ensemble[i_par, :], next_time,
                                                                       true_observation)
            _, transition_probs[i_par] = self.model.transition_probability(current_state_ensemble[i_par, :], current_time, 
                                                                           imp_sampling_ensemble[i_par, :])
            updated_weights[i_par] = prior_weights[i_par]*obs_likelihoods[i_par]*transition_probs[i_par]/imp_density_ensemble[i_par]
        # updated_weights = prior_weights*obs_likelihoods*transition_probs/imp_density_ensemble
        updated_weights = updated_weights/np.sum(updated_weights)
        return [updated_weights, obs_likelihoods, transition_probs]
    
    def resampling(self, updated_states, updated_weight_vec):
        # assimilated_states : [n_particles, ...])
        cumulative_prob = np.cumsum(updated_weight_vec)
        resampled_freq_vec = np.zeros(self.n_particles)
        resampled_states = np.zeros_like(updated_states)
        u_seed = np.random.uniform()/self.n_particles
        k = 0
        for i_par in range(self.n_particles):
            temp_u = u_seed + (i_par/self.n_particles)
            while(temp_u > cumulative_prob[k]):
                k = k + 1
            resampled_states[i_par, ...] = updated_states[k, ...]
            resampled_freq_vec[k] = resampled_freq_vec[k] + 1        
        resampled_weight_vec = np.ones(self.n_particles)/self.n_particles
        return [resampled_states, resampled_weight_vec, resampled_freq_vec]

    def get_weighted_mean(self, state_ensemble, weight_ensemble):
        ## state_ensemble: n_steps, n_particles, dim
        ## weight_ensemble: n_steps, n_particles
        return np.sum(state_ensemble*weight_ensemble[:, :, np.newaxis], axis = 1)

    def propagation(self, current_states, current_time, aux_data):
        propagated_states = np.zeros_like(current_states)
        # TODO: multiprocessing
        for i_par in range(self.n_particles):
            propagated_states[i_par, ...] = self.model.forward_model(current_states[i_par, ...],
                                                                    current_time, aux_data)
        return propagated_states
    