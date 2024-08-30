import numpy as np
from Filtering.ExampleModels import SimpleModel, ModelClassTemplate
from Filtering.KalmanFilters import UnscentedKF

from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count

from scipy.stats import multivariate_normal
    
class BootstrapPF():
    # Safe to use multiprocessing
    def __init__(self, n_particles:int, state_dim:int, obs_dim:int, model:ModelClassTemplate) -> None:
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.model = model
        
    def propagation(self, current_states, current_time, aux_data, multiprocessing):
        """Propagation using transition density

        Args:
            current_states (ndarray): n_particles x state_dim
            current_time (any): required for forward model
            aux_data (any): additional arguments for forward model, must have with_noise argument
            multiprocessing (bool, optional): Flag for multiprocessing. Defaults to True.

        Returns:
            ndarray: returns states sampled from p(x_k | x_{k - 1}).
        """
        if multiprocessing:
            # To prevent repeated sampling of same random numbers
            noises = self.model.forward_noise_distribution.rvs((self.n_particles, self.state_dim))
            def propagation_helper(i_par):
                return self.model.forward_model(current_states[i_par, ...], current_time, aux_data, noises[i_par])
            with Pool(nodes=cpu_count()) as pool:
                propagated_states = pool.map(propagation_helper, range(self.n_particles))
            propagated_states = np.array(propagated_states)
        else:
            propagated_states = np.zeros_like(current_states)
            for i_par in range(self.n_particles):
                propagated_states[i_par, ...] = self.model.forward_model(current_states[i_par, ...],
                                                                        current_time, aux_data)
        return propagated_states
    
    def update(self, propagated_states, propagated_weights, propagated_time, true_observation, multiprocessing):
        """Updates and normalizes the weights of the states bases on the observation likelihood

        Args:
            propagated_states (ndarray): n_particles x state_dim
            propagated_weights (ndarray): (n_particles,)
            propagated_time (any): required for observation operator
            true_observation (ndarray): (obs_dim, )
            multiprocessing (bool, optional): Flag for multiprocessing. Defaults to True.

        Returns:
            List[ndarray]: return the updated states (same as propagated states) and updated weight vec
        """
        if multiprocessing:
            # No Noise Generation involved
            def update_helper(i_par):
                return self.model.observation_likelihood(propagated_states[i_par, ...], propagated_time, true_observation)[1]
            with Pool(cpu_count()) as pool:
                obs_likelihood_vec = pool.map(update_helper, range(self.n_particles))
            obs_likelihood_vec = np.array(obs_likelihood_vec).squeeze()
        else:
            obs_likelihood_vec = np.zeros_like(propagated_weights)
            for i_par in range(self.n_particles):
                _, obs_likelihood_vec[i_par] = self.model.observation_likelihood(propagated_states[i_par, ...],
                                                                            propagated_time, true_observation)
        updated_weight_vec = propagated_weights*obs_likelihood_vec
        updated_weight_vec = updated_weight_vec/np.sum(updated_weight_vec)
        updated_states = propagated_states # No update since the states are sampled from the transition PDF
        return [updated_states, updated_weight_vec]

    
class UnscentedPF():
    # Safe to use multiprocessing
    def __init__(self, n_particles:int, state_dim:int, obs_dim:int, model:ModelClassTemplate,
                 alpha:float, beta:float, kappa:float) -> None:
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.model = model
        
        alpha = alpha
        beta = beta
        kappa = kappa
        self.ukf_obj = UnscentedKF(state_dim, obs_dim, alpha, beta, kappa, model)
        
    def importance_sampling(self, current_state_ens, current_cov_ens, current_time, propagated_time, aux_data, true_observation, multiprocessing):
        """Samples state from the posterior density obtained using Unscented Particle Filter.

        Args:
            current_state_ens (ndarray): n_particle x state_dim
            current_cov_ens (ndarray): n_particle x state_dim x state_dim
            current_time (float): Required for the forward model
            propagated_time (float): Required for the observation operator
            true_observation (ndarray): (obs_dim,) observation at propagated time
            multiprocessing (bool, optional): Flag for multiprocessing.

        Returns:
            List[ndarray]: Importance sampling ensemble (n_particles x state_dim), imp_density_ensemble (n_particles, )
        """
        
        if multiprocessing:
            def importance_sampling_helper_wrapper(i_par):
                return self.importance_sampling_helper(current_state_ens[i_par, :], current_cov_ens[i_par, :, :], current_time, propagated_time, 
                                                       aux_data, true_observation)
            with Pool(nodes=cpu_count()) as pool:
                results = pool.map(importance_sampling_helper_wrapper, range(self.n_particles))
            imp_sample_ens, imp_sample_cov_ens, imp_sample_density_ens = zip(*results)
            imp_sample_ens = np.array(imp_sample_ens)
            imp_sample_cov_ens = np.array(imp_sample_cov_ens)
            imp_sample_density_ens = np.array(imp_sample_density_ens)

        else:
            imp_sample_ens = np.zeros((self.n_particles, self.state_dim))
            imp_sample_cov_ens = np.zeros((self.n_particles, self.state_dim, self.state_dim))
            imp_sample_density_ens = np.zeros(self.n_particles)
            for i_par in range(self.n_particles):
                [imp_sample_ens[i_par, :],
                 imp_sample_cov_ens[i_par, :, :],
                 imp_sample_density_ens[i_par]] = self.importance_sampling_helper(current_state_ens[i_par, :], current_cov_ens[i_par, :, :], current_time, propagated_time,
                                                                                aux_data, true_observation)
        return [imp_sample_ens, imp_sample_cov_ens, imp_sample_density_ens]
    
    def importance_sampling_helper(self, current_state, current_cov, current_time, propagated_time, aux_data, true_observation):
        """Helper for Importance sampling: Samples a state from posterior and corresponding pdf for each particle using UKF

        Args:
            current_state (ndarray): (state_dim, )
            current_time (any): Required by the forward model
            next_time (any): Required by the observation operator
            next_true_observation (ndarray): (obs_dim, )
            current_state_cov (ndarray): state_dim x state_dim
            
        Note:
            Using updated_state_mean as the importance sample leads to a poor filter. Must sample from the posterior PDF.

        Returns:
            List[ndarray]: imp_sample (state_dim, ) and imp_density (float)
        """        
        [prop_sigma_states,
         prop_sigma_obs] = self.ukf_obj.propagation(current_state, current_cov, current_time, propagated_time, aux_data, multiprocessing=False)
        [_, _, updated_state_mean, updated_state_cov] = self.ukf_obj.update(prop_sigma_states, prop_sigma_obs, true_observation)
        ukf_posterior = multivariate_normal(updated_state_mean, updated_state_cov)
        imp_sample = ukf_posterior.rvs()
        imp_density = ukf_posterior.pdf(imp_sample)
        return [imp_sample, updated_state_cov, imp_density]
    
    def weight_update(self, current_state_ens, imp_sample_ens, imp_sample_density_ens, prior_weights, current_time, propagated_time, true_observation, multiprocessing):
        """Weight update for the particle filter based on the importance sample

        Args:
            current_state_ens (ndarray): n_particles x state_dim
            imp_sample_ens (ndarray): n_particles x state_dim
            imp_sample_density_ens (ndarray): (n_particles, )
            prior_weights (ndarray): (n_particles, )
            current_time (any): Required by the forward model
            propagated_time (any): Required by the observation operator
            true_observation (ndarray): (obs_dim, ) - Observation at propagated time
            multiprocessing (Bool): Flag for multiprocessing

        Returns:
            List[ndarray]: updated_weights (n_particles, ), obs_likelihoods (n_particles, ), and transition_probs (n_particles, )
        """
        if multiprocessing:
            def weight_update_helper(i_par):
                _, obs_likelihood = self.model.observation_likelihood(imp_sample_ens[i_par, :], propagated_time, true_observation)
                _, transition_prob = self.model.transition_probability(current_state_ens[i_par, :], current_time, imp_sample_ens[i_par, :])
                return obs_likelihood, transition_prob
            with Pool(nodes=cpu_count()) as pool:
                results = pool.map(weight_update_helper, range(self.n_particles))
            obs_likelihoods, transition_probs = zip(*results)
            obs_likelihoods = np.array(obs_likelihoods)
            transition_probs = np.array(transition_probs)
        else:
            updated_weights = np.zeros(self.n_particles)
            obs_likelihoods = np.zeros(self.n_particles)
            transition_probs = np.zeros(self.n_particles)
            for i_par in range(self.n_particles):
                _, obs_likelihoods[i_par] = self.model.observation_likelihood(imp_sample_ens[i_par, :], propagated_time, true_observation)
                _, transition_probs[i_par] = self.model.transition_probability(current_state_ens[i_par, :], current_time, imp_sample_ens[i_par, :])
        
        updated_weights = prior_weights*obs_likelihoods*transition_probs/imp_sample_density_ens
        updated_weights = updated_weights/np.sum(updated_weights)
        return [updated_weights, obs_likelihoods, transition_probs]


class PFUtils():
    def __init__(self, pf_obj) -> None:
        self.pf_obj = pf_obj
        
    def uniform_initiate_particles(self, par_ranges):
        particles = np.zeros((self.pf_obj.n_particles, self.pf_obj.state_dim))
        for i in range(self.pf_obj.state_dim):
            min_val, max_val = par_ranges[i]
            particles[:, i] = np.random.uniform(min_val, max_val,
                                                self.pf_obj.n_particles)
        return particles
    
    def get_weighted_mean(self, state_ensemble, weight_ensemble):
        ## state_ensemble: n_steps, n_particles, dim
        ## weight_ensemble: n_steps, n_particles
        return np.sum(state_ensemble*weight_ensemble[:, :, np.newaxis], axis = 1)
    
    def get_weighted_var(self, state_ensemble, weight_ensemble):
        weighted_mean = self.get_weighted_mean(state_ensemble, weight_ensemble)
        deviation = state_ensemble - weighted_mean[:, np.newaxis, :]
        weighted_var = np.sum(weight_ensemble[:, :, np.newaxis]*deviation**2, axis=1)
        weight_sum = np.sum(weight_ensemble, axis=1)[:, np.newaxis]
        weighted_var = weighted_var/weight_sum
        return weighted_var
    
    def resampling(self, updated_states, updated_weight_vec, upf = False, upf_cov_ens = None):
        # assimilated_states : [n_particles, ...])
        weight_sum = np.sum(updated_weight_vec)
        if np.isnan(weight_sum):
            print("NaNs encountered: Weighing all particles equally")
            resampled_states = updated_states
            resampled_freq_vec = np.ones(self.pf_obj.n_particles)
            if upf:
                resampled_cov_ens = upf_cov_ens
        elif weight_sum == 0:
            print("All weights are zero: Weighing all particles equally")
            resampled_states = updated_states
            resampled_freq_vec = np.ones(self.pf_obj.n_particles)
            if upf:
                resampled_cov_ens = upf_cov_ens
        else:
            if upf:
                resampled_cov_ens = np.zeros_like(upf_cov_ens)
            updated_weight_vec = updated_weight_vec/weight_sum
            cumulative_prob = np.cumsum(updated_weight_vec)
            resampled_freq_vec = np.zeros(self.pf_obj.n_particles)
            resampled_states = np.zeros_like(updated_states)
            u_seed = np.random.uniform()/self.pf_obj.n_particles
            k = 0
            for i_par in range(self.pf_obj.n_particles):
                temp_u = u_seed + (i_par/self.pf_obj.n_particles)
                while(temp_u > cumulative_prob[k]):
                    k = k + 1
                resampled_states[i_par, ...] = updated_states[k, ...]
                resampled_freq_vec[k] = resampled_freq_vec[k] + 1
                if upf:
                    resampled_cov_ens[i_par, ...] = upf_cov_ens[k, ...]
        
        resampled_weight_vec = np.ones(self.pf_obj.n_particles)/self.pf_obj.n_particles
        if upf:
            return [resampled_states, resampled_weight_vec, resampled_freq_vec, resampled_cov_ens]
        else:
            return [resampled_states, resampled_weight_vec, resampled_freq_vec]
    
 
    