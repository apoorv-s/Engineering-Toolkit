import numpy as np
from Filtering.ExampleModels import SimpleModel, ModelClassTemplate

from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count

class KalmanFilter:
    def __init__(self, forward_model_mat, obs_operator_mat, obs_cov):
        self.forward_model_mat = forward_model_mat
        self.obs_operator_mat = obs_operator_mat
        self.obs_cov = obs_cov
    
    def analysis(self, current_state, obs_vec, current_state_cov):
        # input: current_state (current state(from forecast)), obs_vec (vector of observations), 
        #        current_state_cov (covarianceMatrix of the current state)
        # output: updated_state (state after assimilating the data from the observations)
        #         updated_state_cov (covariance matrix of the updated state)
        
        h = obs_vec - np.dot(self.obs_operator_mat, current_state)
        p_inv = np.linalg.inv(self.obs_cov + np.dot(self.obs_operator_mat, np.dot(current_state_cov, self.obs_operator_mat.T)))
        r = np.dot(self.obs_operator_mat, current_state_cov.T)
        updated_state = current_state + np.dot(np.transpose(r), np.dot(p_inv, h))
        updated_state_cov = current_state_cov - np.dot(np.transpose(r), np.dot(p_inv, r))
        return updated_state, updated_state_cov
    
    def propagation(self, updated_state, updated_state_cov, forward_model_noise_cov):
        # input: updated_state (state after assimilating the data from the observations), 
        #        updated_state_cov (covarianceMatrix of the assimilated state), 
        #        forward_model_noise_cov (covariance matrix of noise in forward model)
        # output: next_state (state after propagating the assimilated state through the model)
        #         next_state_cov (covariance matrix of the next state)
        
        next_state = np.dot(self.forward_model_mat, updated_state)
        next_state_cov = np.dot(self.forward_model_mat, np.dot(updated_state_cov, self.forward_model_mat.T)) + forward_model_noise_cov
        return next_state, next_state_cov
    
         
class UnscentedKF():
    # Safe to use multiprocessing
    def __init__(self, state_dim:int, obs_dim:int, alpha:float,
                 beta:float, kappa:float, model:ModelClassTemplate) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.model = model
        
        self.total_dim = 2*state_dim + obs_dim
        self.n_sigma_points = 2*self.total_dim + 1
        self.lambda_ukf = (alpha**2)*(self.total_dim + kappa) - self.total_dim
        
        self.sigma_weights_mean, self.sigma_weights_cov = self.get_sigma_weights()
        
        print("Make sure that the mean is included with forward model and observation operator when used with with_noise=False")
    
    def get_weighted_mean(self, state_ensemble):
        """Mean with particles weighted by sigma_weights_mean

        Args:
            state_ensemble (ndarray): n_sigma_points x dim

        Returns:
            ndarray: dim
        """
        return np.array(np.average(state_ensemble, axis=0, weights=self.sigma_weights_mean))
    
    def get_weighted_cov(self, state_ensemble):
        """Cov with particles weighted by sigma_weights_cov (using mean with particles weighted by sigma_weights_mean)

        Args:
            state_ensemble (ndarray): n_sigma_points x dim

        Returns:
            ndarray: dim x dim
        """
        weighted_mean = self.get_weighted_mean(state_ensemble)
        weighted_cov = np.zeros((state_ensemble.shape[1], state_ensemble.shape[1]))
        for i_sigma in range(self.n_sigma_points):
            bias = (state_ensemble[i_sigma, :] - weighted_mean).reshape(-1, 1)
            weighted_cov = weighted_cov + self.sigma_weights_cov[i_sigma]*np.dot(bias, bias.T)
        return weighted_cov
    
    def get_weighted_cross_cov(self, state_ensemble_1, state_ensemble_2):
        """Cross Cov with particles weighted by sigma_weights_cov (using mean with particles weighted by sigma_weights_mean)

        Args:
            state_ensemble_1 (ndarray): n_sigma_points x dim1
            state_ensemble_2 (ndarray): n_sigma_points x dim2

        Returns:
            ndarray: dim1 x dim2
        """
        weighted_mean_1 = self.get_weighted_mean(state_ensemble_1)
        weighted_mean_2 = self.get_weighted_mean(state_ensemble_2)
        weighted_cross_cov = np.zeros((state_ensemble_1.shape[1],
                                       state_ensemble_2.shape[1]))
        for i_sigma in range(self.n_sigma_points):
            bias_1 = (state_ensemble_1[i_sigma, :] - weighted_mean_1).reshape(-1, 1)
            bias_2 = (state_ensemble_2[i_sigma, :] - weighted_mean_2).reshape(-1, 1)
            weighted_cross_cov = weighted_cross_cov + self.sigma_weights_cov[i_sigma]*np.dot(bias_1, bias_2.T)
        return weighted_cross_cov
    
    def get_sigma_weights(self):
        """Sigma weights based on UPF parametes

        Returns:
            ndarray: sigma_weights_mean (n_sigma_points, ) and sigma_weights_cov (n_sigma_points, )
        """
        sigma_weights_mean = np.zeros(self.n_sigma_points)
        sigma_weights_cov = np.zeros(self.n_sigma_points)
        sigma_weights_mean[0] = self.lambda_ukf/(self.total_dim + self.lambda_ukf)
        sigma_weights_cov[0] = sigma_weights_mean[0] + (1 - (self.alpha**2) + self.beta)
        for i in range(self.total_dim):
            sigma_weights_mean[i + 1] = 1/(2*(self.total_dim + self.lambda_ukf))
            sigma_weights_cov[i + 1] = 1/(2*(self.total_dim + self.lambda_ukf))
            sigma_weights_mean[i + self.total_dim + 1] = 1/(2*(self.total_dim + self.lambda_ukf))
            sigma_weights_cov[i + self.total_dim + 1] = 1/(2*(self.total_dim + self.lambda_ukf))
        return sigma_weights_mean, sigma_weights_cov
    
    def get_sigma_points(self, state_mean, state_cov):
        """Calcultate sigma points based on the given mean and covariance

        Args:
            state_mean (ndarray): (total_dim, )
            state_cov (ndarray): (total_dim, total_dim)

        Returns:
            ndarray: (n_sigma_points, total_dim)
        """
        sigma_points = np.zeros((self.n_sigma_points, self.total_dim))
        
        ## Can use the block diagonal structure of the matrix to compute this more efficiently?
        ## Numpy cholesky of the for A = L*L.T, so columns of L to be used, (Unscented Filtering and Nonlinear Estimation, Julier and Uhlmann, 2004)
        sqrt_state_cov = np.linalg.cholesky(state_cov*(self.total_dim + self.lambda_ukf))
        sigma_points[0, :] = state_mean
        for i_dim in range(self.total_dim):
            sigma_points[i_dim + 1, :] = state_mean + sqrt_state_cov[i_dim, :]
            sigma_points[i_dim + self.total_dim + 1, :] = state_mean - sqrt_state_cov[i_dim, :]
        return sigma_points
    
    def propagation(self, current_state_mean, current_state_cov,
                    current_time, propagated_time, aux_data, multiprocessing):
        """Propagates the normal approximation of current state distribution through time and returns the normal approximation of propagated distribution along with the observations corresponding to the sigma states.

        Args:
            current_state_mean (ndarray): (state_dim, )
            current_state_cov (ndarray): (state_dim, state_dim)
            current_time (float): Required by the forward model
            propagated_time (float): Required by the observation operator
            multiprocessing (Bool): Flag for multiprocessing.

        Returns:
            List[ndarray]: propagated_sigma_states (n_sigma_points, state_dim), propagated_sigma_obs (n_sigma_points, obs_dim)
        """
        extended_state_mean = np.zeros(self.total_dim)
        
        ## Mean of state $x_k$
        extended_state_mean[0:self.state_dim] = current_state_mean
        
        ## Mean of forward model noise at the current state $x_{k+1} = f(x_k) + v_k$
        ## Not updating this because mean is already included in simple_model.forward() without noise
        # extended_state_mean[self.state_dim:2*self.state_dim] = self.model.forward_noise_mean
        
        ## Mean of observation noise at the next step $y_{k} = g(x_k) + w_k$
        # extended_state_mean[2*self.state_dim:2*self.state_dim + self.obs_dim] = self.model.obs_noise_mean
        
        extended_state_cov = np.zeros((self.total_dim, self.total_dim))
        extended_state_cov[0:self.state_dim, 0:self.state_dim] = current_state_cov
        
        ## Covariance of forward model noise distribution at the current step
        extended_state_cov[self.state_dim:2*self.state_dim,
                     self.state_dim:2*self.state_dim] = self.model.fwd_noise_cov
  
        ## Covariance of observation noise distribution at the next time step
        extended_state_cov[2*self.state_dim:2*self.state_dim + self.obs_dim,
                     2*self.state_dim:2*self.state_dim + self.obs_dim] = self.model.obs_noise_cov
        
        
        sigma_points = self.get_sigma_points(extended_state_mean, extended_state_cov)
          
        if multiprocessing:
            def forward_model_wrapper_helper(i_sigma):
                return self.forward_model_wrapper(sigma_points[i_sigma, :], current_time, propagated_time, aux_data)
            with Pool(nodes=cpu_count()) as pool:
                res = pool.map(forward_model_wrapper_helper, range(self.n_sigma_points))
            propagated_sigma_states, propagated_sigma_obs = zip(*res)
            propagated_sigma_states = np.array(propagated_sigma_states)
            propagated_sigma_obs = np.array(propagated_sigma_obs)
        else:
            propagated_sigma_states = np.zeros((self.n_sigma_points, self.state_dim))
            propagated_sigma_obs = np.zeros((self.n_sigma_points, self.obs_dim))
            for i_sigma in range(self.n_sigma_points):
                [propagated_sigma_states[i_sigma, :], propagated_sigma_obs[i_sigma, :]] = self.forward_model_wrapper(sigma_points[i_sigma, :], current_time, propagated_time, aux_data)
                
        return [propagated_sigma_states, propagated_sigma_obs]
        
    def forward_model_wrapper(self, extended_state, current_time, propagated_time, aux_data):
        """ Wrapper to handle forward model with the extended state. Also takes care of the noise in propagated state and the corresponding observation using noise obtained from extneded state

        Args:
            extended_state (ndarray): (total_dim, )
            current_time (any): Required by the forward model
            next_time (any): Required by the observation operator
            
        Returns:
            List[ndarray]: propagated_state (state_dim, ), propagated_obs (obs_dim, )
        """
        current_state = extended_state[0:self.state_dim]
        current_state_noise = extended_state[self.state_dim:2*self.state_dim]
        current_obs_noise = extended_state[2*self.state_dim:2*self.state_dim + self.obs_dim]
        
        propagated_state = self.model.forward_model(current_state, current_time, aux_data) + current_state_noise
        propagated_obs = self.model.observation_operator(propagated_state, propagated_time, aux_data) + current_obs_noise
        
        return [propagated_state, propagated_obs]
        
    def update(self, propagated_sigma_states, propagated_sigma_obs, true_observation):
        """Kalman update based on weighted mean and cov estimated from unscented transform

        Args:
            propagated_sigma_states (ndarray): (n_sigma_points, state_dim)
            propagated_sigma_obs (ndarray): (n_sigma_points, obs_dim)
            true_observation (ndarray): (obs_dim, )

        Returns:
            List[ndarray]: propagated_state_mean, propagated_state_cov, propagated_obs_mean, propagated_obs_cov, updated_state_mean, updated_state_cov
        """
        propagated_state_mean = self.get_weighted_mean(propagated_sigma_states)
        propagated_state_cov = self.get_weighted_cov(propagated_sigma_states)
        
        propagated_obs_mean = self.get_weighted_mean(propagated_sigma_obs)
        propagated_obs_cov = self.get_weighted_cov(propagated_sigma_obs)
        
        propagated_cross_cov = self.get_weighted_cross_cov(propagated_sigma_states,
                                                           propagated_sigma_obs)
        
        kalman_gain = np.dot(propagated_cross_cov, np.linalg.inv(propagated_obs_cov))
        innovation = true_observation - propagated_obs_mean
        
        updated_state_mean = propagated_state_mean + np.dot(kalman_gain, innovation)
        updated_state_cov = propagated_state_cov - np.dot(kalman_gain, 
                                                    np.dot(propagated_obs_cov,
                                                           kalman_gain.T))
        
        return [propagated_state_mean, propagated_state_cov,
                updated_state_mean, updated_state_cov]


class EnsembleKF():
    # Safe to use multiprocessing
    def __init__(self, n_particles, state_dim, obs_dim, model:ModelClassTemplate) -> None:
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.model = model
        
    def propagation(self, current_ensemble, current_time, propagated_time, aux_data, multiprocessing):
        """Propagation using transition density

        Args:
            current_ensemble (ndarray): n_particles x state_dim
            current_time (float): required for forward model
            propagated_time (float): required for observation operator
            aux_data (any): additional arguments for forward model, must have with_noise argument
            multiprocessing (bool, optional): Flag for multiprocessing. Defaults to True.

        Returns:
            list: [propagated_ensemble, propagated_obs], where each is an ndarray.
        """
        if multiprocessing:
            # To prevent repeated sampling of same random numbers
            fwd_model_noise = self.model.forward_noise_distribution.rvs((self.n_particles, self.state_dim))
            obs_operator_noise = self.model.obs_noise_distribution.rvs((self.n_particles, self.obs_dim))
            def propagation_helper(i_par):
                temp_prop_state = self.model.forward_model(current_ensemble[i_par, ...], current_time, aux_data, fwd_model_noise[i_par])
                temp_prop_obs = self.model.observation_operator(temp_prop_state, propagated_time, aux_data, obs_operator_noise[i_par])
                return temp_prop_state, temp_prop_obs
            
            with Pool(nodes=cpu_count()) as pool:
                result = pool.map(propagation_helper, range(self.n_particles))
            propagated_ensemble, propagated_obs = zip(*result)
            propagated_ensemble = np.array(propagated_ensemble)
            propagated_obs = np.array(propagated_obs)
        else:
            propagated_ensemble = np.zeros((self.n_particles, self.state_dim))
            propagated_obs = np.zeros((self.n_particles, self.obs_dim))
            for i_par in range(self.n_particles):
                propagated_ensemble[i_par, :] = self.model.forward_model(current_ensemble[i_par, :], current_time, aux_data) # Noisy model
                propagated_obs[i_par, :] = self.model.observation_operator(propagated_ensemble[i_par, :], propagated_time, aux_data) # Perturbed observation
        return [propagated_ensemble, propagated_obs]
    
    def update(self, propagated_states, propagated_obs, true_observation, get_cov=False):
        """Kalman Update with propagated particles

        Args:
            propagated_states (ndarray): n_particles x dim - propagated state obtained using transition density (include noise)
            propagated_obs (ndarray): n_particles x dim - noisy observations corresponding to propagated states (include noise)
            true_observation (ndarray): true observation
            get_cov (bool, optional): option to calculate the updated state cov. Defaults to False.

        Returns:
            ndarray: updated state ensemble
        """
        propagated_obs_cov = np.cov(propagated_obs.T) #+ self.model.obs_noise_distribution.cov() - No need to add this since the observations are already perturbed
        propagated_state_obs_cov = self.get_cross_cov(propagated_states, propagated_obs)
        kalman_gain = np.dot(propagated_state_obs_cov, np.linalg.inv(propagated_obs_cov))
        innovation = (true_observation - propagated_obs).T
        
        updated_states = propagated_states.T + np.dot(kalman_gain, innovation)
        
        return updated_states.T
        
    def get_cross_cov(self, state_ensemble_1, state_ensemble_2):
        """cross cov between two ensembles

        Args:
            state_ensemble_1 (ndarray): n_particles x dim_1
            state_ensemble_2 (ndarray): n_particles x dim_2

        Returns:
            ndarray: dim_1 x dim_2
        """
        mean_1 = np.mean(state_ensemble_1, axis=0)
        mean_2 = np.mean(state_ensemble_2, axis=0)
        
        bias_1 = state_ensemble_1 - mean_1
        bias_2 = state_ensemble_2 - mean_2
        
        cross_cov = (bias_1.T @ bias_2)/self.n_particles
        return cross_cov 


