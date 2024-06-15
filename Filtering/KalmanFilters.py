import numpy as np
from Filtering.ExampleModels import SimpleModel

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
    ## Requires dynamics to be encapsulated in a class with following methods:
    ## forward_model, observation_operator 
    ## obs_noise_distribution.var() (variance of obs noise)
    ## forward_noise_distribution.var() (variance of forward model noise)
    def __init__(self, state_dim:int, obs_dim:int, alpha:float,
                 beta:float, kappa:float, model:SimpleModel) -> None:
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
    
    def get_weighted_mean(self, state_ensemble):
        ## state ensemble: n_sigma_points x dim
        return np.array(np.average(state_ensemble, axis=0, weights=self.sigma_weights_mean))
    
    def get_weighted_cov(self, state_ensemble):
        ## state ensemble: n_sigma_points x dim
        weighted_mean = self.get_weighted_mean(state_ensemble)
        weighted_cov = np.zeros((state_ensemble.shape[1], state_ensemble.shape[1]))
        for i_sigma in range(self.n_sigma_points):
            bias = (state_ensemble[i_sigma, :] - weighted_mean).reshape(-1, 1)
            weighted_cov = weighted_cov + self.sigma_weights_cov[i_sigma]*np.dot(bias, bias.T)
        return weighted_cov
    
    def get_weighted_cross_cov(self, state_ensemble_1, state_ensemble_2):
        ## state ensemble: n_sigma_points x dim
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
        ## state_mean: self.total_dim, state_cov: self.total_dim x self.total_dim
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
                    current_time, next_time):
        extended_state_mean = np.zeros(self.total_dim)
        
        ## Mean of state $x_k$
        extended_state_mean[0:self.state_dim] = current_state_mean
        
        ## Mean of forward model noise at the current state $x_{k+1} = f(x_k) + v_k$
        ## Not updating this because mean is already included in simple_model.forward() without noise
        # extended_state_mean[self.state_dim:2*self.state_dim] = self.model.forward_noise_distribution.mean()
        
        ## Mean of observation noise at the next step $y_{k} = g(x_k) + w_k$
        # extended_state_mean[2*self.state_dim:2*self.state_dim + self.obs_dim] = self.model.obs_noise_distribution.mean()
        
        extended_state_cov = np.zeros((self.total_dim, self.total_dim))
        extended_state_cov[0:self.state_dim, 0:self.state_dim] = current_state_cov
        
        ## Covariance of forward model noise distribution at the current step
        extended_state_cov[self.state_dim:2*self.state_dim,
                     self.state_dim:2*self.state_dim] = self.model.forward_noise_distribution.var()
        
        ## Covariance of observation noise distribution at the next time step
        extended_state_cov[2*self.state_dim:2*self.state_dim + self.obs_dim,
                     2*self.state_dim:2*self.state_dim + self.obs_dim] = self.model.obs_noise_distribution.var()
        
        sigma_points = self.get_sigma_points(extended_state_mean, extended_state_cov)
        
        propagated_sigma_states = np.zeros((self.n_sigma_points, self.state_dim))
        propagated_sigma_obs = np.zeros((self.n_sigma_points, self.obs_dim))
        for i_sigma in range(self.n_sigma_points):
            [propagated_sigma_states[i_sigma, :], propagated_sigma_obs[i_sigma, :]] = self.forward_model_wrapper(sigma_points[i_sigma, :], current_time, next_time)
            
        return [propagated_sigma_states, propagated_sigma_obs]
        
    def forward_model_wrapper(self, extended_state, current_time, next_time):
        ## Wrapper to handle forward model with the extended state
        ## Also takes care of the noise in propagated state and the corresponding observation using noise entried from extneded state
        current_state = extended_state[0:self.state_dim]
        current_state_noise = extended_state[self.state_dim:2*self.state_dim]
        current_obs_noise = extended_state[2*self.state_dim:2*self.state_dim + self.obs_dim]
        
        ## noise mean is included when with_noise=False
        propagated_state = self.model.forward_model(current_state, current_time, with_noise=False) + current_state_noise
        propagated_obs = self.model.observation_operator(propagated_state, next_time, with_noise=False) + current_obs_noise
        
        return [propagated_state, propagated_obs]
        
    def update(self, propagated_sigma_states, propagated_sigma_obs, true_observation):
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
                propagated_obs_mean, propagated_obs_cov,
                updated_state_mean, updated_state_cov]
        
        


