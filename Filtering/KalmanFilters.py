import numpy as np

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
    
class EnsembleKalmanFilter:
    def __init__(self, forward_model, obs_operator, obs_cov):
        self.forward_model = forward_model
        self.obs_operator = obs_operator
        self.obs_cov = obs_cov
    
    def dataAssimilation(self, current_state_mat, obs_vec):
        # TODO
        pass
    
    def modelPropagation(self, updated_state):
        # input: updated_state (state after assimilating the data from the observations), 
        #        updated_state_cov (covarianceMatrix of the assimilated state), 
        #        forward_model_noise_cov (covariance matrix of noise in forward model)
        # output: nextState (state after propagating the assimilated state through the model)
        #        nextStateCovariance (covariance matrix of the next state)
        
        next_state = self.forward_model(updated_state)
        return next_state
    
class UnscentedKalmanFilter:
    def __init__(self) -> None:
        # TODO
        pass
    
class ExtendedKalmanFilter:
    def __init__(self) -> None:
        # TODO
        pass