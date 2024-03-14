import numpy as np
from numpy.linalg import solve

class SquaredExpKernel():
    def __init__(self, params):
        self.name="squared_exponential"
        self.params = params
        
    def cov_kernel(self, x1, x2):
        # x1 = n_points_1 x inp_dim
        # x2 = n_points_2 x inp_dim
        # params = {'sigma_f', 'l'}
        cov_mat = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1).reshape(1, -1) - 2*np.dot(x1, x2.T)
        cov_mat = (self.params['sigma_f']**2)*np.exp(-cov_mat*(0.5/(self.params['l']**2)))
        return cov_mat

class GaussianProcess:
    def __init__(self, inp_x, inp_y, params, kernel_name):
        # inp_x = n_points x inp_dim
        # inp_y = n_points x out_dim
        self.inp_dim=inp_x.shape[1]
        self.out_dim=inp_y.shape[1]
        
        if self.out_dim != 1: # currently only for out_dim=1
            raise NotImplementedError
        
        self.inp_x=inp_x
        self.inp_y=inp_y
        self.params=params
        if kernel_name=="squared_exponential":
            self.cov_kernel=SquaredExpKernel(params).cov_kernel
        else:
             raise NotImplementedError
                
        # Using Cholesky decomposition
        inp_cov=self.cov_kernel(inp_x, inp_x) + (params['inp_noise']**2)*np.eye(len(inp_x))
        self.L_inp_cov=np.matrix(np.linalg.cholesky(inp_cov))
        
        self.mean=self.prior_mean(inp_x)
    
    def evaluate(self, eval_x):
        # eval_x = n_points_eval x inp_dim
        eval_cov=self.cov_kernel(eval_x, eval_x) # n_points_eval x n_points_eval
        cross_cov=self.cov_kernel(self.inp_x, eval_x) # n_points x n_points_eval
        
        mean = self.prior_mean(eval_x) + cross_cov.T@solve(self.L_inp_cov.H,
                                                           solve(self.L_inp_cov, self.inp_y - self.mean))
        
        temp_cov_red = solve(self.L_inp_cov, cross_cov)
        cov = eval_cov - temp_cov_red.T@temp_cov_red
        return mean, cov
        
    def prior_mean(self, x1):
        return np.zeros((len(x1), self.out_dim))