import numpy as np        

class LinearRegression():
    def __init__(self, x_train, y_train):
        # inp_x = n_points x inp_dim
        # inp_y = n_points x out_dim
        
        self.inp_dim=x_train.shape[1]
        self.out_dim=y_train.shape[1]
        
        if self.out_dim != 1: # currently only for out_dim=1
            raise NotImplementedError
        
        self.x_train=x_train
        self.out_train=y_train
        
        x_feature=self.get_features(x_train)
        
        ## SVD Decomposition
        u,s,vh=np.linalg.svd(x_feature)
        y_star=u.T@y_train
        beta_star=np.zeros((x_feature.shape[1], self.out_dim))
        beta_star[:len(s)]=y_star[:len(s)]/s[:, None] # Can do cutoff based to avoid very small singular values
        self.beta=vh.T@beta_star
        
        
    def get_features(self, x_inp):
        #move the data into the feature space
        x_feature=np.hstack([np.ones((x_inp.shape[0], 1)), x_inp]) # x -> [1, x]
        return x_feature
    
    def evaluate(self, x_eval):
        return self.get_features(x_eval)@self.beta
 