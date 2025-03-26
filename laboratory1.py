"""
Problem 1: We investigate here in more detail the coi tossing problem. A coin is tossed N times and heads come up H times. Let theta denote the probability of heads in one flip. 
a) compute analytically the posterior probability for thetat, for the case of a Jeffrys prior and a uniform prior on theta. 
"""

import numpy as np
import scipy
from scipy.stats import beta
import matplotlib.pyplot as plt


N = 1000
p = 0.3





class my_class():
    def __init__(self,N,M,p):
        self.N = N
        self.p = p
        self.M = M

    def pseudo_data(self):
        data = np.empty((self.M,self.N))
        for i in range(self.M):
            data[i] = np.random.rand(self.N)
        data_r = np.empty((self.M,self.N))
        data_r[data>self.p] = 0
        data_r[data<=self.p] = 1        
        return data_r

    def pdf_(self,H,theta):
        return beta(H,self.N - H).pdf(theta)

    def plot(self):
        H = (self.pseudo_data())
        H = len(H[0][H[0]==1])
        print(H)
        x = np.linspace(0,1,self.M)
        y = self.pdf_(H,x)
        plt.plot(x,y)
        plt.show()

beta_class = my_class(10,1000,0.1)
beta_class.plot()
        

        

