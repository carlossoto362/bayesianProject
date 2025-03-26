import numpy as np
import matplotlib.pyplot as plt

def linear_(x, theta):
    """
    Linear function: y = theta * x.
    
    Parameters:
    x (array or float): Input values.
    theta (float): Slope parameter.
    
    Returns:
    array or float: The result of the linear function.
    """
    return theta * x

def linear_measure(x, theta, sigma):
    """
    Simulate measurements with noise added to the linear function.
    
    Parameters:
    x (array or float): Input values.
    theta (float): Slope parameter.
    sigma (float): Standard deviation of the noise.
    
    Returns:
    array or float: The noisy measurements.
    """
    try:
        size = len(x)
    except TypeError:
        size = 1
    return linear_(x, theta) + np.random.normal(loc=0, scale=np.sqrt(sigma), size=size)


def tau_a(tauS, yS, y, Delta):
    """
    Compute the adjusted time constant tau_a.
    
    Parameters:
    tauS (float): Reference time constant.
    yS (float): Reference y value.
    y (array or float): Input y values.
    Delta (float): Scaling factor.
    
    Returns:
    array or float: The adjusted time constant values.
    """
    return tauS * np.exp((y - yS) ** 2 / (2 * Delta ** 2))

def logB_(x1,x2,x3,sigma,tau):
    return -0.5*np.log((x1/sigma)+ (x2/sigma) + (x3/tau))
# Define constants for the simulation
x0 = 0.5
x1 = 1
xmax = 3
sigma = 0.1 ** 2
theta = 1
tauS = sigma / 5  # tau star
Delta = 0.2

x2 = np.linspace(0, 3.1, 1000)

yS_values = np.linspace(linear_(0, theta), linear_(3, theta), 5)
for yS in yS_values:
    tau = tau_a(tauS, yS, linear_(x2, theta), Delta)
    logB = np.zeros(len(x2))
    for _ in range(100):

        logB +=  logB_(x0,x1,x2,sigma,tau)
    logB /= 100
    
    plt.plot(x2, np.abs(logB),label='mean logB values')
    plt.xlabel('x_f')
    plt.ylabel('Mean Utility function')
    
    plt.title('y*: {:.3f}'.format(yS))
    plt.legend()
    plt.savefig(f"logB_y_{yS:.3f}.png")
    plt.close()

# Compute and plot optimal experiment results
max_x = []
ySsss = np.linspace(linear_(0, theta), linear_(3, theta), 50)
for yS in ySsss:
    tau = tau_a(tauS, yS, linear_(x2, theta), Delta)
    logB = np.zeros(len(x2))
    for _ in range(100):
        logB += logB_(x0,x1,x2,sigma,tau)
    logB /= 100
    max_x.append(x2[np.argmax(np.abs(logB))])
    
plt.plot(ySsss, max_x, label='Optimal Experiment')
plt.xlabel('y*')
plt.ylabel('x_f')
plt.title('X_f using logB criterio')
plt.plot(ySsss, ySsss / theta)
plt.savefig('y_x_f.png')




