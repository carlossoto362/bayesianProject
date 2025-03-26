import numpy as np
import matplotlib.pyplot as plt

def linear_(x, theta):
    """
    Linear function: y = theta * x.
    
    Parameters:
    x (array): Input values.
    theta (float): Slope parameter.
    
    Returns:
    array: The result of the linear function.
    """
    return theta * x

def linear_measure(x, theta, sigma):
    """
    Simulate measurements with noise added to the linear function.
    
    Parameters:
    x (array): Input values.
    theta (float): Slope parameter.
    sigma (float): Standard deviation of the noise.
    
    Returns:
    array: The noisy measurements.
    """
    try:
        size = len(x)
    except TypeError:
        size = 1
    return linear_(x, theta) + np.random.normal(loc=0, scale=np.sqrt(sigma), size=size)

def u1(x0, x1, sigma, tau, x):
    """
    Compute the Utility function u1 with parameters x0, x1, sigma, tau, and xmax.
    
    Parameters:
    x0 (float): Input value.
    x1 (float): Input value.
    sigma (float): A parameter of the model.
    tau (float): A parameter of the model.
    x (float): x_f.
    
    Returns:
    float: The result of the model function.
    """
    return ((x1**2 + x0**2) / sigma + x**2/ tau)

def sigma1_(x1, x0, sigma):
    """
    Compute sigma_12 from x1, x0, and sigma.
    
    Parameters:
    x1 (float): Input value.
    x0 (float): Input value.
    sigma (float): A parameter of the model.
    
    Returns:
    float: The standard deviation after the first experiment.
    """
    return sigma / (x1**2 + x0**2)

def m1_(x1, x0, y1, y0):
    """
    Compute m_12.
    
    Parameters:
    x1 (float): Input value.
    x0 (float): Input value.
    y1 (float): Input value.
    y0 (float): Input value.
    
    Returns:
    float: The mean of the prior after the first experiment.
    """
    return (x1 * y1 + x0 * y0) / (x1**2 + x0**2)

def u2(x0, x1, sigma, tau, x, Delta, y, m1, sigma1):
    """
    Compute the utility function u2.
    
    Parameters:
    x0 (float): Input value.
    x1 (float): Input value.
    sigma (float): A parameter of the model.
    tau (float): A parameter of the model.
    x (float): x_f.
    Delta (float): A parameter of the model.
    y (float): A parameter of the model.
    m1 (float): A computed parameter.
    sigma1 (float): A computed parameter.
    
    Returns:
    float: The utility funciton for model 2.
    """
    sqrt = (1 + (x**2) * sigma / Delta) ** (-1/2)
    exp = np.exp(-(0.5) * ((m1**2 / sigma1) + (y**2 / Delta) - (m1 / sigma1 + x * y / np.sqrt(Delta))**2 / (1 / sigma1 + x**2 / Delta)))
    return ((x1**2 + x0**2) / sigma + (x / tau) * sqrt * exp)

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
    
def plot_u(x0_in, x1_in, sigma_in, tau_in, tauS_in, x_in, Delta_in, yS_in, m1_in, sigma1_in, theta_in):
    """
    Plot the functions u1 and u2 on the same graph.
    
    Parameters:
    x0_in (float): Input value.
    x1_in (float): Input value.
    sigma_in (float): A parameter of the model.
    tau_in (float): A parameter of the model.
    tauS_in (float): A parameter of the model.
    x_in (array): Array of x values.
    sigmaB_in (float): A parameter of the model.
    yS_in (float): A parameter of the model.
    m1_in (float): A computed parameter.
    sigma1_in (float): A computed parameter.
    theta_in (float): A parameter of the model.
    """
    ue_ = u1(x0_in, x1_in, sigma_in, tau_in, x_in)
    xmaxe = np.argmax(ue_)  # Find the index where u1 is maximized
    ua_ = u2(x0_in, x1_in, sigma_in, tauS_in, x_in, Delta_in, yS_in, m1_in, sigma1_in)
    xmaxa = np.argmax(ua_)  # Find the index where u2 is maximized
    
    if np.max(ue_) > np.max(ua_):
        xmax = x_in[xmaxe]
    else:
        xmax = x_in[xmaxa]
    
    # Print the result of a particular formula
    
    ymin = np.min([np.min(ue_), np.min(ua_)])
    ymax = np.max([np.max(ue_), np.max(ua_)])
    
    #plt.title('Delta: {}'.format(Delta_in))  # Set plot title with the sigmaB value
    plt.plot(x_in, ue_, label='Utility function e', color='blue')  # Plot u1 (ue)
    plt.plot(x_in, ua_, label='Utility function a', color='green')  # Plot u2 (ua)
    plt.text(.01, 1., '$\Delta$' + ': {:.4f}'.format(np.sqrt(Delta_in)), ha='left', va='bottom', transform=plt.gca().transAxes,fontsize=10)  # Add text to the plot
    plt.vlines(xmax, ymin=ymin, ymax=ymax, linestyles='dashed', color='red', label='$argmax_x(U(x))$')  # Add vertical line at xmax
    plt.xlabel('x')
    plt.ylabel('U(x)')
    plt.legend()  # Show legend
    plt.savefig('{:.3f}.png'.format(Delta_in))  # Display the plot
    plt.close()

# Define constants for the simulation
x0 = 0.5
x1 = 1
xmax = 3
sigma = 0.1 ** 2
theta = 1
tau = 0.1
tauS = sigma / 5 # tau star
yS = 1.5 #y star

# Generate noisy measurements for y0 and y1
y0, y1 = linear_measure(np.array([x0, x1]), theta, sigma)

# Loop through different values of sigmaB and plot the results
for Delta_sqrt in np.linspace(0.4, 0.7, 6): 

    Delta = Delta_sqrt ** 2
    sigma1 = sigma1_(x1, x0, sigma)
    m1 = m1_(x1, x0, y1, y0)
    x = np.linspace(0, xmax, 1000)
    #plot_u(x0, x1, sigma, tau, tauS, x, Delta, yS, m1, sigma1, theta)  # Call plotting function
    

# Compute and plot optimal experiment results
max_x = []
ySsss = np.linspace(linear_(0, theta), linear_(3, theta), 50)
Delta = 0.6
x2 = np.linspace(0, 3.1, 1000)
for yS in ySsss:
    logB = np.zeros(len(x2))
    for _ in range(100):
    	sigma1 = sigma1_(x1, x0, sigma)
    	m1 = m1_(x1, x0, linear_measure(x1, theta, sigma), linear_measure(x0, theta, sigma))
    	logB += u2(x0, x1, sigma, tauS, x2, Delta, yS, m1, sigma1)
    logB /= 100
    max_x.append(x2[np.argmax(logB)])
    
plt.plot(ySsss, max_x, label='Optimal Experiment')
plt.xlabel('y*')
plt.ylabel('x_f')
plt.title('X_f using 1/var critero')
plt.plot(ySsss, ySsss / theta)
plt.savefig('y_x_f_u.png')


