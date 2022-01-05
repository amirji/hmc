import numpy as np
import matplotlib.pyplot as plt

def HMC(current_State, n):
    """
    A function that do a series of gradient-informed steps to produce a Metropolis proposal.
    Parameters: 
        1. Current_state: An array that store the states of the systme.
        2. Number of steps
        Return: An array representing a chain of states
    """
    # Initialize the chain
    chain = np.zeros((n, len(current_State)))
    chain[0] = current_State
    # Initialize the momentum
    momentum = np.random.normal(size=len(current_State))
    for i in range(1, n):
        # Update the momentum
        momentum = np.random.normal(size=len(current_State))
        # Do a leapfrog steps
        new_state, new_momentum = leapfrog(chain[i-1], momentum, 0.01)
        # Calculate the acceptance probability
        accept_prob = min(1, np.exp(log_posterior(new_state)-log_posterior(chain[i-1])))
        # Generate a random number to determine whether to accept or reject the proposal
        if np.random.uniform() < accept_prob:
            chain[i] = new_state
        else:
            chain[i] = chain[i-1]
    return chain
def leapfrog(current_state, current_momentum, step_size):
    """
    A function that do a series of gradient-informed steps to produce a Metropolis proposal.
    Parameters: 
        1. Current_state: An array that store the states of the systme.
        2. Number of steps
        Return: An array representing a chain of states
    """
    # Initialize the new state and new momentum
    new_state = current_state
    new_momentum = current_momentum
    # Do half a step for momentum
    new_momentum -= 0.5*step_size*grad_log_posterior(new_state)
    # Do a full step for state
    new_state += step_size*new_momentum
    # Do half a step for momentum
    new_momentum -= 0.5*step_size*grad_log_posterior(new_state)
    return new_state, new_momentum
def log_posterior(x):
    """
    A function that calculate the log posterior of a given state.
    Parameters: 
        1. x: An array that store the states of the systme.
        Return: The log posterior of the given state.
    """
    return -0.5*np.dot(x, x)
def grad_log_posterior(x):
    """
    A function that calculate the gradient of the log posterior of a given state.
    Parameters: 
        1. x: An array that store the states of the systme.
        Return: The gradient of the log posterior of the given state.
    """
    return -x


# Initialize the chain
chain = np.zeros((10000, 2))
chain[0] = np.array([-1, 1])

# Do a series of HMC steps
for i in range(1, 10000):
    # Update the momentum
    momentum = np.random.normal(size=2)
    # Do a leapfrog steps
    new_state, new_momentum = leapfrog(chain[i-1], momentum, 0.01)
    # Calculate the acceptance probability
    accept_prob = min(1, np.exp(log_posterior(new_state)-log_posterior(chain[i-1])))
    # Generate a random number to determine whether to accept or reject the proposal
    if np.random.uniform() < accept_prob:
        chain[i] = new_state
    else:
        chain[i] = chain[i-1]
# Plot the trace plot
plt.plot(chain[:, 0])
plt.show()
plt.plot(chain[:, 1])
plt.show()

