import numpy as np

def online_newtons_method(T, x0, gradient_func, hessian_func, beta, h, L):
    # Initialization
    gamma = min(beta, 2 * h / (3 * L))
    assert np.linalg.norm(x0 - x0_star) < gamma, "Initial x0 does not satisfy the condition"
    
    x = x0
    for t in range(T):
        # Play the decision xt
        xt = x
        
        # Observe the outcome at t: ft(xt)
        grad_ft = gradient_func(xt)
        
        # Update the decision: xt+1 = xt − Ht−1 (xt) ∇ft (xt)
        hessian_inv = np.linalg.inv(hessian_func(xt))
        x = xt - np.dot(hessian_inv, grad_ft)
        
    return x

# Example usage:
# Define the gradient and hessian functions for your specific problem
def gradient_func(x):
    # Example gradient function
    return np.array([2*x[0], 2*x[1]])

def hessian_func(x):
    # Example hessian function
    return np.array([[2, 0], [0, 2]])

# Initial decision
x0 = np.array([1.0, 1.0])
x0_star = np.array([0.0, 0.0])  # Example optimal point

# Parameters
beta = 1.0
h = 1.0
L = 1.0

# Number of iterations
T = 10

# Run the algorithm
final_decision = online_newtons_method(T, x0, gradient_func, hessian_func, beta, h, L)
print("Final decision:", final_decision)