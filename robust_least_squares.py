import dsp
import cvxpy as cp
import numpy as np

# Define the problem data
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([1, 2, 3])
M = 10  # Define the constant bound for r

# Define the variables
x = cp.Variable(2)
r = cp.Variable(3)

# Define the residuals
residuals = A @ x - b + r

# Define the objective function to minimize the maximum residual
objective = dsp.MinimizeMaximize(cp.norm(residuals, 2))

# Define the constraints
constraints = [cp.norm1(r) <= M]

# Formulate the problem
prob = dsp.SaddlePointProblem(objective, constraints)

# Solve the problem
prob.solve()

# Print the results
print("Optimal value:", prob.value)
print("Optimal x:", x.value)
print("Residuals:", residuals.value)