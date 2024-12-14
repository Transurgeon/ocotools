import numpy as np
from sklearn.svm import SVR
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt

# Initialization
min_val = 464  # smallest max-load in 1997-1998
max_val = 876  # largest max-load in 1997-1998

# Reading Data
x, y = load_svmlight_file(f='/home/willyzz/Documents/ocotools/constrained-SOONO/data/eunite2001')
x = x.toarray()
# Training the SVM Model
model = SVR(C=4096, gamma=0.0625, epsilon=0.5)
model.fit(x, y)

# Reading Test Data
tx, ty = load_svmlight_file(f='/home/willyzz/Documents/ocotools/constrained-SOONO/data/eunite2001.t')

# Prediction Loop
p = np.zeros(31)
for i in range(31):
    if i == 0:
        txi = tx[i, :]
    else:
        txi = np.concatenate([tx[i, :9], [(p[i-1] - min_val) / (max_val - min_val)], tx[i-1, 9:]])
    p[i] = model.predict([txi])[0]

# Calculating Metrics
mape = 100 / 31 * np.sum(np.abs((p - ty) / ty))
mse = np.mean((p - ty) ** 2)

print(f'MAPE: {mape}')
print(f'MSE: {mse}')

# Plotting Results
plt.plot(range(1, 32), p, '--', label='predicted')
plt.plot(range(1, 32), ty, '-', label='real')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Load')
plt.title('Predicted vs Real Load')
plt.grid(True)
plt.show()