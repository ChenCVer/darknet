import numpy as np

# train
inputs = np.ones([5, 5])
mask = np.random.rand(inputs.shape[0], inputs.shape[1]) < 0.5
inputs = inputs * mask

print(inputs)

# test
inputs = np.ones([5, 5]) * 0.5
print(inputs)

print("-------------------")

# invert dropout
# train
inputs = np.ones([5, 5])
mask = np.random.rand(inputs.shape[0], inputs.shape[1]) < 0.5
inputs = inputs * mask / 0.5
print(inputs)

# test
inputs = np.ones([5, 5])
print(inputs)