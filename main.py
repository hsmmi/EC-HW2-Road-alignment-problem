# Read map.npy

import numpy as np
import matplotlib.pyplot as plt

# Read map.npy
map = np.load("map.npy")


# Display map
fig = plt.figure()
im = plt.imshow(map)

# Display map
plt.show()

print("pause")
