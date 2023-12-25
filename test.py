import numpy as np
from sklearn.decomposition import PCA

# Sample 2D data
data = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

# Fit PCA on the dataset
pca = PCA(n_components=2)
pca.fit(data)

# Get the principal components
principal_components = pca.components_

# Get the standard deviation (sigma) of each principal component
# The singular values correspond to the length of the semi-axes of the ellipsoid
# representing the data in the PCA space and are related to the standard deviation along
# the principal components. To get the standard deviation, we divide the singular values
# by the square root of the number of samples.
sigmas = pca.singular_values_ / np.sqrt(data.shape[0])

print("Principal components:")
print(principal_components)
print("Standard deviation along each principal component:")
print(sigmas)

# To visualize the data along the principal components and the principal components themselves,
# you can plot the data and the vectors that represent the components
import matplotlib.pyplot as plt

# Get the mean of the data
mean = pca.mean_

# Scatter plot for the data
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Original Data')

# Plot the principal components as vectors
for i, (comp, var) in enumerate(zip(principal_components, sigmas)):
    start, end = mean, mean + comp * var
    plt.annotate('', xy=end, xytext=start, arrowprops=dict(facecolor='red', width=2.0))
    plt.text(end[0], end[1], 'PC{}'.format(i+1))

# Plot settings
plt.axis('equal')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Data and Principal Components')
plt.show()
