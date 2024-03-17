import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define compute_k_nearest_neighbors function
def compute_k_nearest_neighbors(sample, new_sample, k):
    distances = np.zeros(len(sample)) 
    
    # computing the euclidean distance between the new sample and all the samples in the dataset
    for i in range(len(sample)):
        distances[i] = np.linalg.norm(sample[i] - new_sample)
        
    # getting indices of the k smallest distances array excluding the current sample itself
    sorted_indices = np.argsort(distances)
    return sorted_indices[1:k+1]  # Exclude the current sample and take the top k nearest neighbors

# Load the dataset
df = pd.read_csv('/kaggle/input/dataset1/train.csv')

# Filter the rows for label 3
label_3_data = df[df['label'] == 3]

# Take first 20 images
label_3_data_20 = label_3_data.head(20)

# Extract pixel values for first 20 images
pixel_values_20 = label_3_data_20.drop(columns=['label']).values

# Reshape pixel values into 28x28 matrices for first 20 images
images_20 = pixel_values_20.reshape(-1, 28, 28)

# Function to generate synthetic images
def generate_synthetic_image(sample, neighbor_index, gap):
    return sample + gap * (sample - images_20[neighbor_index])

# Create subplots for each image along with its two nearest neighbors and synthetic images
to_be_produced = 20
fig, axs = plt.subplots(to_be_produced, 5, figsize=(20, to_be_produced * 3))

for i in range(to_be_produced):
    axs[i, 0].imshow(images_20[i], cmap='gray')
    axs[i, 0].set_title('Image {}'.format(i + 1))
    axs[i, 0].axis('off')

    # Compute nearest neighbors for each of the first 20 images
    neighbors = compute_k_nearest_neighbors(images_20, images_20[i], 2)
    
    # Show nearest neighbor images and generate synthetic images
    for j, neighbor_index in enumerate(neighbors):
        axs[i, j+1].imshow(images_20[neighbor_index], cmap='gray')
        axs[i, j+1].set_title('Neighbor {}'.format(j + 1))
        axs[i, j+1].axis('off')
        
        # Calculate difference between sample and neighbor
        dif = images_20[neighbors[j]][0] - images_20[i][0]
        
        # Generate random gap
        gap = np.random.uniform(0, 1)
        
        # Generate synthetic image
        synthetic_image = generate_synthetic_image(images_20[i], neighbors[j], gap)
        axs[i, j+3].imshow(synthetic_image, cmap='gray')
        axs[i, j+3].set_title('Synthetic {}'.format(j + 1))
        axs[i, j+3].axis('off')

plt.tight_layout()
plt.show()
