# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

fashion_train_df = pd.read_csv('/Users/jasonfang/Work/Dataset/Practical/P39-Fashion-MNIST-Datasets/fashion-mnist_train.csv', sep = ',')
fashion_test_df = pd.read_csv('/Users/jasonfang/Work/Dataset/Practical/P39-Fashion-MNIST-Datasets/fashion-mnist_test.csv', sep = ',')

# visualization
fashion_train_df.head()
fashion_test_df.tail()
print(fashion_train_df.shape)
print(fashion_test_df.shape)

training = np.array(fashion_train_df,dtype = 'float32')
testing = np.array(fashion_test_df,dtype = 'float32')

i = random.randint(1,60000)
plt.imshow(training[i ,1:].reshape(28,28))
label = training[i,0]
print(label)

# 0 => T-shirt/top
# 1 => Trouser
# 2 => Pullover
# 3 => Dress
# 4 => Coat
# 5 => Sandal
# 6 => Shirt
# 7 => Sneaker
# 8 => Bag
# 9 => Ankle boot

# Define the dimensions of the plot grid
W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_training = len(training) # get the length of the training dataset

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables

    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index
    axes[i].imshow( training[index,1:].reshape((28,28)) )
    axes[i].set_title(training[index,0], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


