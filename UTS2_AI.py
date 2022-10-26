#Initializing numpy
import numpy as np

#Creating Programs for Variable Inputs
inputs	=  [[2.6, 1.6, 6.7, 6.2, 6.7, 1.4, 5.0, 2.8, 0.9, 6.3],
            [3.5, 3.4, 6.1, 2.8, 6.8, 5.5, 1.0, 6.7, 3.3, 0.8],
            [2.2, 5.5, 5.5, 2.1, 6.7, 2.3, 4.7, 3.3, 6.2, 5.6],
            [6.5, 6.9, 1.3, 0.5, 0.2, 1.7, 3.2, 1.2, 3.7, 3.2],
            [5.8, 2.9, 2.9, 6.9, 4.7, 0.5, 3.8, 4.4, 6.3, 1.7],
            [0.7, 5.0, 6.5, 3.4, 3.3, 6.8, 6.5, 6.3, 4.0, 0.2]]

#Creating Program for Hidden Layer 1
weights =  [[0.12, 0.11, 0.07, 0.14, 0.04, 0.09, 0.10, 0.09, 0.13, 0.14],
            [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
            [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39],
            [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49],
            [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]]

#Creating Hidden Layer 1 Bias
bias = [3, 4, 5, 6, 8]

#Creating Program For Output Hidden Layer 1
output  = np.dot(inputs, np.array(weights).T) + bias

#Creating Programs For Weight Hidden Layer 2
weights2 = [[0.1, 0.2, 0.3, 0.4, 0.5],
            [0.48, 0.89, 0.99, 0.09, 1.00],
            [0.09, 0.23, 0.27, 0.32, 0.21]]

#Creating Hidden Layer 2 Bias
bias2 = [2, 5, 9]

#Creating Program For Output Hidden Layer 2
output2 = np.dot(output, np.array(weights2).T) + bias2	

#Printing Output
print(output2)