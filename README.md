# DeepLearningCourse
Assignments and learnings from Coursera's Deep learning course

## The broad steps in implementing neural netwoks are as follows

1) Define Model Structure 
    - number of layers, # neurons per layer
    - activation functions to use
2) Initialize model parameters
3) Iterate through this
    - Calculate cost function J using forward propogation
    - Calculate the current gradients using backward propogation
    - Update weights 

## Common Python commands used
```
import numpy as np  
m_train = train_set_x_orig.shape[0]     => use .shape, .reshape to access dimensions, vectorize matrices into columns
np.zeros([dim,1])                       
dw = np.dot(X, (A - Y).T) /m            => dot product equivalent to Matrix multiplication ( 2D matrices)
c1 = np.multiply(Y, np.log(A))          => Element-wise matrix multiplication
db = np.sum(A - Y ) /m                  => Subtract matrices A - Y (broadcasting if needed), then add element-wise to give a single number
```
    
