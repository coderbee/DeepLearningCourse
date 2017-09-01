## Course2 
### Week1 
- Applied ML is highly iterative, hyperparameter choices aren't clear initially
- Training/Dev(hold-out crossvalidation)/Test sets 
- In Big Data era (1Million training samples etc) 98%/1%/1% split isn't uncommon
- Bias / Variance of the model (look at the training set, dev set errors for clues) 
    Assuming base error rate 0% i.e humans etc can do the job very accurately....
    - High Bias - Underfitting the data. 
      - Training set error is high(~15%), implying the model didnt fit the dataset properly
      - In add'n if dev set error is  much lower (~30%) then model has high bias AND high variance
    - High Variance - Overfitting the data.
      - Training set error (~1%)  << Dev set error (~11%)
      
 -  Firstly, High Bias in the model?? Try this
    - Bigger network, more layers, neurons. 
    - Better NN architecture
    - Train for longer time 
 - Secondly, High Variance in model?? Try this
    - More training examples
    - _Regularization_
    - NN architecture search
    
  - Regularization is a common technique used to couter overfiting. It involves adding an extra term to the cost function J(w,b). 
    - L2 regularization: J(w,b) = (1/m)Sum_over_m(Loss function) + **lambda/2m * ||w||22** (~rms value of w's)
    - L1 regularization: J(w,b) = (1/m)Sum_over_m(Loss function) + **lambda/2m * ||w||11** (absolute values of w)
  
  - L2 Regularization is commonly used. Note that during backprop, we update weaights for the layers as follows
    - dw[l] is the partial derivative of dJ/dw. with L2 regularization, this now has extra term **(lambda/m)W[l]** (which is the parivative of the L2 term in J(w,b)
    - w[l] := w[l] - alpha * dw[l] = w[l] **(1 - alpha x lambda/m)** - alpha(from backprob term)
    - L2 regularization has effect of **weight decay** 

### Week2 

- Mini-batch gradient descent: Process a mini-batch of inputs X{i} at a time. FOr large inputs mini batch runs much faster than batch gradient descent.
    - In gradient descent we run fwd and backwd propagation over ALL training samples in a single iteration
    - When input size is large (~5 million), going over all data thru each iteration makes gradient descent slow. We want Gradient descent to learn weights quickly and not wait for all the input data od be seen before making progress.
    - Mini batch size = m => Normal batch gradient descent
        * takes big steps and converges towards the minima in each step
        * Vectorizing inputs helps but each grad descent epoch is longer
    - Mini batch size = 1 => Stochastic gradient descent
        * Can be noisy, it doesnt converge and takes a zig zag path while approaching minima
        * benefit of vectorization is lost with minibatch size only 1
    - Small training set ( ~2000 ) , use Batch gradient descent
    - Otherwise, use minibatch sizes like 64,128...1024 (Make minibatch input data fit in CPU/GPU memory)
 
 - Exponentially weighted averages: **V_t = beta * V_t-1  +  (1 - beta)* Theta_t**
    - roughly equal to average of the last 1/(1 -beta) theta samples
    - Not as accurate as moving window average, but easy to code and track history of a quantity efficiently (1 Line of code)  
    - Initialize with bias term to prevent initial condidion
        * use V_t / ( 1 - beta^t) in place of V_t 
        
 - Gradient descent with **Momentum** uses exp. weighted averages
    - instead of dW(or db) use V_dW = beta* V_dW + (1 - beta)* V_dW in the update equations
    - W := W - alpha * V_dW
    
 - **RMSprop** 
    - S_dW = beta2 * S_dW + (1 - beta2)dW2
    - W:= W - alpha* DW / sqrt(S_dW)
    
    
    


    
  
