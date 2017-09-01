## Course2 
###Week1 
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


    
  
