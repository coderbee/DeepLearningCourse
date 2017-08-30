## Course2 Week1 
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
    - Regularization
    - NN architecture search
    
