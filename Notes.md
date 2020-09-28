## Course1 Neural Networks and Deep Learning
### Week1
- Single neuron
- Supervised Learning



## Course2 - Hyperparameter tuning, Regularization and Optimization
### Week1
- Applied ML is highly iterative, hyperparameter choices aren't clear initially
- Training/Dev(hold-out crossvalidation)/Test sets 
- In Big Data era (1Million training samples etc) 98%/1%/1% split isn't uncommon
- Bias / Variance of the model (look at the training set, dev set errors for clues) 
    Assuming base error rate 0% i.e humans etc can do the job very accurately....
    - High Bias - Underfitting the data 
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
    - When input size is large (~5 million), going over all data thru each iteration makes gradient descent slow. We want Gradient descent to learn weights quickly and not wait for all the input data to be seen before making progress.
    - Mini batch size = m => Normal batch gradient descent
        * takes big steps and converges towards the minima in each step
        * Vectorizing inputs helps but the grad descent iteration is longer
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
    
 - Adam's optimization: Momentum + RMSProp 
 
 - Saddle points and Plateaus are more common in high dimentional spaces. Local optima are much more unlikely though.
 
### Week3 

 - Hyperparameter tuning: Use random points in the hyperparameter space, go from coarse to fine. This way each dimention gets to see many values reflected in the various random points.
 - Use the approproate scale while sampling hyperparameters. For eg it might be beneficial to choose a logarithmic scale for alpha, beta
 - *Train many models in parallel* if you have the time/computation, with different hyperparameters
 - Batch Normalization
 - Programing frameworks (like TensorFlow)
    - The two main object classes in tensorflow are Tensors and Operators.
        - Create a graph containing Tensors (Variables, Placeholders ...) and Operations (tf.matmul, tf.add, ...)
        - Create a session, Initialize the session
        - Run the session to execute the graph
        - The backpropagation and optimization is automatically done when running the session 
  - tf.Constant() , tf.Variable(), tf.global_variables_initializer(), tf.multiply(), tf.Session(), tf.placeholder()
 

## ML strategies, Course3
### Week1 
- Chain of assumptions in ML 
    - Fit training set well on cost function 
    - Fit dev set well on cost function 
    - Fit test set well on cost function
    - Performs well in real world 
- Setting up your goal
    - Use a single number Evaluation Metric
    - Optimizing and Satisficing metrics: If you care about several metrics make one of them optimizing and the remining ones satisficing (i.e any value subject to a threshold is satisfactory) 
        * For eg. in an image classifier, Accuracy is the optimizing metric, subject to Runtime(satisficing metric) being less than say 100ms. 
    - Select Dev/Test sets from the **same distribution**. THis prevents optimizing the model to a dev set and later having to shift goalpost to satisfy the test set nuances. Use random sampling and choose dev/test set from datra you assume to see in the future. 
    - Train/dev/test data split 
        * Historically for small data it was 70/30 or 60/20/20  split. With big data (Million plus inputs) its ~98/1/1
    - Other consideration: If you do well with metric + dev/test set but this doesnt translate to doing well on real world application, time to change either of the two. For eg. Cat classifier doing well with dev/test cat images but not so well when confronted with low res, blurry images from users. 
- Comparing to human level performance: (Bayes optimal error -> lowest theoritical error that can be reached for a X->y mapping)
        * Typically, systems dont improve a huge lot after surpassing human level performance because a) the human level performance is pretty close to Bayes optimal performance and b) lot of tactics can be used to improve systems to come upto human level performance.
    - Comparing to human level performance can guide whether we want to improve the bias or variance performance of a model. 
        - Avoidable bias is the amount of bias the system has compared to a human level performance
    - Human level error can act as a proxy to Bayes error. 
- Surpassing Human level performance:  When the training error of the system is less than the training error of experienced humans, it becomes unclear whether we focus our efforts on reducing Bias or variance (no eassy frame of reference)  
- ML surpasses human level performance in 
    - Logistics
    - Loan approval
    - Online advertising
    - Product recommendations
    -(typically Structured Ddata problems, Huge data input models, not a natural perception task)
   
- Error Analysis
    - Incorrectly labeled examples
        - In training set: DL algorithms are robust to *random* mislabeled errors. 
        - in dev/test set: carry out error analysis to see it this is causing significant % of the errors
        - Training set and dev/test data have slightly different distributions -> DL algos are OK with it
        - Dev and Test data *must* come from same distribution thought.. DL algos sensitive to this. 
    - Build first system quickly, then iterate. 
        - Set up dev/test set and metric
        - Build intitial system quickly
        - Use Bias/Variance analysis and Error analysis to prioritize next steps
    - Test set: a set of examples used only to assess the performance of a fully-trained classifier. After assessing the final model on the test set, ONE MUST NOT tune the model any further. 
- Training and testing on different distributions
    - Define a new set carved our ot the training set called training-dev set. The training set and training-dev set have *same* distribution. 
    - If Training-dev set performance is much worse that training set performance, we have a *Variance* problem since distribution is same
    - If Training-dev set performance much worse than dev-set performance, we have a *Data Mismatch* problem since since cause for this is different distribution
    - Data Mismatch is a new problem to solve when you have a different training and dev/test distributions
        - Carry out manual error analysis 
        - Make training data more similar or collect more data similar to dev/test sets
            - Artificial data synthesis. This works, but be careful that you arent synthesizing data from a small subset of all possible values..or else the model will overfit to your particular synthesized subset.
            
-  Learning from Multiple tasks: 
    - Transfer learning (A->B)makes sense when a) both tasks have same inputs (audio, image etc), b) you had a lot of data to train for A  and the new task B is relatively small 
    - Multitask learning is learning many tasks simultaneously
        - Unlike softmax ML which assigns single label to each image, Multitask learning assigns multiple labels.
        - Makes sense when a) set of tasks benefit from shared lower-level features, b) amount of data for each task is quite similar
        - Can train big enough neural networks
        
-End to End Deep learning: Neural networks are nowadays replacing a a System with a set of setps with a single end to end model.
    - Require alarge dataset for these problems.
    - Pros of E2E Deep Learning
        - E2E lets the data speak instead of being forced to reflect preconceptions for sub block designers
        - Less hand designing of components, features
    - Cons 
        - Need a large amount of data
        - Excludes potential useful hand-designed components (these play bigger role when training set isnt large)
    
    
    


        

    

    
  
