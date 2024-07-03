I keep track of possible errors and notes on how we implement things. If we have bugs, consult here for possible issues.

# Regularization

* In the original code, binary classification has a regularization term and multi task classification does not
* This regularization term explodes the loss of binary to around 500, where multi task without it is less than one

# Major Issue

* The output of the model is sometimes NaN, I think this has to do with the input data
    * This emerges from NaN's in the base_model!
    * Text Masking and Text positions are reversed (should it be like this?)
    * The NaNs were in my code before major refactoring (splitting multi task into more heads, having different training loop)
    See if they were there originally.
* I suspect issues may arise from minor details in how input shapes and padding are handled that might have been slightly 
  modified.
* I only see this occur on multi task training after some training runs - it must come from the gradients somehow
* It's also possible that this issue comes from some specific pieces of training data, but I don't think so (worth asking about)


# Padding & Masking

* I pad and mask in the dataset, maybe better to do this in the data loader?
* Also, text padding was previously done to have padding before: [11, 12, 14] would become [0, 0, 0, 11, 12, 14] instead of [11, 12, 14, 0, 0, 0]
  I changed this, but maybe this isn't good

# The Model

* The model has some weird implementation things
* I don't think we should mess with it too much though because we have to use pretrained weights
* Carefully check all the shapes if we have issues, because although it runs there may be some other issue

# Dynamic Pretraining

* There are a lot of little details in this code right now regarding indices
* We track the indices of each dataset as we make batches which may lead to errors if done incorrectly
* Also data is not randomly shuffled and training stops when one dataset is completed
* Also there are redundancies in the code that should be modularized