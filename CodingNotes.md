# Regularization

* In the original code, binary classification has a regularization term and multi task classification does not
* This regularization term explodes the loss of binary to around 500, where multi task without it is less than one

# Major Issue

* The output of the model is sometimes NaN, I think this has to do with the input data
    * This emerges from NaN's in the base_model!
    * Text Masking and Text positions are reversed (should it be like this?)
    * The NaNs were in my code before major refactoring (splitting multi task into more heads, having different training loop)
    See if they were there originally.

# Padding & Masking

* I pad and mask in the dataset, maybe better to do this in the data loader?
* Also, text padding was previously done to have padding before: [11, 12, 14] would become [0, 0, 0, 11, 12, 14] instead of [11, 12, 14, 0, 0, 0]
  I changed this, but maybe this isn't good

# The Model

* The model has some weird implementation things
* I don't think we should mess with it too much though because we have to use pretrained weights