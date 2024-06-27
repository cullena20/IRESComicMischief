# Regularization

* In the original code, binary classification has a regularization term and multi task classification does not
* This regularization term explodes the loss of binary to around 500, where multi task without it is less than one

# Major Issue

* The output of the model is sometimes NaN, I think this has to do with the input data
    * This emerges from NaN's in the base_model!
    * Text Masking and Text positions are reversed (should it be like this?)