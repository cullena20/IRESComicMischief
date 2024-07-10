I keep track of possible errors and notes on how we implement things. If we have bugs, consult here for possible issues.

# The Model - Updated 7/5/24

* The model has some weird implementation things
* Video and Audio are already embedded prior to being fed in while text is only tokenized and is embedded via BERT in the model
* Then padding and masking for video and audio are handled by inserting rows of 0s corresponding to each empty embedded token
* For BERT, we add padding and masking to the beginning of the tokens -> but when we run it through BERT, 0 tokens do not get embedded as 0 
* So that makes me think the way that audio and video are handled is very strange. That's why they further embed it
  * also note what BERT returns: hidden states, attention, loss, and logits -> we just grab the logits which correspond to our embeddings

* After fixing how padding is done to allign with the original code, everything should truly be equivalent now.

7/9


# Training and Evaluation Loop
* There is serious modularization that needs to be done. For example, all the validation stuff does
not work for dynamic difficulty sampling as is. One easy work around is to have separate per epoch
trianing code that is then wrapped in a larger function
* In general these different strategies can be modularized better. Also how the validation is done 
should really be in a seperate function. I suppose with a larger meta loop it works fine as is.
Maybe do something like the original code and have a class in charge of all of this.
* I may want to return more or less with the validation and training code (e.g. optimizer history).
Also for now I just pickle the model, but probably we want to save the model state somehow.

# Regularization

* In the original code, binary classification has a regularization term and multi task classification does not
* This regularization term explodes the loss of binary to around 500, where multi task without it is less than one


# Dynamic Pretraining

* There are a lot of little details in this code right now regarding indices
* We track the indices of each dataset as we make batches which may lead to errors if done incorrectly
* Also data is not randomly shuffled and training stops when one dataset is completed
* Also there are redundancies in the code that should be modularized