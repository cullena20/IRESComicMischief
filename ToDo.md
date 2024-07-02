# To Do 

We are working to evaluate different curriculum learning esque strategies on finetuning the HCA based model on Comic Mischief data.
We may also wish to evaluate these strategies on pretraining methods (and finding different pretraining tasks), and if we find 
something interesting to perform further evaluations on a wider array of models.

## Methods To Implement

* Naive Approach 1: Train all tasks together by adding loss functions (learned weights or unweighted)
    - IMPLEMENTED
    - Currently the weights are just parameters in the models, but this may have issues
* Naive Approach 2: Train one task at a time
    - Way One: Just have single tasks models - IMPLEMENTED
        - We are able to train this as is right now by simply defining models with different task specific heads
        - But we haven't tested this yet
    - Way Two: Truly train one at a time on multi task model (finish one before moving onto the next)
* Approach 3: Dynamic Difficulty Sampling - IMPLEMENTED 
    - The original inspiration from the Dynamic Pretraining paper
    - Control the ratio of each task in a batch at a single time by keeping track of task losses
    - Within a single batch (which is composed of several tasks), we add together the losses somehow. This could be reweighted, but    note potential issues that may arise from balancing the loss weights and the losses used in the ratios (combining these two would be a novel thing to try and may not be so naive)
* Approach 4: Dynamic Stop And Go - WIP
    - We have round robin sampling implemented (cycle between tasks one batch at a time, instead of whole dataset)
    - Need to incorporate dynamic stop and go into this loop
    - Also the method of training one batch at a time, using the same input data may have flaws (not so hard to generalize to several tasks). Currently we grab a batch of n examples, and use that batch to train each task.
* Approach 5: Some kind of Curriculum Learning
    - 12-in-1/DSG paper does this by classifying hardest and easiest tasks by convergence time, training on that, and then training everything
      using their DSG method. We may wish to do something similar.
    - Probably need to survey literature better for this.
* Approach 6: Loss Scaling Approaches:
    - Learnable weights on each loss is already implemented
    - Also see uncertainty weights (harder I think), GradNorm, and PCGrad (chang directions)
* Approach 7: Train One Modality at a Time:
    - This requires further model modularization. However it should be done in a way that allows us to use pretrained weights.
    - Could it be that in the original work this was tested by just zeroing out other modalities like now? Ask.
* Possible Other Approach? Task Groupings
    - Identify best groupings that belong together, use these to train separate models that optimize performance on downstream tasks
    - Could also give good insight on how tasks relate and reasons for current results in ablations
* Possible combinations of above techniques and new ideas if we come up with:
    - One: Loss Reweighting Approaches with Anything that adds losses (dynamic pretraining)
        - Possibly even to precompute loss weights for round robin or DSG based methods
