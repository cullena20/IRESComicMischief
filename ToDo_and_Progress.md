# To Do and Progress Tracker

We are working to evaluate different curriculum learning esque strategies on finetuning the HCA based model on Comic Mischief data.
We may also wish to evaluate these strategies on pretraining methods (and finding different pretraining tasks), and if we find 
something interesting to perform further evaluations on a wider array of models.

## Methods To Implement

* Naive Approach 1: Train all tasks together by adding loss functions (learned weights or unweighted)
    - IMPLEMENTED
* Naive Approach 2: Train one task at a time
    - Way One: Just have single tasks models - IMPLEMENTED
    - Way Two: Truly train one at a time on multi task model
* Approach 3: Dynamic Difficulty Sampling - IMPLEMENTED 
* Approach 4: Dynamic Stop And Go - WIP
    - We have round robin sampling implemented (cycle between tasks one batch at a time, instead of whole dataset)
    - Need to incorporate dynamic stop and go into this loop
    - Also the method of training one batch at a time, using the same input data may have flaws (not so hard to generalize to several tasks)
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
* Possible combinatinos of above techniques and new ideas if we come up with:
    - Can incorporate different loss based approaches with dynamic stop and go and anything that trains several tasks at once
    - Can even do DSG on dynamic stop and go

## 6/28/24

As of now we have refined model code, a custom Dataset, refined training loops, and refined evaluation loops. These are
tested to an extent, but there are some NaN errors when training for a bit and I am not sure of the source and this warrants
closer investigation and comparison to the original code. We have certain training algorithms implemented (see above)

To Do:
* Figure out why the model gets NaN errors.
* Implement more methods - DSG and loss scaling approaches should be easiest - this is maybe enough for the weekend
* Once we get Original Code and access to data, start running real experiments and fixing errors/ adapting our code as needed.
* Hopefully by end of next week we can have a suite of basic experimental results on finetuning (at very least infrastructure
  so we can run everything - I say this because adapting stuff and bug checking is usually harder than expected)
