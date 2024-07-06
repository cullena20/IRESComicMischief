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

## 7/1/24

Read and implemeted GradNorm. The reading was helpful to get a better sense of the difficulties and considerations in multi task learning. The coding was good exercise with manually handling somewhat complicated optimization (tracking gradients and the computation graphs, ensuring no conflicts, etc.) and also with figuring out how to better design and modularize code. Everything seems to work, but it is worth double checking.

* Read and implemented GradNorm. I modularized how GradNorm is done so it only returns weights for each loss function, and other stuff is done outside.
* Possible Issues: This modularization could lead to issues due to the mix between how the normal training loss is handled and the grad norm loss. Be careful with this general process (maybe need to detach things, etc.)
* Cleaning Up: How the optimizer and model layer (the layer is used to calculate gradients for grad norm) is not so nice right now and can be made nicer. In general things in the training process can be modularized better
* Still have NaN errors after some rounds of training !

To Do Otherwise:
* Test code with actual I3D and VGG features and figure out how to use code with GPU
* Fix NaN errors if they persist on real data
* Implement PCGrad
* Further clean up training code as I realize our demands
* Keep brainstorming new possible ideas

## 7/5/24

Spent some time this week dealing with debugging and getting code running on the GPU. I figured out the
source of NaN issues (misalligned text padding and mask). This took much longer than it should have but
was a good lesson in debugging and helped me get a better understanding of what is going on in the 
model. I also added and tested a bunch of validation code and incorporated this into the training loop.
Now we should be able to run experiments and return what we need!

* Sorted out NaN issue and verified with proper GPU configs things work. The model works now!
* Added extensive validation code to return more valuable information.
* Incorporated validation code into training loop - training loop now returns a lot of valuable information
* After some cleaning, we are now ready to run extensive experiments.

To Do:
* Log into Arnold's GPU cluster and run experiments. Reach out to Elaheh about the UH cluster.
* Build further experimental infrastructure - add matplotlib plotting functions to plot task loss history, total loss history, validation stuff, etc.. Take the time to have code for pickling and unpickling!
* By Monday I should have main experiment replicated with plots using this. It shouldn't be that
hard to additionally run some other experiments I have (GradNorm and some baselines particularly)
* Spend some time cleaning up and documenting code. This will help during the week as I wish to run
more experiments.
* With code cleaned up, implement PCGrad, run more experiments, work on presentation!

Also note that I have checked all the code I have run so far so things look good. But I should be really careful that everything is indeed running as expected.