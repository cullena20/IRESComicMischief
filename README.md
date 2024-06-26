# Comic Mischief Prediction

IRES Project to explore finetuning and pretraining methods for Multimodal Comic Mischief Prediction, particularly from the perspective of curriculum learning.

# Current Code Outline

BaseModel: The main model. Takes in BERT tokenized text, I3D video embeddings, VGG audio embeddings and outputs concatenated embeddings (768 * 3)
TaskHeads: Task specific heads. Takes in output of base model (768 * 3) and produces some output. There are binary and multi task heads currently.
UnifiedModel: Wrapper to unify the base model with task specific heads. Has some demo code to show model works.

finetuning_dataloader: A data loader for the comic mischief data used for fine tuning
train: Training code for the model. Unified training loop for binary and multi task prediction

# To Do

* There are probably issues in this code and train sometimes crashes, I also have not tested it using actual I3D and VGG embeddings yet.
* Need to define and incorporate evaluation scripts and maybe some additional meta training loop stuff as in the original code (missing learning rate scheduler e.g.)
* Once we're sure all of this works, begin incorporating dynamic training strategies (possibly requiring code refactoring and redesign)