# Comic Mischief Prediction (WIP)

IRES Project to explore finetuning and pretraining methods for Multimodal Comic Mischief Prediction, particularly from the perspective of curriculum learning. This is an early stage project in progress right now.

## Usage Note

The way directories are handled is a little specific unfortunately. Here is how you properly use it.

* processed_data: contains json files for test, train, and val features. These are BERT tokenized text with some extra information.
* i3d-vggish-features: i3d_vecs_extended_merged should house video features. Each file in it will be an individual i3d embedding. vgg_vecs_extended_merged should house audio features. Again, each file in it will be an individual vgg embedding. These folders are empty because they are too large to push to GitHub.
    * For now, copy the contents of this folder in the Joshi server directly here (or easier, copy the folder itself which has this name)
    * In the future, we can upload these to Google Drive so that they are more accessible.
* checkpoint-pretraining: This should contain a file titled "best_pretrain_matching.pth" which contains the pretrained model weights. Again, this file is too large to push to Github.
    * For now, copy this file from the Joshi server here directly. Again, we can upload this to a Google Drive in the future.

# Current Code Outline

* BaseModel: The main model. Takes in BERT tokenized text, I3D video embeddings, VGG audio embeddings and outputs concatenated embeddings (768 * 3)
* TaskHeads: Task specific heads. Takes in output of base model (768 * 3) and produces some output. There are binary and multi task heads currently. We only really need the BinaryClassification heads as of now.
* UnifiedModel: Wrapper to unify the base model with task specific heads. Has some demo code to show model works.
* finetuning_dataloader: A data loader for the comic mischief data used for fine tuning. Relies on the data being organized as described in the Usage Notes.
* train: Training code for the model. Unified training loop for binary and multi task prediction. Also has subroutines to handle different training methods. This is the bulk of where we are coding, but we might want to further modularize (e.g. gradnorm is in its own file)
* evaluate: Evaluation code for the model, unified for binary and multi task prediction. This has not been extensively tested and should probably return more information.
* helpers: contains some functions used for regularization, padding, and masking. Could add more helper functions here as needed.

There is probably some better organization and further modularization that we may wish to do on this code.