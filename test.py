import torch
import numpy as np
from Model.UnifiedModel import UnifiedModel
from Model.BaseModel import Bert_Model
from Model.TaskHeads import BinaryClassification, MultiTaskClassification
from train import train, dynamic_difficulty_sampling # are these names an issue
from evaluate import evaluate
import os
import re # needed to load the state dict into the slightly modified model

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(7)
if torch.cuda. \
        is_available():
    torch.cuda.manual_seed_all(7)
torch.backends.cudnn.enabled = False

# Define input shapes 
batch_size = 2
sequence_length_text = 500 # these are what are currently used
sequence_length_image = 36 # but model should work without
sequence_length_audio = 63
input_size_image = 1024 # I3D feature size
input_size_audio = 128 # VGGish feature size
embedding_dim = 768

# Main Two Task Splits to recreate original code
binary_tasks = ["binary"]
multi_tasks = ["mature", "gory", "sarcasm", "slapstick"]

# different head for each task
def initiate_model_new():
    base_model = Bert_Model()
    task_heads = {
        "binary": BinaryClassification(),
        "mature": BinaryClassification(),
        "gory": BinaryClassification(),
        "sarcasm": BinaryClassification(),
        "slapstick": BinaryClassification()
    }
    unified_model = UnifiedModel(base_model, task_heads)
    base_model.to(device)
    unified_model.to(device)
    return unified_model, base_model

# TO DO:
# The loading pretraining code should really be put separately
# Also need to further adapt this to Sukruth's code (same big ideas, but the dictionary may need to be changed a little)
key_mapping = {
    'features.0': 'bert',
    'features.1': 'rnn_img',
    'features.2': 'rnn_img_drop_norm',
    'features.3': 'rnn_audio',
    'features.4': 'rnn_audio_drop_norm',
    'features.5': 'sequential_audio',
    'features.6': 'sequential_image',
    'features.7': 'att1',
    'features.8': 'att1_drop_norm1',
    'features.9': 'att1_drop_norm2',
    'features.10': 'att2',
    'features.11': 'att2_drop_norm1',
    'features.12': 'att2_drop_norm2',
    'features.13': 'att3',
    'features.14': 'att3_drop_norm1',
    'features.15': 'att3_drop_norm2',
    'features.16': 'attention',
    'features.17': 'attention_drop_norm',
    'features.18': 'attention_audio',
    'features.19': 'attention_audio_drop_norm',
    'features.20': 'attention_image',
    'features.21': 'attention_image_drop_norm'
}  

# handles transforming the keys from having a ModuleList (e.g. calling self.features[i])
# to our new more readable version where we call the modules directly
def transform_key(input_key):
    pattern = r'^features\.(\d+)\.(.*)$'
    match = re.match(pattern, input_key)
    if match:
        x = match.group(1)
        y = match.group(2)
        base_key = f'features.{x}'
        if base_key in key_mapping:
            return f'{key_mapping[base_key]}.{y}'
    return input_key

def initiate_pretrained_model(verbose=False):
    base_path = os.path.dirname(os.path.abspath(__file__))
    pretrained_weights_path = os.path.join(base_path, "checkpoint-pretraining/best_pretrain_matching.pth")
    base_model = Bert_Model()
    pretrained_state_dict = torch.load(pretrained_weights_path)
    pretrained_model_state = {transform_key(k): v for k, v in pretrained_state_dict["model_state"].items()}

    if verbose:
        print(pretrained_state_dict.keys())
        print(pretrained_state_dict['epoch']) # this is just a number, I assume the epoch where the best training occurs at
        print(pretrained_state_dict['score']) # this is a number, it is 0.1 ?
        print(pretrained_state_dict['optimizer'].keys()) # State and Param Groups
        print(pretrained_state_dict['optimizer']["state"].keys()) # numbers 0 to 335
        print(pretrained_state_dict['optimizer']["state"][0])
        print(pretrained_state_dict['optimizer']["param_groups"]) # contains learning rate, betas, eps, weight decay, some other hyprparameters, and params (which is just numbers 0 to 335)
        print()

        # print the parameters and their shapes saved in the pretrained parameters
        # the model has extra task specific heads
        # simply set strict=False to deal with (this would also work if we had new task specific heads in our model - but some names might be different if we were to used UnifiedModel directly)
        for param_tensor in pretrained_state_dict["model_state"]:
            print(f"{param_tensor}  {pretrained_state_dict["model_state"][param_tensor].size()}") # we need to replace things like features.0 with bert - use regex
        print()

        # print the parameters and their shapes in our new base model
        for param_tensor in base_model.state_dict():
            print(f"{param_tensor}  {base_model.state_dict()[param_tensor].size()}")
        print()

        # print the converted parameters in the pretrained parameters - should work in our model now (and will have some extra parameters)
        for param_tensor in pretrained_model_state:
            print(f"{param_tensor}  {pretrained_model_state[param_tensor].size()}") # we need to replace things like features.0 with bert - use regex

    # Load the state dictionary into the model with strict=False
    missing_keys, unexpected_keys = base_model.load_state_dict(pretrained_model_state, strict=False)
    
    # Print missing and unexpected keys for verification
    if verbose:
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        print(f"Successfully loaded base model weights from {pretrained_weights_path}")

    task_heads = {
        "binary": BinaryClassification(),
        "mature": BinaryClassification(),
        "gory": BinaryClassification(),
        "sarcasm": BinaryClassification(),
        "slapstick": BinaryClassification()
    }
    unified_model = UnifiedModel(base_model, task_heads)
    base_model.to(device)
    unified_model.to(device)
    return unified_model, base_model


def basic_forward_pass(unified_model):
    # Create random inputs and move them to the appropriate device
    # Sentences is going to be BERT tokenized sentences 
    text_tokens = torch.randint(0, 30522, (batch_size, sequence_length_text)).to(device)  # BERT vocab size is 30522 for 'bert-base-uncased' -> each number corresponds to a token
    text_mask = torch.randint(0, 2, (batch_size, sequence_length_text)).float().to(device) # 0 or 1 for size (batch_size, sequence_length_txt) -> determines which tokens are valid

    # Image is really going to be I3D video embeddings
    image = torch.randn(batch_size, sequence_length_image, input_size_image).to(device)
    image_mask = torch.randint(0, 2, (batch_size, sequence_length_image)).float().to(device)

    # Audio is going to be VGGish embeddings
    audio = torch.randn(batch_size, sequence_length_audio, input_size_audio).to(device)
    audio_mask = torch.randint(0, 2, (batch_size, sequence_length_audio)).float().to(device) 

    # Forward pass
    binary_output = unified_model(text_tokens, text_mask, image, image_mask, audio, audio_mask, tasks=binary_tasks)
    # multi_output = unified_model(text_tokens, text_mask, image, image_mask, audio, audio_mask, tasks=multi_tasks)

    # Print the output shape
    print("Binary Output shape:", binary_output.shape) # batch size by 2 (one for each prediction ?, why not by 1)
    # print("Multi Output shape:", multi_output.shape) # batch size by 4 by 2 (4 for 4 tasks and 2 for each task)

    print(binary_output[0]) 
    # print(multi_output[0])

def basic_train_pass(model, device, tasks, training_method="all_at_once", loss_setting="unweighted"):
    # just see if it actually runs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, "train_features_lrec_camera.json", tasks, loss_setting=loss_setting, training_method=training_method, batch_size=batch_size, num_epochs=1, shuffle=False, device=device)

def basic_eval_pass(model, device, tasks):
    evaluate(model, "test_features_lrec_camera.json", tasks, batch_size=batch_size, shuffle=False, device=device)

def dynamic_difficulty_sample_test(model, device, tasks):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001).to(device)
    dynamic_difficulty_sampling(model, optimizer, "train_features_lrec_camera.json", tasks, loss_setting="unweighted", batch_size=32, num_epochs=1, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=device)

# GPU ISSUES
# The GPU runs out of memory with batch size greater th(despite it working on CPU)
# Also, after a view runs you get NaN errors -> the GPU doesn't report this clearly, but I think it's the same error

if __name__ == "__main__":

    print(torch.cuda.device_count())  # Number of available GPUs
    print(torch.cuda.current_device())  # Current GPU device index
    print(torch.cuda.get_device_name(0))  # Name of the GPU
    print(f'Allocated: {torch.cuda.memory_allocated() / 1024**2} MB')
    print(f'Cached: {torch.cuda.memory_reserved() / 1024**2} MB')
    torch.cuda.empty_cache()
    # non_pretrained_model, _ = initiate_model_new()
    model, _ = initiate_pretrained_model()


    # basic_forward_pass(model) # seems to work, including on GPU
    # basic_train_pass(model, device, binary_tasks) # loss on order of 500 when beginning because of regularization (like original model)
    basic_train_pass(model, device, multi_tasks, loss_setting="unweighted") # loss around 2.5 when you just add loss for each task
    # basic_train_pass(model, device, multi_tasks, loss_setting="predefined_weights") # multi task setting in paper - loss less than one beginning (no regularization)
    # basic_train_pass(model, device, multi_tasks, training_method="round_robin")
    # basic_eval_pass(model, device, binary_tasks)
    # basic_eval_pass(model, device, multi_tasks)
    # dynamic_difficulty_sample_test(model, device, multi_tasks) # loss starts around 2.5, same as unweighted as it should be
    # print(model.base_model.named_parameters)
    # basic_train_pass(model, device, multi_tasks, loss_setting="gradnorm")