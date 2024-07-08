import torch
import torch.nn as nn
import numpy as np
from Model.UnifiedModel import UnifiedModel
from Model.BaseModel import Bert_Model
from Model.TaskHeads import BinaryClassification, MultiTaskClassification
from train import train, dynamic_difficulty_sampling, train_loop 
from helpers import initiate_pretrained_model
from evaluate import evaluate
import os
import re # needed to load the state dict into the slightly modified model
import psutil
from transformers import AdamW


os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(7)
if torch.cuda. \
        is_available():
    torch.cuda.manual_seed_all(7)
torch.backends.cudnn.enabled = False

# not sure why GPU doesn't work
device = "cpu"

## TEMPORARY
# Function to get the memory usage in MB
def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 ** 2  # Convert from bytes to MB


# Define input shapes 
batch_size = 16
sequence_length_text = 500 # these are what are currently used
sequence_length_image = 36 # but model should work without
sequence_length_audio = 63
input_size_image = 1024 # I3D feature size
input_size_audio = 128 # VGGish feature size
embedding_dim = 768

# Main Two Task Splits to recreate original code
binary_tasks = ["binary"]
multi_tasks = ["mature", "gory", "sarcasm", "slapstick"]

multi_task_heads = {
        "mature": BinaryClassification(),
        "gory": BinaryClassification(),
        "sarcasm": BinaryClassification(),
        "slapstick": BinaryClassification()
    }

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
    accuracies, f1_scores, average_accuracy, average_f1_score, val_average_total_loss, val_average_task_loss, all_labels, all_true_labels = evaluate(model, "test_features_lrec_camera.json", tasks, batch_size=batch_size, shuffle=False, device=device)
    for task in tasks:
        print(f"Task {task}")
        print(f"Number of items: {len(all_labels[task])}")
        print(f"Predictions: {all_labels[task]}")
        print(f"True: {all_true_labels[task]}")
        print(f"Accuracy: {accuracies[task]}, F1 Score: {f1_scores[task]:.4f}")
        print(f"Average Task Loss {val_average_task_loss[task]}")
        print()
    
    print(f"Average Total Loss {val_average_total_loss}")
    print(f"Average Accuracy: {average_accuracy}")
    print(f"Average F1 Score: {average_f1_score}")

def dynamic_difficulty_sample_test(model, device, tasks):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001).to(device)
    dynamic_difficulty_sampling(model, optimizer, "train_features_lrec_camera.json", tasks, loss_setting="unweighted", batch_size=32, num_epochs=1, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=device)

# REFACTORED TRAINING PASS
# also with optimizer and scheduler
def basic_updated_train_pass(model, device, tasks, training_method="all_at_once", loss_setting="unweighted"):
    optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-8)
    model, loss_history, task_loss_history, validation_results = train_loop(model, optimizer, "train_features_lrec_camera.json", tasks, scheduler=scheduler, loss_setting=loss_setting, training_method=training_method, batch_size=batch_size, num_epochs=1, shuffle=False, device=device)
    print(f"Loss History {loss_history}")
    print(f"Task Loss History {task_loss_history}")
    # print(f"Validation Results: {validation_results}")


# GPU ISSUES
# The GPU runs out of memory with batch size greater th(despite it working on CPU)
# Also, after a view runs you get NaN errors -> the GPU doesn't report this clearly, but I think it's the same error

if __name__ == "__main__":
    if device == "cuda":
        print(torch.cuda.device_count())  # Number of available GPUs
        print(torch.cuda.current_device())  # Current GPU device index
        print(torch.cuda.get_device_name(0))  # Name of the GPU
        print(f'Allocated: {torch.cuda.memory_allocated() / 1024**2} MB')
        print(f'Cached: {torch.cuda.memory_reserved() / 1024**2} MB')
        torch.cuda.empty_cache()

    # add if statement depending on whether directory has file
    model, _ = initiate_model_new()
    # model, _ = initiate_pretrained_model(multi_task_heads) cA6587!@
    model.to(device)

    # basic_updated_train_pass(model, device, multi_tasks, loss_setting="predefined_weights")
    # basic_updated_train_pass(model, device, multi_tasks, loss_setting="gradnorm") # this is a good amount slower, maybe slower than it should be?
    # basic_updated_train_pass(model, device, multi_tasks, training_method="dynamic_difficulty_sampling")
    basic_updated_train_pass(model, device, multi_tasks, training_method="dynamic_difficulty_sampling", loss_setting="gradnorm")

    # basic_forward_pass(model) # seems to work, including on GPU
    # basic_train_pass(model, device, binary_tasks) # loss on order of 500 when beginning because of regularization (like original model)
    # basic_train_pass(model, device, multi_tasks, loss_setting="unweighted") # loss around 2.5 when you just add loss for each task
    # basic_train_pass(model, device, multi_tasks, loss_setting="predefined_weights") # multi task setting in paper - loss less than one beginning (no regularization)
    # basic_train_pass(model, device, multi_tasks, training_method="round_robin")
    # basic_eval_pass(model, device, binary_tasks)
    # basic_eval_pass(model, device, multi_tasks)
    # dynamic_difficulty_sample_test(model, device, multi_tasks) # loss starts around 2.5, same as unweighted as it should be
    # print(model.base_model.named_parameters)
    # basic_train_pass(model, device, multi_tasks, loss_setting="gradnorm")
