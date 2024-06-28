import torch
import numpy as np
from Model.UnifiedModel import UnifiedModel
from Model.BaseModel import Bert_Model
from Model.TaskHeads import BinaryClassification, MultiTaskClassification
from train import train # are these names an issue
from evaluate import evaluate

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define input shapes 
batch_size = 8
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
    multi_output = unified_model(text_tokens, text_mask, image, image_mask, audio, audio_mask, tasks=multi_tasks)

    # Print the output shape
    print("Binary Output shape:", binary_output.shape) # batch size by 2 (one for each prediction ?, why not by 1)
    print("Multi Output shape:", multi_output.shape) # batch size by 4 by 2 (4 for 4 tasks and 2 for each task)

    print(binary_output[0]) 
    print(multi_output[0])

def basic_train_pass(model, device, tasks, training_method):
    # just see if it actually runs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, "train_features_lrec_camera.json", tasks, training_method=training_method, batch_size=batch_size, num_epochs=1, shuffle=False, device=device)

def basic_eval_pass(model, device, task):
    evaluate(model, "test_features_lrec_camera.json", task, batch_size=batch_size, shuffle=False, device=device)

if __name__ == "__main__":
    model, _ = initiate_model_new()
    # basic_forward_pass(model)
    # basic_train_pass(model, device, binary_tasks) # loss on order of 500 when beginning (like original model)
    # basic_train_pass(model, device, multi_tasks) # loss now on order of 500 (original model was like 0.8)
    # basic_train_pass(model, device, multi_tasks, training_method="round_robin")
    # basic_eval_pass(model, device, binary_tasks)
    basic_eval_pass(model, device, multi_tasks)