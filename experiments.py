from helpers import initiate_pretrained_model, initiate_model_new, multi_task_heads, multi_tasks
import os
import torch
from torch import optim
from train import train
from transformers import AdamW, get_linear_schedule_with_warmup
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(7)
if torch.cuda. \
        is_available():
    torch.cuda.manual_seed_all(7)
torch.backends.cudnn.enabled = False

# not sure why GPU doesn't work
device = "cpu"

# Define input shapes 
batch_size = 16
sequence_length_text = 500 # these are what are currently used
sequence_length_image = 36 # but model should work without
sequence_length_audio = 63
input_size_image = 1024 # I3D feature size
input_size_audio = 128 # VGGish feature size
embedding_dim = 768

# copying over some hyperarameters from the original code
max_epochs = 50
learning_rate = 1.5e-5
clip_grad = 0.5
weight_decay_val = 0
optimizer_type = 'adam'  # sgd

l2_regularize = True
l2_lambda = 0.1

# Learning rate scheduler
lr_schedule_active = False
reduce_on_plateau_lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau


def recreate_multitask_experiment(epochs=1, pretrained=True):
    # first initiate a pretrained (not finetuned model) with the proper multi task heads
    if pretrained:
        model, _ = initiate_pretrained_model(multi_task_heads)
    else:
        model, _ = initiate_model_new(multi_task_heads)
    model.to(device)

    # AdamW optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)

    # using scheduler described in the paper (variables described in paper -> code has different variables)
    # we need val data to use the below, if it is not provided it is equivalent to no scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-8)

    model, loss_history, task_loss_history, validation_results = train(model, optimizer, "train_features_lrec_camera.json", multi_tasks, scheduler=scheduler, json_val_path="val_features_lrec_camera.json", loss_setting="predefined_weights", training_method="all_at_once", batch_size=batch_size, num_epochs=epochs, shuffle=False, device=device)

    return model, loss_history, task_loss_history, validation_results


if __name__ == "__main__":
    # total loss of original experiment does not use loss weightings used - must perform again
    filename = 'recreate_multitask_1epoch.pkl'
    model, loss_history, task_loss_history, validation_results = recreate_multitask_experiment(pretrained=False)
    print(f"Loss History {loss_history}")
    print(f"Task Loss History {task_loss_history}")
    print(f"Validation Results: {validation_results}")

    # Save the results to a file
    with open(filename, 'wb') as f:
        pickle.dump({
            'model': model,
            'loss_history': loss_history,
            'task_loss_history': task_loss_history,
            'validation_results': validation_results
        }, f)

    print(f"Experiment results saved to {filename}")