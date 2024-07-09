from helpers import initiate_pretrained_model, initiate_model_new, multi_task_heads, multi_tasks
import os
import torch
from torch import optim
from train import train_loop
import pickle
from plot_results import plot_results

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(7)
if torch.cuda. \
        is_available():
    torch.cuda.manual_seed_all(7)
torch.backends.cudnn.enabled = False

# not sure why GPU doesn't work
# device = "cpu"

# Define input shapes 
batch_size = 32
sequence_length_text = 500 # these are what are currently used
sequence_length_image = 36 # but model should work without
sequence_length_audio = 63
input_size_image = 1024 # I3D feature size
input_size_audio = 128 # VGGish feature size
embedding_dim = 768

shuffle=True

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

# pretrained could really be done outside
def run_experiment(training_method, loss_setting, epochs, pretrained=True):
    # first initiate a pretrained (not finetuned model) with the proper multi task heads
    if pretrained:
        model, _ = initiate_pretrained_model(multi_task_heads)
    else:
        model, _ = initiate_model_new(multi_task_heads)
    model.to(device)

    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-8)

    model, loss_history, task_loss_history, validation_results = train_loop(model, optimizer, "train_features_lrec_camera.json", multi_tasks, scheduler=scheduler, json_val_path="val_features_lrec_camera.json", loss_setting=loss_setting, training_method=training_method, batch_size=batch_size, num_epochs=epochs, shuffle=shuffle, device=device)

    return model, loss_history, task_loss_history, validation_results

def save_and_plot(best_model_weights, loss_history, task_loss_history, validation_results, name):
    dir_path = f"Results/{name}"
    os.makedirs(dir_path, exist_ok=True)

    pickle_path = f"{dir_path}/{name}.pkl"

    # Save the results to a file
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'model': best_model_weights,
            'loss_history': loss_history,
            'task_loss_history': task_loss_history,
            'validation_results': validation_results
        }, f)

    print(f"Experiment results saved to {pickle_path}")

    plot_results(loss_history, task_loss_history, validation_results, name=name, plot=False, save=True)

def run_save_experiment(training_method, loss_setting, epochs, name, pretrained=True):
    model, loss_history, task_loss_history, validation_results = run_experiment(training_method, loss_setting, epochs, pretrained=pretrained)
    save_and_plot(model, loss_history, task_loss_history, validation_results, name=name)

if __name__ == "__main__":
    epochs = 10
    
    # Replicate paper's multi task experiment
    #run_save_experiment("all_at_once", "predefined_weights", 1, "Test")

    # Grad Norm
    run_save_experiment("all_at_once", "gradnorm", 10, "GradNorm10")

     # Dynamic difficulty sampling
    run_save_experiment("dynamic_difficulty_sampling", "unweighted", 3, "DynamicDifficultySampling10")

    # Dynamic difficulty sampling with Grad Norm
    run_save_experiment("dynamic_difficulty_sampling", "gradnorm", 3, "DynamicDifficulty_GradNorm10")

    # total loss of original experiment does not use loss weightings used - must perform again
    # filename = 'Results/pickled_results/recreate_multitask_10epoch.pkl'
    # best_model_weights, loss_history, task_loss_history, validation_results = run_experiment(training_method="all_at_once", loss_setting="predefined_weights", epochs=10, pretrained=True)

    # # Save the results to a file
    # with open(filename, 'wb') as f:
    #     pickle.dump({
    #         'model': best_model_weights,
    #         'loss_history': loss_history,
    #         'task_loss_history': task_loss_history,
    #         'validation_results': validation_results
    #     }, f)

    # print(f"Experiment results saved to {filename}")