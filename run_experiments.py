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
batch_size = 16
sequence_length_text = 500 # these are what are currently used
sequence_length_image = 36 # but model should work without
sequence_length_audio = 63
input_size_image = 1024 # I3D feature size
input_size_audio = 128 # VGGish feature size
embedding_dim = 768

shuffle=True

# copying over some hyperarameters from the original code - these aren't used and don't match paper
# max_epochs = 50
# learning_rate = 1.5e-5
# clip_grad = 0.5
# weight_decay_val = 0
# optimizer_type = 'adam'  # sgd

l2_regularize = True
l2_lambda = 0.1

# Learning rate scheduler
lr_schedule_active = True

def load_pickled_model(pickle_path, heads):
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)
    best_model_weights = results["model"]

    model, _ = initiate_model_new(heads)
    model.load_state_dict(best_model_weights)

    return model


def init_model(heads=multi_task_heads, pretrained=True):
    # first initiate a pretrained (not finetuned model) with the proper multi task heads
    if pretrained:
        model, _ = initiate_pretrained_model(heads)
    else:
        model, _ = initiate_model_new(heads)
    model.to(device)

    return model

def init_single_task_models(heads, pretrained=True):
    models = {name: None for name in heads.keys()}
    for name, head in heads.items():
        if pretrained:
            model, _ = initiate_pretrained_model({name: head}) # model excepts heads in the form of a dictionary
        else:
            model, _ = initiate_pretrained_model({name:head})
        model.to(device)
        models[name] = model
    return models


# pretrained could really be done outside
def run_experiment(model, training_method, loss_setting, tasks, epochs=None, task_epochs=None):
    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)

    if lr_schedule_active:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-8)
    else:
        scheduler = None

    model, loss_history, task_loss_history, validation_results, strategy_results = train_loop(model, optimizer, "train_features_lrec_camera.json", tasks, scheduler=scheduler, json_val_path="val_features_lrec_camera.json", loss_setting=loss_setting, training_method=training_method, batch_size=batch_size, num_epochs=epochs, task_epochs=task_epochs, shuffle=shuffle, device=device)

    return model, loss_history, task_loss_history, validation_results, strategy_results

# strategy results are here too now
def save_and_plot(best_model_weights, loss_history, task_loss_history, validation_results, strategy_results, name):
    try:
        dir_path = f"Results/{name}"
        os.makedirs(dir_path, exist_ok=True)

        pickle_path = f"{dir_path}/{name}.pkl"

        # Save the results to a file
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'model': best_model_weights,
                'loss_history': loss_history,
                'task_loss_history': task_loss_history,
                'validation_results': validation_results,
                'strategy_results': strategy_results
            }, f)

        print(f"Experiment results saved to {pickle_path}")
    except:
        pass

    try:
        plot_results(loss_history, task_loss_history, validation_results, strategy_results, name=name, plot=False, save=True, plot_strategy_results=True)
    except:
        pass

def run_save_experiment(model, training_method, loss_setting, name, tasks, epochs=None, task_epochs=None):
    try:
        model, loss_history, task_loss_history, validation_results, strategy_results = run_experiment(model, training_method, loss_setting, tasks, epochs=epochs, task_epochs=task_epochs)
    except:
        pass
    try:
        save_and_plot(model, loss_history, task_loss_history, validation_results, strategy_results, name=name)
    except:
        pass

def extend_task_history(init_history, new_history):
    history = {task: torch.cat(init, new) for (task, init), new in zip(init_history.items(), new_history.values())}
    return history

# takes in a dictionary of results and extends each of them. Individual items may be tensors or dictionaries of tensors
def extend_mixed_results_dict(init_results, new_results):
    results = {}
    for (metric, init), new in zip(init_results.items(), new_results.values()):
        if type(init) == dict:
            results[metric] = extend_task_history(init, new)
        else:
            results[metric] = torch.cat((init, new), dim=0)
    return results

# currently the steps variable in the second run won't be accurate but this doesn't affect anything
def continue_training(pickle_path, training_method, loss_setting, name, tasks, epochs=None, task_epochs=None):
    # load model and results where we left off
    # using best model is a little misleading and could results in some weird affects in histories, but it should be fine
    # also will this mess up the optimizer and scheduler? I think it might - but worry about this more later
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)
    best_model_weights = results["model"]
    init_loss_history = results["loss_history"]
    init_task_loss_history = results["task_loss_history"]
    init_validation_results = results["validation_results"]
    init_strategy_results = results["strategy_results"]

    # load a random model with the appropriate heads and initialize with the appropriate weights
    model, _ = initiate_model_new(heads)
    model.load_state_dict(best_model_weights)

    # simply run the experiment building off the initial model
    model, loss_history, task_loss_history, validation_results, strategy_results = run_experiment(model, training_method, loss_setting, tasks, epochs=epochs, task_epochs=task_epochs)
    
    # now combine the results
    loss_history = torch.cat((init_loss_history, loss_history), dim=0)
    task_loss_history = extend_task_history(init_task_loss_history, task_loss_history)
    validation_results = extend_mixed_results_dict(init_validation_results, validation_results)
    strategy_results = extend_mixed_results_dict(init_strategy_results, strategy_results)

    # save and plot the final results
    save_and_plot(model, loss_history, task_loss_history, validation_results, strategy_results, name=name)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    epochs = 10

    single_task_models = init_single_task_models(heads=multi_task_heads, pretrained=True)

    # run each single task model
    for name, model in single_task_models.items():
        if name == "gory":
            pass
        if name == "mature":
            pass
        else:
            print(f"Training Single Task Model for {name}")
            try:
                run_save_experiment(model, "all_at_once", "unweighted", f"{name}50", [name], epochs=50) # name should be the same as the task - also expected in array form
            except:
                pass
            # run_save_experiment(model, "all_at_once", "unweighted", f"TEST_{name}", [name], 3)
    

    model = init_model(heads=multi_task_heads, pretrained=True)

    # One At A Time Experiment
    # a better implementation might train til each converges rather than just prespecified epohcs, buit probably fine for now
    # training easiest to hardest based on final accuracy reported in paper (same ordering for single task and multi task)
    # not the most robust way to do it but works fine for a baseline
    task_epochs = {"slapstick": 10, "gory": 10, "sarcasm": 10, "mature": 10}
    try:
        run_save_experiment(model, "one_at_a_time", "unweighted", "One_At_A_Time10Each", multi_tasks, epochs=None, task_epochs=task_epochs)
    except:
        pass

    # Dynamic difficulty sampling - continue more overnight
    try:
        run_save_experiment(model, "dynamic_difficulty_sampling", "unweighted", "DynamicDifficultySampling50", multi_tasks, epochs=50)
    except:
        pass
    #run_save_experiment(model, "dynamic_difficulty_sampling", "unweighted", "TestDynamicDifficultySampling", multi_tasks, epochs=2)

    # Replicate paper's multi task experiment - restart from scratch with 30 epochs overnight
    try:
        run_save_experiment(model, "all_at_once", "predefined_weights", "Recreate50", multi_tasks, epochs=50)
    except:
        pass

    # Unweighted Experiment
    try:
        run_save_experiment(model, "all_at_once", "unweighted", "Unweighted50", multi_tasks, epochs=50)
    except:
        pass

    # Grad Norm - error check and if it seems to woork do it from scratch
    # run_save_experiment(model, "all_at_once", "gradnorm", "GradNorm10", multi_tasks, epochs=2)


    # Dynamic difficulty sampling with Grad Norm
    # run_save_experiment(model, "dynamic_difficulty_sampling", "gradnorm", "DynamicDifficulty_GradNorm10", multi_tasks, 3)

    # for name, model in single_task_models.items():
    #     if name == "gory":
    #         pass
    #     if name == "mature":
    #         pass
    #     else:
    #         print(f"Training Single Task Model for {name}")
    #         continue_training(f"/usuarios/arnold.moralem/IRESComicMischief/Results/{name}/{name}.pkl", "all_at_once", "unweighted", f"{name}_30", [name], 20) 

    # continue dynamic difficulty sampling training
    # continue_training(f"/usuarios/arnold.moralem/IRESComicMischief/Results/{name}/{name}.pkl", "all_at_once", "unweighted", f"{name}_30", [name], 20)

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