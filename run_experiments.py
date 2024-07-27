from helpers import initiate_pretrained_model, initiate_model_new, multi_task_heads, multi_tasks
import os
import torch
from torch import optim
from train import train_loop
import pickle
from plot_results import plot_results
from evaluate import evaluate
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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
    best_model_state_dict = results["model_state_dict"]

    model, _ = initiate_model_new(heads)
    model.load_state_dict(best_model_state_dict)

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
def run_experiment(model, training_method, loss_setting, tasks, epochs=None, task_epochs=None, name="Test"):
    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)

    if lr_schedule_active:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-8)
    else:
        scheduler = None

    best_model_state_dict, loss_history, task_loss_history, validation_results, strategy_results, optimizer_state_dict, scheduler_state_dict = train_loop(model, optimizer, "train_features_lrec_camera.json", tasks, scheduler=scheduler, json_val_path="val_features_lrec_camera.json", loss_setting=loss_setting, training_method=training_method, batch_size=batch_size, num_epochs=epochs, task_epochs=task_epochs, shuffle=shuffle, device=device, name=name)

    return best_model_state_dict, loss_history, task_loss_history, validation_results, strategy_results, optimizer_state_dict, scheduler_state_dict

# Task epochs is not handled and will just save None as epoch

# strategy results are here too now
def save_and_plot(best_model_state_dict, loss_history, task_loss_history, validation_results, strategy_results, name, optimizer_state_dict, scheduler_state_dict, epoch=None):
    try:
        dir_path = f"Results/{name}"
        os.makedirs(dir_path, exist_ok=True)

        pickle_path = f"{dir_path}/{name}.pkl"

        # Save the results to a file
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'epoch' : epoch,
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'scheduler_state_dict': scheduler_state_dict,
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

def test_model(model, tasks, best_model_state_dict=None, results_pickle_path=None, loss_weights=None, save_dir=None):    
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    elif results_pickle_path is not None:
        with open(results_pickle_path, "rb") as f:
            results = pickle.load(f)
        best_model_state_dict = results["model_state_dict"] # for newly formated pickle files
        # best_model_state_dict = results["model"] # for old pickle files
        model.load_state_dict(best_model_state_dict)

        # BELOW IS JANK
        if loss_weights is None:
            loss_weight_history = results["strategy_results"]["loss_weight_history"]
            final_loss_weights = {task: loss_weights[-1] for task, loss_weights in loss_weight_history.items()} # i think this should work, might have format wrong
            final_loss_weights = [weight for weight in final_loss_weights.values()] # this is jank but should work fine if we do things in the right order
    else:
        model = init_model(heads=tasks, pretrained=True)
    print("Model Loaded")

    # each of these is just a single value (except for the labels) unlike the validation results
    accuracies, f1_scores, average_accuracy, average_f1_score, val_average_total_loss, val_average_task_loss, all_labels, all_true_labels = evaluate(model, "test_features_lrec_camera.json", tasks, loss_weights)

    # Structure the results in a single dictionary
    results_dict = {
        'accuracies': accuracies,
        'f1_scores': f1_scores,
        'average_accuracy': average_accuracy,
        'average_f1_score': average_f1_score,
        'val_average_total_loss': val_average_total_loss,
        'val_average_task_loss': val_average_task_loss
    }

    print(results_dict)
    
    # Save the results dictionary to a JSON file
    if save_dir is not None:
        output_path = f"{save_dir}/test_results.json"
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

    return accuracies, f1_scores, average_accuracy, average_f1_score, val_average_total_loss, val_average_task_loss, all_labels, all_true_labels

def run_save_experiment(model, training_method, loss_setting, name, tasks, epochs=None, task_epochs=None):
    best_model_state_dict, loss_history, task_loss_history, validation_results, strategy_results, optimizer_state_dict, scheduler_state_dict = run_experiment(model, training_method, loss_setting, tasks, epochs=epochs, task_epochs=task_epochs, name=name)
    save_and_plot(best_model_state_dict, loss_history, task_loss_history, validation_results, strategy_results, name=name, optimizer_state_dict=optimizer_state_dict, scheduler_state_dict=scheduler_state_dict, epoch=epochs)

    # try:
    #     best_model_state_dict, loss_history, task_loss_history, validation_results, strategy_results, optimizer_state_dict, scheduler_state_dict = run_experiment(model, training_method, loss_setting, tasks, epochs=epochs, task_epochs=task_epochs, name=name)
    # except Exception as e:
    #     print(f"Running Experiment Failed with Following Exception: {e}")

    # try:
    #     save_and_plot(best_model_state_dict, loss_history, task_loss_history, validation_results, strategy_results, name=name, optimizer_state_dict=optimizer_state_dict, scheduler_state_dict=scheduler_state_dict)
    # except Exception as e:
    #     print(f"Saving and Plotting Experiment Failed with Following Exception: {e}")
    
    # this is pretty jank
    loss_weight_history = strategy_results["loss_weight_history"]
    final_loss_weights = {task: loss_weights[-1] for task, loss_weights in loss_weight_history.items()} # i think this should work, might have format wrong
    final_loss_weights = [weight for weight in final_loss_weights.values()]

    test_results = test_model(model, tasks, best_model_state_dict, loss_weights=final_loss_weights, save_dir=f"Results/{name}")


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

# ADJUST THIS NOW THAT WE PICK UP THE SCHEDULER AND OPTIMIZER
# also how we handle the loss history and what not - currently we can save these checkpoints as lists so we can resume, but this isn't the most flexible
# currently the steps variable in the second run won't be accurate but this doesn't affect anything
def continue_training(pickle_path, training_method, loss_setting, name, tasks, epochs=None, task_epochs=None):
    # load model and results where we left off
    # using best model is a little misleading and could results in some weird affects in histories, but it should be fine
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)
    model_state_dict = results["model_state_dict"]
    optimizer_state_dict = results['optimizer_state_dict']
    scheduler_state_dict = results['scheduler_state_dict']
    init_loss_history = results["loss_history"]
    init_task_loss_history = results["task_loss_history"]
    init_validation_results = results["validation_results"]
    init_strategy_results = results["strategy_results"]

    # load a random model with the appropriate heads and initialize with the appropriate weights
    model, _ = initiate_model_new(tasks)
    model.load_state_dict(model_state_dict)

    # simply run the experiment building off the initial model
    model, loss_history, task_loss_history, validation_results, strategy_results, optimizer_state_dict, scheduler_state_dict = run_experiment(model, training_method, loss_setting, tasks, epochs=epochs, task_epochs=task_epochs)
    
    # now combine the results
    loss_history = torch.cat((init_loss_history, loss_history), dim=0)
    task_loss_history = extend_task_history(init_task_loss_history, task_loss_history)
    validation_results = extend_mixed_results_dict(init_validation_results, validation_results)
    strategy_results = extend_mixed_results_dict(init_strategy_results, strategy_results)

    # save and plot the final results
    save_and_plot(model, loss_history, task_loss_history, validation_results, strategy_results, name=name)

# i don't think the try except loops here are helpful because we have try and except loops elsewhere
if __name__ == "__main__":

    # testing models that have already been trained before updating run_save_experiment function
    single_task_models = init_single_task_models(heads=multi_task_heads, pretrained=True)

    # test_model(model, multi_task_heads, save_dir="Results/")
    # Single Task Models
    # for task, single_task_model in single_task_models.items():
    #     test_model(single_task_model, [task], results_pickle_path=f"Results/{task}0/{task}.pkl", save_dir=f"Results/{task}10")

    # # Original Experiment (predefined weights) - pretrained and not
    # test_model(model, multi_task_heads, results_pickle_path="Results/Recreate10-Part2/Recreate10-Part2.pkl", save_dir="Results/Recreate10-Part2")
    # test_model(model, multi_task_heads, results_pickle_path="Results/NotPretrainedRecreate10/NotPretrainedRecreate10.pkl", save_dir="Results/NotPretrainedRecreate10")

    # # One at a time
    # test_model(model, multi_task_heads, results_pickle_path="Results/One_At_A_Time4Each/One_At_A_Time4Each.pkl", save_dir="Results/One_At_A_Time4Each")

    # # Unweighted adding loss functions pretrained
    # test_model(model, multi_task_heads, results_pickle_path="Results/Unweighted10/Unweighted10.pkl", save_dir="Results/Unweighted10")

    torch.autograd.set_detect_anomaly(True)

    # epochs = 10

    # single_task_models = init_single_task_models(heads=multi_task_heads, pretrained=True)

    model = init_model(heads=multi_task_heads, pretrained=True)

    # Grad Norm - error check and if it seems to woork do it from scratch
    #run_save_experiment(model, "all_at_once", "gradnorm", "GradNormTest", multi_tasks, epochs=10)

    not_pretrained = init_model(heads=multi_task_heads, pretrained=False)

    # try:
    # print("Naive Learnable Weights")
    # run_save_experiment(model, "all_at_once", "naive_learnable", "NaiveLearnableWeights5", multi_tasks, epochs=5)

    # run each single task model
    # for task, model in single_task_models.items():
    #     print(f"Training Single Task Model for {task}")
    #     try:
    #         # run_save_experiment(model, "all_at_once", "unweighted", f"TEST2_{name}", [name], 1)
    #         run_save_experiment(model, "all_at_once", "unweighted", f"{task}30", [task], epochs=30) # name should be the same as the task - also expected in array form
    #     except Exception as e:
    #         print(f"Single Task on {task} failed with Exception: {e}")
    
    # except Exception as e:
    #     print(f"Naive Learnable Weights Crashed {e}")
    #     pass 

    # run_save_experiment(model, "dynamic_difficulty_sampling", "unweighted", "TestDynamicDifficultySampling", multi_tasks, epochs=2)

    # One At A Time Experiment
    # a better implementation might train til each converges rather than just prespecified epohcs, buit probably fine for now
    # training easiest to hardest based on final accuracy reported in paper (same ordering for single task and multi task)
    # not the most robust way to do it but works fine for a baseline
    # task_epochs = {"slapstick": 10, "gory": 10, "sarcasm": 10, "mature": 10}
    # try:
    #     print("Running One At At A Time")
    #     run_save_experiment(model, "one_at_a_time", "unweighted", "One_At_A_Time10Each", multi_tasks, epochs=None, task_epochs=task_epochs)
    # except Exception as e:
    #     print(f"One At At A Time Crashed with Exception: {e}")
    # run above again, crashed


    # Replicate paper's multi task experiment - restart from scratch with 30 epochs overnight
    try:
        print("Pretrained Recreate 30")
        # run_save_experiment(model, "all_at_once", "predefined_weights", "Recreate50", multi_tasks, epochs=50)
        run_save_experiment(model, "all_at_once", "predefined_weights", "Recreate30", multi_tasks, epochs=30)
    except Exception as e:
        print(f"Pretrained Recreate Crashed {e}")
        pass

    #Dynamic difficulty sampling - crashed at 5 epochs! What happened?? 
    try:
        print("Running Dynamic Difficulty Sampling")
        run_save_experiment(model, "dynamic_difficulty_sampling", "unweighted", "DynamicDifficultySampling30", multi_tasks, epochs=30)
    except Exception as e:
        print(f"Dynamic Difficulty Sampling Crashed with Exception: {e}")

    # run below after for some error checking and to compare to different stuff 

    # try:
    #     print("Pretrained Unweighted 30")
    #     # run_save_experiment(model, "all_at_once", "predefined_weights", "Recreate50", multi_tasks, epochs=50)
    #     run_save_experiment(model, "all_at_once", "unweighted", "MultiUnweighted30", multi_tasks, epochs=30)
    # except Exception as e:
    #     print(f"Pretrained Unweighted Crashed {e}")
    #     pass

    # try:
    #     print("Running Not Pretrained Recreate 30")
    #     run_save_experiment(not_pretrained, "all_at_once", "predefined_weights", "NotPretrainedRecreate30", multi_tasks, epochs=10)
    # except Exception as e:
    #     print(f"Not Pretrained Recreate Crashed {e}")
    #     pass


    # Dynamic difficulty sampling - continue more overnight
    # try:
    #     print("Running Dynamic Difficulty Sampling with Predefined Weights")
    #     run_save_experiment(model, "dynamic_difficulty_sampling", "predefined_weights", "PredefinedDynamicDifficultySampling5", multi_tasks, epochs=5)
    # except Exception as e:
    #     print(f"Dynamic Difficulty Sampling Crashed {e}")
    #     pass

    # Unweighted Experiment
    # try:
    #     run_save_experiment(model, "all_at_once", "unweighted", "Unweighted10", multi_tasks, epochs=10)
    # except:
    #     pass


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