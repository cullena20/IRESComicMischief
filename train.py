import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
from finetuning_dataloader import CustomDataset
from gradnorm import gradnorm
import math

from helpers import compute_l2_reg_val

import time

# TO DO:
# this should be better modularized
# I don't like that the scheduler is deifned inside the model like this
# also should scheduler step at every batch or every epoch
# also how validation results are handled is terrible right now

# LOSS WEIGHTS ARE NOT CURRENLY USED IN OPTIMIZATION EXCEPT FOR GRADNORM

# this should really be elsewhere
k=15

debug = False

def dprint(text):
    if debug:
        print(text)

# Shapes and everything are working here, but not sure if this is actually working.

# I also got an issue on one batch, so there are some datapoints with issues
# not all values were 0 or 1 - some were nan
# the other time same issue but it had an error that we are backpassing twice

# currently this has no  lr_scheduler, or checkpoint modeling which the original does
# also does not have evaluation right now

# where did these come from? supposed to be learnable
mature_w = 0.1
gory_w = 0.4
slap_w = 0.2
sarcasm_w = 0.2

# NOTE - THINGS IMPLEMENTED:
# ROUND ROBIN NAIVE IMPLEMENTED (incorporate DSG)
# LOSS REWEIGHTING WITH LOSSES AS PARAMETER IMPLEMENTED
# CAN DO SINGLE TASK BY TRAINING ONE HEAD AT A TIME IMPLEMENTED

# TO DO Implementations
# DSG
# Dynamic Pretraining
# etc.

# To Do :
# Training Loops should return loss history (averaged over epochs)
# Further modularize training loops

# input list of tasks
# ["binary", "mature", "gory", "sarcasm", "slapstick"]

def train_loop(model, optimizer, json_train_path, tasks, scheduler=None, json_val_path=None, training_method="all_at_once", loss_setting="unweighted", batch_size=16, num_epochs=1, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # the dataset has x values: 
    # text, text_mask, image, image_mask, audio, audio_maxk
    # and y value (deal with depending on task)
    if training_method == "dynamic_difficulty_sampling":
        datasets = {} # store the dataset for every task
        sample_weights = {} # controls weight of each dataset in the batch
        for task in tasks:
            datasets[task] = CustomDataset(json_train_path, text_pad_length, img_pad_length, audio_pad_length)
            # TO DO Probably want to implement shuffling here, or implement it as a method of the dataset
            sample_weights[task] = 1/len(tasks) # initialize so all tasks are weighted the same
        total_steps = math.ceil(len(datasets[tasks[0]]) / batch_size) * len(tasks) * num_epochs # weird and asumes dataset same size for everything
        # the times 4 is because we really need to go through every data point from every task
        # this needs some thinking about and can be really optimized
    else:
        dataset = CustomDataset(json_train_path, text_pad_length, img_pad_length, audio_pad_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 
        total_steps = len(dataloader) * num_epochs # Total number of training steps (epochs * batches)
    
    print("Created Datasets")
    print(f"Total Steps {total_steps}")

    # alternate scheduler below - probably no need for
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=0,
    #                                             num_training_steps=total_steps,
    #                                             last_epoch=-1)

    loss_history = torch.zeros(total_steps) # track the total loss at every step
    task_loss_history = {task : torch.zeros(total_steps) for task in tasks} # track the loss for every task

    validation_results = {
        "accuracies": {},
        "f1_scores": {},
        "average_accuracy": [],
        "average_f1_score": [],
        "val_average_total_loss": [],
        "val_average_task_loss": {}
    }

    # Initialize loss weights as vector of all ones before training
    loss_weights = torch.ones(len(tasks))
    loss_weights = nn.Parameter(loss_weights)
    dprint(f"Initial Loss Weights {loss_weights}")

    steps = 0 # a step is one batch aka one training step

    best_f1 = 0
    best_model_weights = None

    initial_task_losses, gradnorm_optimizer = None, None
    for epoch in range(num_epochs):
        # first train for an epoch
        # currently returns the updated loss histories and model is trained in place (but maybe the losses are too)
        # I don't like how we need all these extra variables just for gradnorm -> can be solved using a class that saves these values
        
        if training_method == "dynamic_difficulty_sampling":
            loss_history, task_loss_history, steps, loss_weights, initial_task_losses, gradnorm_optimizer = dynamic_difficulty_sampling_one_step(model, optimizer, datasets, tasks, loss_weights, epoch, steps, loss_history, task_loss_history, sample_weights=sample_weights, k=k, loss_setting=loss_setting, batch_size=batch_size, text_pad_length=text_pad_length, img_pad_length=img_pad_length, audio_pad_length=audio_pad_length, shuffle=shuffle, device=device, initial_task_losses=initial_task_losses, gradnorm_optimizer=gradnorm_optimizer)
        else:
            loss_history, task_loss_history, steps, loss_weights, initial_task_losses, gradnorm_optimizer = train_one_step(model, optimizer, dataloader, tasks, loss_weights, epoch, steps, loss_history, task_loss_history, training_method=training_method, loss_setting=loss_setting, device=device, initial_task_losses=initial_task_losses, gradnorm_optimizer=gradnorm_optimizer)
        
        # print(loss_history)
        # print(task_loss_history)
        # print(steps)

        # then perform validation
        if json_val_path is not None:
            accuracies, f1_scores, average_accuracy, average_f1_score, val_average_total_loss, val_average_task_loss, all_labels, all_true_labels = evaluate(model, json_val_path, tasks, loss_weights=loss_weights, batch_size=batch_size, shuffle=shuffle, device=device)
            for task in tasks:
                print(f"Task {task}")
                # print(f"Number of items: {len(all_labels[task])}")
                # print(f"Predictions: {all_labels[task]}")
                # print(f"True: {all_true_labels[task]}")
                print(f"Accuracy: {accuracies[task]}, F1 Score: {f1_scores[task]:.4f}")
                print(f"Average Task Loss {val_average_task_loss[task]}")
                print()
            
            print(f"Average Total Loss {val_average_total_loss}")
            print(f"Average Accuracy: {average_accuracy}")
            print(f"Average F1 Score: {average_f1_score}")
            print()

            # Update the dictionary with the current epoch results
            for task, value in accuracies.items():
                if task not in validation_results["accuracies"]:
                    validation_results["accuracies"][task] = []
                validation_results["accuracies"][task].append(value)

            for task, value in f1_scores.items():
                if task not in validation_results["f1_scores"]:
                    validation_results["f1_scores"][task] = []
                validation_results["f1_scores"][task].append(value)

            for task, value in val_average_task_loss.items():
                if task not in validation_results["val_average_task_loss"]:
                    validation_results["val_average_task_loss"][task] = []
                validation_results["val_average_task_loss"][task].append(value)

            validation_results["average_accuracy"].append(average_accuracy)
            validation_results["average_f1_score"].append(average_f1_score)
            validation_results["val_average_total_loss"].append(val_average_total_loss)

            # step based on the average f1 score per the original code
            # note this step only works for the specific scheduler
            if scheduler is not None:
                scheduler.step(average_f1_score)

            if average_f1_score > best_f1:
                best_f1 = average_f1_score
                best_model_weights = model.state_dict()

    # prepare final results and return
    # Convert lists to tensors
    for task in validation_results["accuracies"]:
        validation_results["accuracies"][task] = torch.tensor(validation_results["accuracies"][task])

    for task in validation_results["f1_scores"]:
        validation_results["f1_scores"][task] = torch.tensor(validation_results["f1_scores"][task])

    for task in validation_results["val_average_task_loss"]:
        validation_results["val_average_task_loss"][task] = torch.tensor(validation_results["val_average_task_loss"][task])

    validation_results["average_accuracy"] = torch.tensor(validation_results["average_accuracy"])
    validation_results["average_f1_score"] = torch.tensor(validation_results["average_f1_score"])
    validation_results["val_average_total_loss"] = torch.tensor(validation_results["val_average_total_loss"])

    return best_model_weights, loss_history, task_loss_history, validation_results # note that loss_history has no meaning unless we add up loss functions
    # this should return another variable which contains model specific stuff (like loss histories, or sampling percentages of each task, etc.)

def train_one_step(model, optimizer, dataloader, tasks, loss_weights, curr_epoch, steps, curr_loss_history, curr_task_loss_history, training_method="all_at_once", loss_setting="unweighted", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), initial_task_losses=None, gradnorm_optimizer=None):

        model.train() # here because we evaluate at the end of every epoch
        start_time = time.time() 

        # print(len(dataloader)) 362 by 8 batches = 2896 examples - each is used for all 4 tasks

        for batch_idx, batch in enumerate(dataloader):

            dprint(f"DEVICE {device}")
            batch_text = batch['text'].to(device)
            batch_text_mask = batch['text_mask'].to(device)
            batch_image = batch['image'].float().to(device)
            batch_mask_img = batch['image_mask'].to(device)
            batch_audio = batch['audio'].float().to(device)
            batch_mask_audio = batch['audio_mask'].to(device)

            dprint(f"Text Batch Item 0 : {batch_text[0]}")
            dprint(f"Text Batch Mask Item 0 : {batch_text_mask[0]}")

            # shapes align with original model
            dprint(f"batch_text: {batch_text.shape}") # 8 500
            dprint(f"batch_text_mask: {batch_text_mask.shape}") # 8 500
            dprint(f"batch_image: {batch_image.shape}") # 8 36 1024
            dprint(f"batch_mask_img: {batch_mask_img.shape}") # 8 36
            dprint(f"batch_audio: {batch_audio.shape}") # 8 63 128
            dprint(f"batch_mask_audio: {batch_mask_audio.shape}") # 8 63


            dprint(f'Allocated: {torch.cuda.memory_allocated() / 1024**2} MB')
            dprint(f'Cached: {torch.cuda.memory_reserved() / 1024**2} MB')

            # this also works for individual tasks
            if training_method == "all_at_once":

                if steps == 0:
                    print(f"Training All At Once on the following tasks: {[task for task in tasks]}")
                
                optimizer.zero_grad()

                out = model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, tasks)

                # note that we can combined below loop with next task loop
                # is any gradient information messed up here? Or how about when you stack the outputs
                out_dict = {}
                for i, task in enumerate(tasks):
                    out_dict[task] = out[:, i, :]

                # again, are ther any issues using this dictionary?
                # CURRENTLY L2 REG ONLY ON BINARY LIKE IN ORIGINAL CODE, KINDA JANK
                task_losses = {}
                for task in tasks:
                    batch_pred = out_dict[task]
                    batch_true_y_task = batch[task].to(device) # TO DO NEED TO CHANGE "LABEL" TO "BINARY" IN DATASET
                    temp_loss = F.binary_cross_entropy(batch_pred, batch_true_y_task) 
                    task_losses[task] = temp_loss

           
                # sloppy
                if loss_setting == "gradnorm" and steps == 0: # initialize task losses and optimizer only in the first training step
                    # we need initial task losses for gradnorm - for each epoch or for total?
                    initial_task_losses = {key: loss.detach() for key, loss in task_losses.items()} # initial code detaches these
                    gradnorm_optimizer = torch.optim.Adam([loss_weights], lr=0.001)
    
                # Weight losses appropriately
                # given a dictionary of losses for every task, handle the loss reweigthing
                loss, loss_weights = handle_losses(task_losses, loss_setting, loss_weights, steps, tasks, initial_task_losses=initial_task_losses, loss_optimizer=gradnorm_optimizer, model=model)
                
                loss.backward()
                optimizer.step()

                # Update Loss Histories
                curr_loss_history[steps] = loss

                #print(f"Batch {batch_idx} Total Loss: {loss}")
                for i, task in enumerate(task_losses):
                    #print(f"Task {task}, Loss {task_losses[task]}")
                    curr_task_loss_history[task][steps] = loss_weights[i] * task_losses[task]
                #print()

            # CULLEN: INITIAL ROUND ROBIN IMPLEMENTATION - total loss history not used here
            # Take as input the task specific heads you wish to use
            # Iterate through each task and get the loss just for that output
            # Currently this assumes the dataset where each item has every output so we reuse batches

            # NOTE - THIS DOES 4 * MORE BACKPASSES THAN THE ORIGINAL MODEL - careful of this
            # the only difference with original is we call backward here instead of adding up losses
            elif training_method == "round_robin":
                if steps == 0:
                    print(f"Training Naive Round Robin on the following tasks: {[task for task in tasks]}")

                # kind of weird thing going on here
                # We need to compute the output seperately for every task in order to do backprop I believe
                for i, task in enumerate(tasks): 
                    optimizer.zero_grad() # we train on every task, so zero_grad on every task

                    # get prediction just for task
                    # this assumes model output size is batch_size by tasks by dimensions
                    batch_pred = model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, [tasks[i]])[:, 0, :]
                    batch_true_y_task = batch[task].to(device)

                    loss = F.binary_cross_entropy(batch_pred, batch_true_y_task)

                    loss.backward()
                    optimizer.step()
                    print(f"Task: {task}, Current Loss {loss}")

                    curr_task_loss_history[task][steps] = loss # update the task loss history with the current loss
                    # WARNING: There are 4 * more steps here because each training example is used for four backprop steps

            # IDEA - could loss reweighting have a roll here at all?
            # combined idea - Loss reweighting on round robin instead of
            # We would train one task at a time but weight them differently
            
            # added below to halt early for easier testing
            # if batch_idx == 1:
            #     break

            # TO DO: This way of reporting loss doesn't make sense with round robin (have to divide total loss by 4 * batch_idx + 1)
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{curr_epoch + 1}, Step [{steps+1}], Total Loss: {loss:0.4f}, Time: {time.time() - start_time:.2f}s')
                for task, task_loss in task_losses.items():
                    print(f"Task {task}, Loss {task_loss}")
                print()

            steps += 1

            # halt early for testing
            if debug and steps == 5:
                break

        return curr_loss_history, curr_task_loss_history, steps, loss_weights, initial_task_losses, gradnorm_optimizer
        # return model, task specific losses, final losses, model specificstuff

# the main part here is how batching is done
# this can probably best be handled using a custom data loader ? 
# this is done in a way that only works for the one dataset multiple tasks method and is pretty model specific
def dynamic_difficulty_sampling_one_step(model, optimizer, datasets, tasks, loss_weights, curr_epoch, steps, curr_loss_history, curr_task_loss_history, sample_weights, k=2, loss_setting="unweighted", batch_size=32, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), initial_task_losses=None, gradnorm_optimizer=None):
    # Dynamic Difficulty Sampling
    # Create a dataset for every individual task (here just four from the same data source)
    # We have to manually create every batch for dynamic difficulty sampling (I don't see how to do with DataLoader)
    # After creating every batch, we can train just as we would on any multi task training loop
    # Here we calculate the loss for every task, and calculate the unweighted sum
    # Every k steps we update the sampling weights by averaging out task losses and total loss and taking the ratio (task_weight = average_task_loss / average_total_loss)
    # We continue until one dataset is exhausted (maybe not the best way)
    model.train()
    start_time = time.time()
    
    if steps == 0:
        print(f"Dynamic Difficulty Sampling on the following tasks: {[task for task in tasks]}")
        
    # keep track of where in the dataset we are
    dataset_indices = {}
    for task in tasks:
        dataset_indices[task] = 0

    accumulated_total_loss = 0 # accumulate loss every k steps for dynamic training
    accumulated_task_loss = {} # also accumulate task specific loss
    for task in tasks:
        accumulated_task_loss[task] = 0

    print(f"Number of Data Points {len(datasets[task])} / Batch Size {batch_size} = Steps {len(datasets[task]) / batch_size}")

    num_batches=0
    # corresponds to iterating through batches - while no batch is exhausted
    while all(dataset_indices[task] < len(datasets[task]) for task in tasks):
        # update sampling weights every k iterations - again check if these steps should count from the begining
        if steps != 0 and steps % k == 0: 
            average_loss = accumulated_total_loss / k # average total loss
            print(f"Average Total Loss {average_loss}")

            average_task_losses = {task: loss/k for task, loss in accumulated_task_loss.items()} # average loss per task
            print(f"Average Loss Per Task: {[f'{task}: {task_loss}' for task, task_loss in average_task_losses.items()]}")

            sample_weights = {task: average_task_losses[task]/average_loss for task in tasks}
            print(f"Updated Sample Weights {sample_weights}")

            # reset these values
            accumulated_total_loss = 0
            accumulated_task_loss = {}
            for task in tasks:
                accumulated_task_loss[task] = 0

        dprint(f"Dataset Indices: {dataset_indices}")
        dprint(f"Sampling Weights {sample_weights}")

        # determine how many data points to get from each task
        num_datas = {}
        for task in tasks:
            num_datas[task] = max(math.ceil(sample_weights[task] * batch_size), 4) # minimum of 4 per batch

        true_batch_size = sum(num_datas.values()) # sum of number of samples from each task
        dprint(f"True Batch Size {true_batch_size}, Original {batch_size}")

        # using true batch size because enforcing minimum 4 samples may make batch size not equal to the true value
        batch_text = torch.zeros((true_batch_size, text_pad_length), dtype=torch.int)
        batch_text_mask = torch.zeros((true_batch_size, text_pad_length))
        batch_image = torch.zeros((true_batch_size, img_pad_length, 1024))
        batch_mask_img = torch.zeros((true_batch_size, img_pad_length))
        batch_audio = torch.zeros((true_batch_size, audio_pad_length, 128))
        batch_mask_audio = torch.zeros((true_batch_size, audio_pad_length))
        batch_true_labels = {}
        batch_idx = 0 # tells you where to start placing data within one batch
        for task in tasks:
            dprint(f"Task {task}, Samples: {num_datas[task]}")
            start_idx = dataset_indices[task] # this is the first piece of data not used yet
            dataset_indices[task] += num_datas[task] # update start index
            batch_true_labels[task] = torch.zeros((num_datas[task], 2)) # store true y values for each task
            dprint(f"Batch idx: {batch_idx}, Task start idx: {start_idx}, Updated start idx (after here): {dataset_indices[task]}")
            for i in range(num_datas[task]):
                batch_text[batch_idx+i] = datasets[task][start_idx+i]["text"]
                batch_text_mask[batch_idx+i] = datasets[task][start_idx+i]["text_mask"]
                batch_image[batch_idx+i] = datasets[task][start_idx+i]["image"]
                batch_mask_img[batch_idx+i] = datasets[task][start_idx+i]["image_mask"]
                batch_audio[batch_idx+i] = datasets[task][start_idx+i]["audio"]
                batch_mask_audio[batch_idx+i] = datasets[task][start_idx+i]["audio_mask"]
                batch_true_labels[task][i] = datasets[task][start_idx+i][task]
            
            batch_idx += num_datas[task]
            # handle true labels separately because it is a dictionary
            batch_true_labels[task] = batch_true_labels[task].to(device)

        batch_text = batch_text.to(device)
        batch_text_mask = batch_text_mask.to(device)
        batch_image =  batch_image.to(device)
        batch_mask_img = batch_mask_img.to(device)
        batch_audio = batch_audio.to(device)
        batch_mask_audio = batch_mask_audio.to(device)

        # now have true_batch_size items of data
        # there are also labels corresponding to each task
        # note that if we have texts where data is drawn from different format datasets, this would need to be modified
        # but this shouldn't be the case since we need these 6 inputs to fit into our model

        dprint(f"batch_text: {batch_text.shape}")
        dprint(f"batch_text_mask: {batch_text_mask.shape}")
        dprint(f"batch_image: {batch_image.shape}")
        dprint(f"batch_mask_img: {batch_mask_img.shape}")
        dprint(f"batch_audio: {batch_audio.shape}")
        dprint(f"batch_mask_audio: {batch_mask_audio.shape}")
        for task in tasks:
            dprint(f"batch_label: {batch_true_labels[task]}")

        # all batches have been created now

        optimizer.zero_grad() # maybe want to place above somewhere

        out = model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, tasks)

        # THE BELOW CAN BE TREATED SAME AS MULTI TASK LEARNING IN ORIGINAL LOOP
        # MODULARIZE TO AVOID COPY PASTING, AND TO ENABLE FURTHER EXPERIMENTATION
        # e.g. what if we do dynamic presampling and loss reweighting strategies?
        # or what if we do round robin on dynamic pretraining batches?

        # put model outputs for each task into a dictionary
        out_dict = {}
        start_idx = 0
        for i, task in enumerate(tasks):
            # start_idx:start_idx+num_datas[task] gives training examples in batch, i gives task specific prediciton
            out_dict[task] = out[start_idx:start_idx+num_datas[task], i, :]
            start_idx += num_datas[task]

        dprint(f"PREDICTIONS {out_dict}")
        dprint(f"TRUE LABELS {batch_true_labels}")

        # with the batches, we can try like before
        task_losses = {}
        for task in tasks:
            batch_pred = out_dict[task]
            batch_true_y_task = batch_true_labels[task]
            temp_loss = F.binary_cross_entropy(batch_pred, batch_true_y_task) 
            task_losses[task] = temp_loss
        
        # this is kind of ugly but it should be fine
        if loss_setting == "gradnorm" and steps == 0: # initialize task losses and optimizer only in the first training step
            # we need initial task losses for gradnorm - for each epoch or for total?
            initial_task_losses = {key: loss.detach() for key, loss in task_losses.items()} # initial code detaches these
            gradnorm_optimizer = torch.optim.Adam([loss_weights], lr=0.001) 

        # Remember that task_losses is unweighted - weight it when we need to
        loss, loss_weights = handle_losses(task_losses, loss_setting, loss_weights, steps, tasks, initial_task_losses=initial_task_losses, loss_optimizer=gradnorm_optimizer, model=model)

        loss.backward()
        optimizer.step()

        # steps can get messed up due to the reordering of stuff
        # we want a more robust solution, but this is fine for now
        try:
            curr_loss_history[steps] = loss # just the naive sum of losses for now
        except:
            pass
        accumulated_total_loss += loss

        # print(f"Batch {num_batches} Total Loss: {loss}")
        for i, task in enumerate(task_losses):
            # print(f"Task {task}, Loss {task_losses[task]}")
            # need more robust solution to the below
            try:
                curr_task_loss_history[task][steps] = loss_weights[i] * task_losses[task]
            except:
                pass
            accumulated_task_loss[task] += (loss_weights[i] * task_losses[task]) # accumulate weighted losses  

        if (num_batches + 1) % 10 == 0:
            print(f'Epoch [{curr_epoch + 1}, Step [{steps+1}], Total Loss: {loss:0.4f}, Time: {time.time() - start_time:.2f}s')
            for task, task_loss in task_losses.items():
                print(f"Task {task}, Loss {task_loss}")
                print(f"Task {task} Current Index: {dataset_indices[task]}")
            print()

        steps += 1 # steps from beginning
        num_batches +=1 # batches for this epoch

        # halt early for testing
        if debug and steps == 5:
            break

    return curr_loss_history, curr_task_loss_history, steps, loss_weights, initial_task_losses, gradnorm_optimizer

# this grad norm step should not be here
def handle_losses(task_losses, loss_setting, loss_weights, steps, tasks, initial_task_losses=None, loss_optimizer=None, model=None):

    # ON 4 MULTI TASKS WITH GIVEN WEIGHTS
    # use this to replicate paper's experiment
    if loss_setting == "predefined_weights" and set(tasks) == {"mature", "gory", "sarcasm", "slapstick"}:
        if steps == 0:
            print("Predefined Weights")
        loss_weights = [mature_w, gory_w, sarcasm_w, slap_w] # for consistency
        loss = mature_w*task_losses["mature"] + gory_w*task_losses["gory"] + sarcasm_w*task_losses["sarcasm"] + slap_w*task_losses["slapstick"]

    # below two are untested
    # learned loss weightings for whichever ordering of tasks
    # might want to change this to declare the parameter right here like grad norm
    # this shouldn't effect other optimization at all simply because these loss weights won't be used
    elif loss_setting == "weighted":
        if steps == 0:
            print("Learnable Weights")
        loss = 0
        for task_loss in task_losses.values():
            loss += loss_weights[i] * task_loss
    
    # CAN PUT GRADNORM HERE (THOUGH IT NEEDS STUFF EARLIER)
    # if i were to modularize, this would take in task losses as values - but it might not be that simple
    # It might be better to do this and training together in case of issues in optimization
    # TO DO: Also return method specific parameters - in this case loss weight history
    elif loss_setting == "gradnorm": # won't work if initial_task_losses and loss_optimizer is None, which should be fine
        if steps == 0:
            print("Gradnorm Weights")
        loss = 0
        # passing in model.base_model here should be put elsewhere since it makes the training loop model_specific
        # can enforce these task specific hyperparameters using **kwargs maybe?
        # Make consistent the way we handle loss weights would be better
        # Currently this returns weigts and does not update model.loss_weights
        loss_weights = gradnorm(task_losses, initial_task_losses, model, model.base_model, loss_optimizer, loss_weights) # update model weights using grad norm
        for i, task_loss in enumerate(task_losses.values()):
            loss += loss_weights[i] * task_loss # currently weights just come from gradnorm and are aliged based on gradnorm implemetation
        # then backward step like before

    # don't weight losses
    else:
        if steps == 0:
            print("Unweighted")
        loss = 0
        for task_loss in task_losses.values():
            loss += task_loss

    return loss, loss_weights


# OLD BEFORE

# double check placement of zero_grad makes sense
def train(model, optimizer, json_train_path, tasks, scheduler=None, json_val_path=None, training_method="all_at_once", loss_setting="unweighted", batch_size=16, num_epochs=1, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # the dataset has x values: 
    # text, text_mask, image, image_mask, audio, audio_maxk
    # and y value (deal with depending on task)
    dataset = CustomDataset(json_train_path, text_pad_length, img_pad_length, audio_pad_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print("CREATED DATASET AND DATALOADER")
    print()

    # Total number of training steps (epochs * batches)
    total_steps = len(dataloader) * num_epochs

    # alternate scheduler below - probably no need for
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=0,
    #                                             num_training_steps=total_steps,
    #                                             last_epoch=-1)

    loss_history = torch.zeros(total_steps) # track the total loss at every step
    task_loss_history = {task : torch.zeros(total_steps) for task in tasks} # track the loss for every task

    validation_results = {
        "accuracies": {},
        "f1_scores": {},
        "average_accuracy": [],
        "average_f1_score": [],
        "val_average_total_loss": [],
        "val_average_task_loss": {}
    }

    # Initialize loss weights as vector of all ones before training
    loss_weights = torch.ones(len(tasks))
    loss_weights = nn.Parameter(loss_weights)
    dprint(f"Initial Loss Weights {loss_weights}")

    steps = 0 # a step is one batch aka one training step
    for epoch in range(num_epochs):
        model.train() # here because we evaluate at the end of every epoch
        model.train() # here because we evaluate at the end of every epoch
        print(f"EPOCH: {epoch}")
        start_time = time.time() 

        # dynamic stop and go would be kind of like all in one (and don't weight losses)

        for batch_idx, batch in enumerate(dataloader):

            dprint(f"DEVICE {device}")
            batch_text = batch['text'].to(device)
            batch_text_mask = batch['text_mask'].to(device)
            batch_image = batch['image'].float().to(device)
            batch_mask_img = batch['image_mask'].to(device)
            batch_audio = batch['audio'].float().to(device)
            batch_mask_audio = batch['audio_mask'].to(device)

            dprint(f"Text Batch Item 0 : {batch_text[0]}")
            dprint(f"Text Batch Mask Item 0 : {batch_text_mask[0]}")

            # shapes align with original model
            dprint(f"batch_text: {batch_text.shape}") # 8 500
            dprint(f"batch_text_mask: {batch_text_mask.shape}") # 8 500
            dprint(f"batch_image: {batch_image.shape}") # 8 36 1024
            dprint(f"batch_mask_img: {batch_mask_img.shape}") # 8 36
            dprint(f"batch_audio: {batch_audio.shape}") # 8 63 128
            dprint(f"batch_mask_audio: {batch_mask_audio.shape}") # 8 63


            dprint(f'Allocated: {torch.cuda.memory_allocated() / 1024**2} MB')
            dprint(f'Cached: {torch.cuda.memory_reserved() / 1024**2} MB')

            # this also works for individual tasks
            if training_method == "all_at_once":

                if steps == 0:
                    print(f"Training All At Once on the following tasks: {[task for task in tasks]}")
                
                optimizer.zero_grad()

                out = model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, tasks)

                # note that we can combined below loop with next task loop
                # is any gradient information messed up here? Or how about when you stack the outputs
                out_dict = {}
                for i, task in enumerate(tasks):
                    out_dict[task] = out[:, i, :]

                # again, are ther any issues using this dictionary?
                # CURRENTLY L2 REG ONLY ON BINARY LIKE IN ORIGINAL CODE, KINDA JANK
                task_losses = {}
                for task in tasks:
                    batch_pred = out_dict[task]
                    batch_true_y_task = batch[task].to(device) # TO DO NEED TO CHANGE "LABEL" TO "BINARY" IN DATASET
                    temp_loss = F.binary_cross_entropy(batch_pred, batch_true_y_task) 
                    task_losses[task] = temp_loss

                # check below
                if loss_setting == "gradnorm" and steps == 0: # initialize task losses and optimizer only in the first training step
                    # we need initial task losses for gradnorm - for each epoch or for total?
                    initial_task_losses = {key: loss.detach() for key, loss in task_losses.items()} # initial code detaches these
                    gradnorm_optimizer = torch.optim.Adam([loss_weights], lr=0.001) 
                
                # TO DO - Below is pretty sloppy I think, see if we can make this nicer
                # Note the use of regularization in one task and not in multi tasks, per the original code (probably fix later)
                            
                # ON ONE TASK - USE THIS FOR SAME RESULT AS BINARY IN ORIGINAL CODE
                if len(tasks) == 1:
                    loss = task_losses[tasks[0]] + compute_l2_reg_val(model) # the use of l2 reg explodes loss and is not used otherwise

                # ON 4 MULTI TASKS WITH GIVEN WEIGHTS
                # use this to replicate paper's experiment
                elif loss_setting == "predefined_weights" and set(tasks) == {"mature", "gory", "sarcasm", "slapstick"}:
                    loss_weights = [mature_w, gory_w, sarcasm_w, slap_w] # for consistency
                    loss = mature_w*task_losses["mature"] + gory_w*task_losses["gory"] + sarcasm_w*task_losses["sarcasm"] + slap_w*task_losses["slapstick"]

                # below two are untested
                # learned loss weightings for whichever ordering of tasks
                # might want to change this to declare the parameter right here like grad norm
                # this shouldn't effect other optimization at all simply because these loss weights won't be used
                elif loss_setting == "weighted":
                    loss = 0
                    for task_loss in task_losses.values():
                        dprint(f"Unweighted Task Loss {task_loss}")
                        dprint(f"Loss Weight: {loss_weights[i]}")
                        dprint(f"Weighted Task Loss: {loss_weights[i] * task_loss}")
                        loss += loss_weights[i] * task_loss
                
                # CAN PUT GRADNORM HERE (THOUGH IT NEEDS STUFF EARLIER)
                # if i were to modularize, this would take in task losses as values - but it might not be that simple
                # It might be better to do this and training together in case of issues in optimization
                # TO DO: Also return method specific parameters - in this case loss weight history
                elif loss_setting == "gradnorm":
                    dprint("Beginning Grad Norm")
                    loss = 0
                    # passing in model.base_model here should be put elsewhere since it makes the training loop model_specific
                    # can enforce these task specific hyperparameters using **kwargs maybe?
                    # Make consistent the way we handle loss weights would be better
                    # Currently this returns weigts and does not update model.loss_weights
                    loss_weights = gradnorm(task_losses, initial_task_losses, model, model.base_model, gradnorm_optimizer, loss_weights) # update model weights using grad norm
                    dprint(f"Updated Weights {loss_weights}")
                    for i, task_loss in enumerate(task_losses.values()):
                        dprint(f"Task {i}")
                        dprint(f"Unweighted Task Loss {task_loss}")
                        dprint(f"Loss Weight: {loss_weights}")
                        dprint(f"Weighted Task Loss: {loss_weights * task_loss}\n\n")
                        loss += loss_weights[i] * task_loss # currently weights just come from gradnorm and are aliged based on gradnorm implemetation
                    # then backward step like before

                # don't weight losses
                else:
                    loss = 0
                    for task_loss in task_losses.values():
                        loss += task_loss


                loss.backward()
                optimizer.step()

                loss_history[steps] = loss

                print(f"Batch {batch_idx} Total Loss: {loss}")
                for task, task_loss in task_losses.items():
                    print(f"Task {task}, Loss {task_loss}")
                    task_loss_history[task][steps] = task_loss
                print()

                steps += 1

            # CULLEN: INITIAL ROUND ROBIN IMPLEMENTATION
            # Take as input the task specific heads you wish to use
            # Iterate through each task and get the loss just for that output
            # Currently this assumes the dataset where each item has every output so we reuse batches

            # NOTE - THIS DOES 4 * MORE BACKPASSES THAN THE ORIGINAL MODEL - careful of this
            # the only difference with original is we call backward here instead of adding up losses
            elif training_method == "round_robin":
                if steps == 0:
                    print(f"Training Naive Round Robin on the following tasks: {[task for task in tasks]}")

                # kind of weird thing going on here
                # We need to compute the output seperately for every task in order to do backprop I believe
                for i, task in enumerate(tasks): 
                    optimizer.zero_grad() # we train on every task, so zero_grad on every task

                    # get prediction just for task
                    # this assumes model output size is batch_size by tasks by dimensions
                    batch_pred = model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, [tasks[i]])[:, 0, :]
                    batch_true_y_task = batch[task].to(device)

                    loss = F.binary_cross_entropy(batch_pred, batch_true_y_task)
                    optimizer.step()
                    print(f"Task: {task}, Current Loss {loss}")

                    task_loss_history[task][steps] = loss # update the task loss history with the current loss
                    # WARNING: There are 4 * more steps here because each training example is used for four backprop steps

                steps += 1 # WARNING - UPDATE STEP AFTER EVERY TASK IS STEPPED THROUGH - but really there are 4 gradient updates

            # IDEA - could loss reweighting have a roll here at all?
            # combined idea - Loss reweighting on round robin instead of
            # We would train one task at a time but weight them differently
            
            # added below to halt early for easier testing
            # if batch_idx == 1:
            #     break

            # TO DO: This way of reporting loss doesn't make sense with round robin (have to divide total loss by 4 * batch_idx + 1)
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Total Loss: {loss:0.4f}, Time: {time.time() - start_time:.2f}s')
                for task, task_loss in task_losses.items():
                    print(f"Task {task}, Loss {task_loss}")
                print()

        # finished an epoch
       
        # TO DO: All of this should be put in a separate function, not here
        # validate at the end of every epoch
        if json_val_path is not None:
            accuracies, f1_scores, average_accuracy, average_f1_score, val_average_total_loss, val_average_task_loss, all_labels, all_true_labels = evaluate(model, json_val_path, tasks, loss_weights=loss_weights, batch_size=batch_size, shuffle=shuffle, device=device)
            for task in tasks:
                print(f"Task {task}")
                # print(f"Number of items: {len(all_labels[task])}")
                # print(f"Predictions: {all_labels[task]}")
                # print(f"True: {all_true_labels[task]}")
                print(f"Accuracy: {accuracies[task]}, F1 Score: {f1_scores[task]:.4f}")
                print(f"Average Task Loss {val_average_task_loss[task]}")
                print()
            
            print(f"Average Total Loss {val_average_total_loss}")
            print(f"Average Accuracy: {average_accuracy}")
            print(f"Average F1 Score: {average_f1_score}")

            # Update the dictionary with the current epoch results
            for task, value in accuracies.items():
                if task not in validation_results["accuracies"]:
                    validation_results["accuracies"][task] = []
                validation_results["accuracies"][task].append(value)

            for task, value in f1_scores.items():
                if task not in validation_results["f1_scores"]:
                    validation_results["f1_scores"][task] = []
                validation_results["f1_scores"][task].append(value)

            for task, value in val_average_task_loss.items():
                if task not in validation_results["val_average_task_loss"]:
                    validation_results["val_average_task_loss"][task] = []
                validation_results["val_average_task_loss"][task].append(value)

            validation_results["average_accuracy"].append(average_accuracy)
            validation_results["average_f1_score"].append(average_f1_score)
            validation_results["val_average_total_loss"].append(val_average_total_loss)

            # step based on the average f1 score per the original code
            # note this step only works for the specific scheduler
            if scheduler is not None:
                scheduler.step(average_f1_score)


    # Convert lists to tensors
    for task in validation_results["accuracies"]:
        validation_results["accuracies"][task] = torch.tensor(validation_results["accuracies"][task])

    for task in validation_results["f1_scores"]:
        validation_results["f1_scores"][task] = torch.tensor(validation_results["f1_scores"][task])

    for task in validation_results["val_average_task_loss"]:
        validation_results["val_average_task_loss"][task] = torch.tensor(validation_results["val_average_task_loss"][task])

    validation_results["average_accuracy"] = torch.tensor(validation_results["average_accuracy"])
    validation_results["average_f1_score"] = torch.tensor(validation_results["average_f1_score"])
    validation_results["val_average_total_loss"] = torch.tensor(validation_results["val_average_total_loss"])

    return model, loss_history, task_loss_history, validation_results # note that loss_history has no meaning unless we add up loss functions
    # this should return another variable which contains model specific stuff (like loss histories, or sampling percentages of each task, etc.)


# IDEA - GradNorm (and adjacent techniques) can also be easily combined with dynamic difficulty sampling
# so we definetly need a modularized implementaiton

# need to incorporate adamw and learning rate scheduler here

# they don't normalize losses, we probably should
# k should be 100, but 3 here for speed
def dynamic_difficulty_sampling(model, optimizer, json_train_path, tasks, k=15, loss_setting="unweighted", batch_size=32, num_epochs=1, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Dynamic Difficulty Sampling
    # Create a dataset for every individual task (here just four from the same data source)
    # We have to manually create every batch for dynamic difficulty sampling (I don't see how to do with DataLoader)
    # After creating every batch, we can train just as we would on any multi task training loop
    # Here we calculate the loss for every task, and calculate the unweighted sum
    # Every k steps we update the sampling weights by averaging out task losses and total loss and taking the ratio (task_weight = average_task_loss / average_total_loss)
    # We continue until one dataset is exhausted (maybe not the best way)
    
    # It's quite messy because there's a lot of variables for keeping track of indices and stuff, don't focus on this too much
    # The difficulty comes from not having a DataLoader to make the batches for us

    datasets = {} # store the dataset for every task
    sample_weights = {} # controls weight of each dataset in the batch
    for task in tasks:
        datasets[task] = CustomDataset(json_train_path, text_pad_length, img_pad_length, audio_pad_length)

        # TO DO Probably want to implement shuffling here, or implement it as a method of the dataset
        
        sample_weights[task] = 1/len(tasks) # initialize so all tasks are weighted the same

    # sampling weights for testing
    # sample_weights["mature"] = 0.4
    # sample_weights["gory"] = 0.4
    # sample_weights["sarcasm"] = 0.1
    # sample_weights["slapstick"] = 0.1

    for epoch in range(num_epochs):
        dataset_indices = {}
        for task in tasks:
            dataset_indices[task] = 0

         # add extra for loop here to deal with all batches

        # CURRENT METHOD STOPS WHEN ONE DATASET IS EXHAUSTED
        # THIS MAY NOT BE IDEAL
        # this is the method to loop through batches

        accumulated_total_loss = 0 # accumulate loss every k steps for dynamic training
        accumulated_task_loss = {} # also accumulate task specific loss
        for task in tasks:
            accumulated_task_loss[task] = 0
        steps = 0 # track the number of steps made, e.g. mini batches trained on
        while all(dataset_indices[task] < len(datasets[task]) for task in tasks):
            # update sampling weights every k iterations
            if steps != 0 and steps % k == 0: 
                average_loss = accumulated_total_loss / k # average total loss
                dprint(f"Average Loss {average_loss}")

                average_task_losses = {task: loss/k for task, loss in accumulated_task_loss.items()} # average loss per task
                print(f"Average Loss Per Task: {[f'{task}: {task_loss}' for task, task_loss in average_task_losses.items()]}")

                sample_weights = {task: average_task_losses[task]/average_loss for task in tasks}
                dprint(f"Updated Sample Weights {sample_weights}")

            dprint(f"Dataset Indices: {dataset_indices}")

            num_datas = {}
            for task in tasks:
                num_datas[task] = max(math.ceil(sample_weights[task] * batch_size), 2)

            true_batch_size = sum(num_datas.values()) # sum of number of samples from each task
            dprint(f"True Batch Size {true_batch_size}, Original {batch_size}")

            # using true batch size because enforcing minimum 4 samples may make batch size not equal to the true value
            batch_text = torch.zeros((true_batch_size, text_pad_length))
            batch_text_mask = torch.zeros((true_batch_size, text_pad_length))
            batch_image = torch.zeros((true_batch_size, img_pad_length, 1024))
            batch_mask_img = torch.zeros((true_batch_size, img_pad_length))
            batch_audio = torch.zeros((true_batch_size, audio_pad_length, 128))
            batch_mask_audio = torch.zeros((true_batch_size, audio_pad_length))
            batch_true_labels = {}
            batch_idx = 0 # tells you where to start placing data within one batch
            for task in tasks:
                dprint(f"Task {task}, Samples: {num_datas[task]}")
                start_idx = dataset_indices[task]
                dataset_indices[task] += num_datas[task]
                batch_true_labels[task] = torch.zeros((num_datas[task], 2))
                for i in range(num_datas[task]):
                    batch_text[batch_idx+i] = datasets[task][start_idx+i]["text"]
                    batch_text_mask[batch_idx+i] = datasets[task][start_idx+i]["text_mask"]
                    batch_image[batch_idx+i] = datasets[task][start_idx+i]["image"]
                    batch_mask_img[batch_idx+i] = datasets[task][start_idx+i]["image_mask"]
                    batch_audio[batch_idx+i] = datasets[task][start_idx+i]["audio"]
                    batch_mask_audio[batch_idx+i] = datasets[task][start_idx+i]["audio_mask"]
                    batch_true_labels[task][i] = datasets[task][start_idx+i][task]
                batch_idx += num_datas[task]
                batch_true_labels[task] = batch_true_labels[task].to(device)

            batch_text = batch_text.to(device)
            batch_text_mask = batch_text_mask.to(device)
            batch_image =  batch_image.float().to(device)
            batch_mask_img = batch_mask_img.to(device)
            batch_audio = batch_audio.float().to(device)
            batch_mask_audio = batch_mask_audio.to(device)

            # now have true_batch_size items of data
            # there are also labels corresponding to each task
            # note that if we have texts where data is drawn from different format datasets, this would need to be modified
            # but this shouldn't be the case since we need these 6 inputs to fit into our model

            dprint(f"batch_text: {batch_text.shape}")
            dprint(f"batch_text_mask: {batch_text_mask.shape}")
            dprint(f"batch_image: {batch_image.shape}")
            dprint(f"batch_mask_img: {batch_mask_img.shape}")
            dprint(f"batch_audio: {batch_audio.shape}")
            dprint(f"batch_mask_audio: {batch_mask_audio.shape}")
            for task in tasks:
                dprint(f"batch_label: {batch_true_labels[task]}")

            optimizer.zero_grad() # maybe want to place above somewhere

            out = model(batch_text.int(), batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, tasks)

            # THE BELOW CAN BE TREATED SAME AS MULTI TASK LEARNING IN ORIGINAL LOOP
            # MODULARIZE TO AVOID COPY PASTING, AND TO ENABLE FURTHER EXPERIMENTATION
            # e.g. what if we do dynamic presampling and loss reweighting strategies?
            # or what if we do round robin on dynamic pretraining batches?

            out_dict = {}
            start_idx = 0
            for i, task in enumerate(tasks):
                out_dict[task] = out[start_idx:start_idx+num_datas[task], i, :]
                start_idx += num_datas[task]

            dprint(f"PREDICTIONS {out_dict}")
            dprint(f"TRUE LABELS {batch_true_labels}")

            # with the batches, we can try like before
            task_losses = {}
            for task in tasks:
                batch_pred = out_dict[task]
                batch_true_y_task = batch_true_labels[task]
                temp_loss = F.binary_cross_entropy(batch_pred, batch_true_y_task) 
                task_losses[task] = temp_loss
            
            # doing this seperately to enable weightings, but could be done better
            loss = 0
            for task in tasks:
                loss += task_losses[task]
                accumulated_task_loss[task] += task_losses[task]
            
            accumulated_total_loss += loss # keep accumulating loss to average out later
            dprint(f"Total Loss: {loss}, {[f'{task}: {task_losses[task]}' for task in tasks]}\n\n")

            loss.backward()
            optimizer.step()

            steps += 1

    return model

if __name__ == "__main__":
    # these import statements don't work here for now
    from BaseModel import Bert_Model
    from TaskHeads import BinaryClassification, MultiTaskClassification
    from UnifiedModel import UnifiedModel
    # Usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model = Bert_Model()
    task_heads = {
        "binary": BinaryClassification(),
        "multi": MultiTaskClassification()
    }
    model = UnifiedModel(base_model, task_heads)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train(model, optimizer, "train_features_lrec_camera.json", "binary", batch_size=8, num_epochs=10)
    train(model, optimizer, "train_features_lrec_camera.json", "multi", batch_size=8, num_epochs=10, shuffle=False)


# previously below was in round robin
# shouldn't need this try except loop
# try:
#     dprint(f"batch_text: {batch_text}")
#     dprint(f"batch_text type and size {type(batch_text), batch_text.shape}") # batch_size by 500 tensor
#     dprint(f"batch_text_mask: {batch_text_mask}")
#     dprint(f"batch_image: {batch_image}")
#     dprint(f"batch_mask_img: {batch_mask_img}")
#     dprint(f"batch_audio: {batch_audio}")
#     dprint(f"batch_mask_audio: {batch_mask_audio}")
#     dprint(f"batch_pred: {batch_pred}")
#     loss = F.binary_cross_entropy(batch_pred, batch_true_y_task)
# except:
#     # Print the variables
#     dprint(f"batch_text: {batch_text}")
#     dprint(f"batch_text type and size {type(batch_text), batch_text.shape}")
#     dprint(f"batch_text_mask: {batch_text_mask}")
#     dprint(f"batch_image: {batch_image}")
#     dprint(f"batch_mask_img: {batch_mask_img}")
#     dprint(f"batch_audio: {batch_audio}")
#     dprint(f"batch_mask_audio: {batch_mask_audio}")
#     dprint(f"batch_pred: {batch_pred}")
#     break
# loss.backward()
