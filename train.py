import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
from finetuning_dataloader import CustomDataset
from gradnorm import gradnorm
import math

from helpers import compute_l2_reg_val

import time

debug = True

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

# double check placement of zero_grad makes sense
def train(model, optimizer, json_data_path, tasks, training_method="all_at_once", loss_setting="unweighted", batch_size=32, num_epochs=1, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # the dataset has x values: 
    # text, text_mask, image, image_mask, audio, audio_maxk
    # and y value (deal with depending on task)
    dataset = CustomDataset(json_data_path, text_pad_length, img_pad_length, audio_pad_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print("CREATED DATASET AND DATALOADER")
    print()

    model.train()
    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        start_time = time.time() 
        total_loss = 0

        # dynamic stop and go would be kind of like all in one (and don't weight losses)

        for batch_idx, batch in enumerate(dataloader):
            # print(f"BATCH NUMBER: {batch_idx}")
            # each batch is a dictionary and contains tensors for everything

            dprint(f"DEVICE {device}")
            batch_text = batch['text'].to(device)
            batch_text_mask = batch['text_mask'].to(device) # this is inverted for some reason, might be part of model code
            batch_image = batch['image'].float().to(device)
            batch_mask_img = batch['image_mask'].to(device)
            batch_audio = batch['audio'].float().to(device)
            batch_mask_audio = batch['audio_mask'].to(device)

            # dprint(f"batch_text: {batch_text}")
            # dprint(f"batch_text type and size {type(batch_text), batch_text.shape}") # batch_size by 500 tensor
            # dprint(f"batch_text_mask: {batch_text_mask}")
            # dprint(f"batch_image: {batch_image}")
            # dprint(f"batch_mask_img: {batch_mask_img}")
            # dprint(f"batch_audio: {batch_audio}")
            # dprint(f"batch_mask_audio: {batch_mask_audio}")

            dprint(f'Allocated: {torch.cuda.memory_allocated() / 1024**2} MB')
            dprint(f'Cached: {torch.cuda.memory_reserved() / 1024**2} MB')

            # this also works for individual tasks
            if training_method == "all_at_once":
                optimizer.zero_grad()

                out = model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, tasks)

                out_dict = {}
                for i, task in enumerate(tasks):
                    out_dict[task] = out[:, i, :]

                if batch_idx == 0 and epoch == 0:
                    print(f"Training All At Once on the following tasks: {[task for task in tasks]}")
                # CURRENTLY L2 REG ONLY ON BINARY LIKE IN ORIGINAL CODE, KINDA JANK
                task_losses = {}
                for task in tasks:
                    batch_pred = out_dict[task]
                    batch_true_y_task = batch[task].to(device) # TO DO NEED TO CHANGE "LABEL" TO "BINARY" IN DATASET
                    temp_loss = F.binary_cross_entropy(batch_pred, batch_true_y_task) 
                    task_losses[task] = temp_loss

                if loss_setting == "gradnorm" and batch_idx == 0:
                    # we need initial task losses for gradnorm
                    initial_task_losses = {key: loss.detach() for key, loss in task_losses.items()} # initial code detaches these
                    loss_weights = torch.ones(len(tasks))
                    loss_weights = nn.Parameter(loss_weights)
                    dprint(f"Initial Loss Weights {loss_weights}")
                    gradnorm_optimizer = torch.optim.Adam([loss_weights], lr=0.001) # this should be an input
                
                # TO DO - Below is pretty sloppy I think, see if we can make this nicer
                # Note the use of regularization in one task and not in multi tasks, per the original code (probably fix later)
                            
                # ON ONE TASK - USE THIS FOR SAME RESULT AS BINARY IN ORIGINAL CODE
                if len(tasks) == 1:
                    loss = task_losses[tasks[0]] + compute_l2_reg_val(model)

                # ON 4 MULTI TASKS WITH GIVEN WEIGHTS - USE THIS FOR SAME RESULT AS MULTI IN ORIGINAL CODE
                elif loss_setting == "predefined_weights" and "mature" in tasks and "gory" in tasks and "sarcasm" in tasks and "slapstick" in tasks and "binary" not in tasks:
                     loss = mature_w*task_losses["mature"] + gory_w*task_losses["gory"] + sarcasm_w*task_losses["sarcasm"] + slap_w*task_losses["slapstick"]

                # below two are untested
                # learned loss weightings for whichever ordering of tasks
                elif loss_setting == "weighted":
                    loss = 0
                    for task_loss in task_losses.values():
                        dprint(f"Unweighted Task Loss {task_loss}")
                        dprint(f"Loss Weight: {model.loss_weights[i]}")
                        dprint(f"Weighted Task Loss: {model.loss_weights[i] * task_loss}")
                        loss += model.loss_weights[i] * task_loss
                
                # CAN PUT GRADNORM HERE (THOUGH IT NEEDS STUFF EARLIER)
                # if i were to modularize, this would take in task losses as values - but it might not be that simple
                # It might be better to do this and training together in case of issues in optimization
                elif loss_setting == "gradnorm":
                    dprint("Beginning Grad Norm")
                    loss = 0
                    # passing in model.base_model here should be put elsewhere since it makes the training loop model_specific
                    # can enforce these task specific hyperparameters using **kwargs maybe?
                    # Make consistent the way we handle loss weights would be better
                    # Currently this returns weigts and does not update model.loss_weights
                    weights = gradnorm(task_losses, initial_task_losses, model, model.base_model, gradnorm_optimizer, loss_weights) # update model weights using grad norm
                    dprint(f"Updated Weights {weights}")
                    for i, task_loss in enumerate(task_losses.values()):
                        dprint(f"Task {i}")
                        dprint(f"Unweighted Task Loss {task_loss}")
                        dprint(f"Loss Weight: {weights[i]}")
                        dprint(f"Weighted Task Loss: {weights[i] * task_loss}\n\n")
                        loss += weights[i] * task_loss # currently weights just come from gradnorm and are aliged based on gradnorm implemetation
                    # then backward step like before

                # don't weight losses
                else:
                    loss = 0
                    for task_loss in task_losses.values():
                        loss += task_loss

                total_loss += loss.item()
                loss.backward(retain_graph=True) # We get problems with retain_graph=False with GradNorm - note possible issues (code I saw does this though)
                optimizer.step()
                print(f"Current Loss: {loss}")

            # CULLEN: INITIAL ROUND ROBIN IMPLEMENTATION
            # Take as input the task specific heads you wish to use
            # Iterate through each task and get the loss just for that output
            # Currently this assumes the dataset where each item has every output so we reuse batches

            # NOTE - THIS DOES 4 * MORE BACKPASSES THAN THE ORIGINAL MODEL
            # the only difference with original is we call backward here instead of adding up losses
            elif training_method == "round_robin":
                if batch_idx == 0 and epoch == 0:
                    print(f"Training Naive Round Robin on the following tasks: {[task for task in tasks]}")
                # kind of weird thing going on here
                # We need to compute the output seperately for every task in order to do backprop I believe
                for i, task in enumerate(tasks): 
                    optimizer.zero_grad() # we train on every task, so zero_grad on every task
                    batch_pred = model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, [tasks[i]])[:, 0, :]
                    # note above: batch_size by tasks by 2 -> we have to get rid of the task dimension for proper size i.e. batch_size by 2
                    batch_true_y_task = batch[task].to(device)
                    # NAN ERRORS -> FIND SOURCE OF ISSUE
                    try:
                        dprint(f"batch_text: {batch_text}")
                        dprint(f"batch_text type and size {type(batch_text), batch_text.shape}") # batch_size by 500 tensor
                        dprint(f"batch_text_mask: {batch_text_mask}")
                        dprint(f"batch_image: {batch_image}")
                        dprint(f"batch_mask_img: {batch_mask_img}")
                        dprint(f"batch_audio: {batch_audio}")
                        dprint(f"batch_mask_audio: {batch_mask_audio}")
                        dprint(f"batch_pred: {batch_pred}")
                        loss = F.binary_cross_entropy(batch_pred, batch_true_y_task)
                    except:
                        # Print the variables
                        dprint(f"batch_text: {batch_text}")
                        dprint(f"batch_text type and size {type(batch_text), batch_text.shape}")
                        dprint(f"batch_text_mask: {batch_text_mask}")
                        dprint(f"batch_image: {batch_image}")
                        dprint(f"batch_mask_img: {batch_mask_img}")
                        dprint(f"batch_audio: {batch_audio}")
                        dprint(f"batch_mask_audio: {batch_mask_audio}")
                        dprint(f"batch_pred: {batch_pred}")
                        break
                    loss.backward()
                    optimizer.step()
                    print(f"Task: {task}, Current Loss {loss}")

            # IDEA - could loss reweighting have a roll here at all?
            # combined idea - Loss reweighting on round robin instead of
            # We would train one task at a time but weight them differently

            # TO DO: This way of reporting loss doesn't make sense with round robin (have to divide total loss by 4 * batch_idx + 1)
            if (batch_idx + 1) % 10 == 0:
                # is this a normal way to report loss?
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {total_loss / (batch_idx + 1):.4f}, Time: {time.time() - start_time:.2f}s')

    return model


# IDEA - GradNorm (and adjacent techniques) can also be easily combined with dynamic difficulty sampling
# so we definetly need a modularized implementaiton


# they don't normalize losses, we probably should
# k should be 100, but 3 here for speed
def dynamic_difficulty_sampling(model, optimizer, json_data_path, tasks, k=2, loss_setting="unweighted", batch_size=32, num_epochs=1, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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
        datasets[task] = CustomDataset(json_data_path, text_pad_length, img_pad_length, audio_pad_length)

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
