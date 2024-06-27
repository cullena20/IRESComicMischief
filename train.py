import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from finetuning_dataloader import CustomDataset

from helpers import compute_l2_reg_val

import time

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

# MULTI TASK NOT IMPLEMENTED IN DATA LOADER
# we should be able to mix up tasks here
# for now lets have the dataset return all labels
# then task will determine how to evaluate and train
# this can be further modularized

# input list of tasks
# ["binary", "mature", "gory", "sarcasm", "slapstick"]
# Currently works on binary and the specific subset of multitask

# double check placement of zero_grad makes sense
def train(model, optimizer, json_data, tasks, training_method="all_at_once", batch_size=32, num_epochs=1, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # the dataset has x values: 
    # text, text_mask, image, image_mask, audio, audio_maxk
    # and y value (deal with depending on task)
    dataset = CustomDataset(json_data, text_pad_length, img_pad_length, audio_pad_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print("CREATED DATASET AND DATALOADER")
    print()

    model.train()
    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        start_time = time.time() 
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            # print(f"BATCH NUMBER: {batch_idx}")

            batch_text = batch['text'].to(device)
            batch_text_mask = batch['text_mask'].to(device)
            batch_image = batch['image'].float().to(device)
            batch_mask_img = batch['image_mask'].to(device)
            batch_audio = batch['audio'].float().to(device)
            batch_mask_audio = batch['audio_mask'].to(device)

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
                
                # TO DO - Below is pretty sloppy I think, see if we can make this nicer
                # Note the use of regularization in one task and not in multi tasks, per the original code (probably fix later)
                            
                # ON ONE TASK - USE THIS FOR SAME RESULT AS BINARY IN ORIGINAL CODE
                if len(tasks) == 1:
                    loss = task_losses[tasks[0]] + compute_l2_reg_val(model)

                # ON 4 MULTI TASKS WITH GIVEN WEIGHTS - USE THIS FOR SAME RESULT AS MULTI IN ORIGINAL CODE
                if "mature" in tasks and "gory" in tasks and "sarcasm" in tasks and "slapstick" in tasks and "binary" not in tasks:
                    loss = mature_w*task_losses["mature"] + gory_w*task_losses["gory"] + sarcasm_w*task_losses["sarcasm"] + slap_w*task_losses["slapstick"]

                # learned loss weightings for whichever ordering of tasks
                else:
                    for i, task_loss in enumerate(task_losses):
                        loss += model.loss_weights[i] * task_loss

                total_loss += loss.item() # not sure the point of this
                loss.backward()
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
                        loss = F.binary_cross_entropy(batch_pred, batch_true_y_task)
                    except:
                        # Print the variables
                        print(f"batch_text: {batch_text}")
                        print(f"batch_text_mask: {batch_text_mask}")
                        print(f"batch_image: {batch_image}")
                        print(f"batch_mask_img: {batch_mask_img}")
                        print(f"batch_audio: {batch_audio}")
                        print(f"batch_mask_audio: {batch_mask_audio}")
                        print(f"batch_pred {batch_pred}")
                        break
                    loss.backward()
                    optimizer.step()
                    print(f"Task: {task}, Current Loss {loss}")

            # TO DO: This way of reporting loss doesn't make sense with round robin (have to divide total loss by 4 * batch_idx + 1)
            if (batch_idx + 1) % 10 == 0:
                # is this a normal way to report loss?
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {total_loss / (batch_idx + 1):.4f}, Time: {time.time() - start_time:.2f}s')

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
