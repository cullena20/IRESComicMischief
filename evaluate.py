import torch
from torch.utils.data import DataLoader
from finetuning_dataloader import CustomDataset
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
import functools

# currently this returns a lot of stuff that may not be necessary, but it is nice for now
def evaluate(model, json_data, tasks, loss_weights=None, batch_size=16, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    dataset = CustomDataset(json_data, text_pad_length, img_pad_length, audio_pad_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    model.eval()

    steps = 0
    val_task_loss_history = {task: torch.zeros(len(dataloader)) for task in tasks}
    val_total_loss_history = torch.zeros(len(dataloader))

    all_labels = {} # dictionary to store model predictions (binary label) for every task
    all_true_labels = {} # dictionary to store true labels for every task
    for task in tasks:
        all_labels[task] = [] # each task will have a list which we append predictions (binary labels) to
        all_true_labels[task] = []

    if loss_weights is None:
        loss_weights = torch.ones(len(tasks))
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_text = batch['text'].to(device)
            batch_text_mask = batch['text_mask'].to(device)
            batch_image = batch['image'].float().to(device)
            batch_mask_img = batch['image_mask'].to(device)
            batch_audio = batch['audio'].float().to(device)
            batch_mask_audio = batch['audio_mask'].to(device)
            batch_pred = batch["binary"].to(device) # batch_size by 2

            # batch_size by 2 for binary
            # batch size by 4 by 2 for multi task
            out = model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, tasks)

            out_dict = {}  # store output of model for each task
            for i, task in enumerate(tasks):
                # sort the output for every task into a dictionary
                # each entry will be batch_size by 2 (for binary classification heads)
                out_dict[task] = out[:, i, :]

            total_loss = 0
            for i, task in enumerate(tasks):
                # sort the loss for every task into a dictionary
                # note that original code accumulates total loss, we can do that later if we wish
                batch_pred = out_dict[task]
                batch_true_y_task = batch[task].to(device)
                temp_loss = F.binary_cross_entropy(batch_pred, batch_true_y_task) # not used
                
                # NOTE: for now this assumes loss weights are handed in as a parameter and not part of the model
                total_loss += loss_weights[i] * temp_loss
                
                val_task_loss_history[task][steps] = loss_weights[i] * temp_loss

                # each of these has labels for a batch
                batch_label = (batch_pred[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                batch_true_label = (batch_true_y_task[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                all_labels[task].extend(batch_label)
                all_true_labels[task].extend(batch_true_label)

            val_total_loss_history[steps] = total_loss

            steps += 1

            # CHANGE FOR REAL EXPERIMENT for debugging report performance over 4 batches
            # if batch_idx == 4:
            #     break
    
    # Calculate accuracy and F1 score

    accuracies = {}
    f1_scores = {}

    # not dealing with loss currently
    # NOTE We are just averaging F1 scores for everything - equivalent to macro
    for task in tasks:
        # print(f"Task {task}")
        # print(f"Number of items: {len(all_labels[task])}")
        # print(all_labels[task])
        # print(all_true_labels[task])
        accuracies[task] = accuracy_score(all_labels[task], all_true_labels[task])
        f1_scores[task] = f1_score(all_labels[task], all_true_labels[task], average='macro') # macro

        # print(f"Accuracy: {accuracies[task]}, F1 Score: {f1_scores[task]:.4f}")

    # Just average out the f1 scores, reduce is a fun way to do this
    average_f1_score = functools.reduce(lambda a, b: a + b, f1_scores.values(), 0) / len(tasks)
    average_accuracy = functools.reduce(lambda a, b: a + b, accuracies.values(), 0) / len(tasks)

    # return average loss. I think this makes sense
    # using [:steps] is only important when we halt validation early for debugging
    # otherwise steps should equal len(dataloader)
    val_average_total_loss = val_total_loss_history.mean()
    val_average_task_loss = {task: loss_history.mean() for task, loss_history in val_task_loss_history.items()}

    # TO RETURN
    # Accuracies and F1 Scores for every task
    # Average F1 score (should this be weighted somehow)
    # Total Loss and Task Loss History (Total loss only makes sense when you add up losses)
    # Also all predictions and true labels in case we want to do anything more, but this is probably unnecessary
    return accuracies, f1_scores, average_accuracy, average_f1_score, val_average_total_loss, val_average_task_loss, all_labels, all_true_labels