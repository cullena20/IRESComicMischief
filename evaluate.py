import torch
from torch.utils.data import DataLoader
from finetuning_dataloader import CustomDataset
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score

# the evaluation functions need refactoring using our data loader
# the bulk of the code is repetition with data loader stuff, but we've cleaned that up

# Further adapt to allow for more evaluation methods
# This does main tasks of returning accuracy and F1 score but not really loss or confusion matrices
def evaluate(model, json_data, tasks, batch_size=32, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    dataset = CustomDataset(json_data, text_pad_length, img_pad_length, audio_pad_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    model.eval()

    total_loss = 0 

    all_labels = {} # dictionary to store model predictions (binary label) for every task
    all_true_labels = {} # dictionary to store true labels for every task
    for task in tasks:
        all_labels[task] = [] # each task will have a list which we append predictions (binary labels) to
        all_true_labels[task] = []
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

            task_losses = {}   
            for task in tasks:
                # sort the loss for every task into a dictionary
                # note that original code accumulates total loss, we can do that later if we wish
                batch_pred = out_dict[task]
                batch_true_y_task = batch[task].to(device)
                temp_loss = F.binary_cross_entropy(batch_pred, batch_true_y_task) # not used
                task_losses[task] = temp_loss # not used

                # each of these has labels for a batch
                batch_label = (batch_pred[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                batch_true_label = (batch_true_y_task[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                all_labels[task].extend(batch_label)
                all_true_labels[task].extend(batch_true_label)

            # for debugging report performance over 20 batches
            if batch_idx == 2:
                break
    
    # Calculate accuracy and F1 score

    accuracies = {}
    f1_scores = {}

    # not dealing with loss currently
    # NOTE We are just averaging F1 scores for everything - equivalent to macro
    for task in tasks:
        print(f"Task {task}")
        print(all_labels[task])
        print(all_true_labels[task])
        accuracies[task] = accuracy_score(all_labels[task], all_true_labels[task])
        f1_scores[task] = f1_score(all_labels[task], all_true_labels[task], average='binary')

        print(f"Accuracy: {accuracies[task]}, F1 Score: {f1_scores[task]:.4f}")
    
    return accuracies, f1_scores
