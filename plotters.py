import matplotlib.pyplot as plt
import numpy as np
import torch

# plot results

# loss_history, task_loss_history, validation_results

# First - x value is steps 
# Plot loss history (task specific and weighted sum)

# Can use this to plot individual plots for several tasks
# use for task loss (training or val) and accuracy and f1 scores
def plot_task_specific(task_history, xlabel="Steps", ylabel="Loss", show=False):
    """
    Plots the histories for several tasks in a grid layout.
    
    Parameters:
    task_history (dict): A dictionary where keys are task names and values are torch tensors of loss histories.
    """
    # Convert dictionary values to numpy arrays for plotting
    for task in task_history:
        task_history[task] = task_history[task].detach().cpu().numpy() if isinstance(task_history[task], torch.Tensor) else task_history[task]

    # Determine the number of tasks
    num_tasks = len(task_history)
    
    # Determine grid size (rows and columns)
    num_cols = int(np.ceil(np.sqrt(num_tasks)))
    num_rows = int(np.ceil(num_tasks / num_cols))
    
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    # Plot each task loss history
    for idx, (task_name, loss_history) in enumerate(task_history.items()):
        axes[idx].plot(loss_history)
        axes[idx].set_title(f"{ylabel} History Task: {task_name}")
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_ylabel(f"Task {ylabel}")
    
    # Remove any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()

    if show:
        plt.show()

    return fig

# use this wherever we need to plot something for all tasks
def plot_total(total_history, xlabel="Steps", ylabel="Loss", show=False):
    total_history = total_history.detach().cpu().numpy() if isinstance(total_history, torch.Tensor) else total_history

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(total_history)
    ax.set_title(f"Total {ylabel} History")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Total {ylabel}")

    plt.tight_layout()

    if show:
        plt.show()
    
    return fig

def plot_all_validation_results(validation_results):
    """
    Plots the validation results including task-specific accuracies, f1_scores, and task losses,
    as well as the overall average accuracy, f1_score, and total loss.

    Parameters:
    validation_results (dict): Dictionary containing validation results with accuracies, f1_scores, and losses.
    """
    # Plot task-specific accuracies
    accuracies_fig = plot_task_specific(validation_results['accuracies'], xlabel="Epochs", ylabel="Accuracy")
    
    # Plot task-specific f1_scores
    f1_scores_fig = plot_task_specific(validation_results['f1_scores'], xlabel="Epochs", ylabel="F1 Score")
    
    # Plot task-specific losses
    task_losses_fig = plot_task_specific(validation_results['val_average_task_loss'], xlabel="Epochs", ylabel="Validation Loss")
    
    # Plot total average accuracy
    average_accuracy_fig = plot_total(validation_results['average_accuracy'], xlabel="Epochs", ylabel="Average Accuracy")
    
    # Plot total average f1_score
    average_f1_score_fig = plot_total(validation_results['average_f1_score'], xlabel="Epochs", ylabel="Average F1 Score")
    
    # Plot total validation loss
    total_loss_fig = plot_total(validation_results['val_average_total_loss'], xlabel="Epochs", ylabel="Validation Loss")
    
    return {
        'accuracies': accuracies_fig,
        'f1_scores': f1_scores_fig,
        'val_task_losses': task_losses_fig,
        'average_accuracy': average_accuracy_fig,
        'average_f1_score': average_f1_score_fig,
        'total_loss': total_loss_fig
    }


# Second - x value is epochs
# Plot accuracy history (plot specific and total)
# Plot f1 scores (specific and total)
# Plot validation loss (task specific and tota;)