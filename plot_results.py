import pickle
from plotters import plot_all_validation_results, plot_task_specific, plot_total
import torch
import os

def unpickle(pickle_path):
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)

    loss_history = results["loss_history"]
    task_loss_history = results["task_loss_history"]
    validation_results = results["validation_results"]

    return loss_history, task_loss_history, validation_results

def plot_results(loss_history, task_loss_history, validation_results, name=None, plot=False, save=False):
    total_loss_figure = plot_total(loss_history, xlabel="Steps", ylabel="Loss")
    task_loss_figure = plot_task_specific(task_loss_history, xlabel="Steps", ylabel="Loss")
    val_figures = plot_all_validation_results(validation_results)

    if plot:
        total_loss_figure.show()
        task_loss_figure.show()
    if save:
        dir_path = f"Results/{name}"
        os.makedirs(dir_path, exist_ok=True)
        total_loss_figure.savefig(f"{dir_path}/total_loss.png")
        task_loss_figure.savefig(f"{dir_path}/task_loss.png")

    for fig_name, fig in val_figures.items():
        if plot:
            fig.show()
        if save:
            fig.savefig(f"{dir_path}/{fig_name}")

    if save:
        print(f"All plots saved to {dir_path}")

def unpickle_and_plot(pickle_path, name=None, plot=False, save=False):
    loss_history, task_loss_history, validation_results = unpickle(pickle_path)
    plot_results(loss_history, task_loss_history, validation_results, name=name, plot=plot, save=save)

if __name__ == "__main__":
    pickle_path = "recreate_multitask_10epoch.pkl"
    unpickle_and_plot(pickle_path, "Recreate10Epochs", plot=False, save=True)