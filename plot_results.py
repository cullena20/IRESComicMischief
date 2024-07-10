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
    strategy_results = results["strategy_results"]

    return loss_history, task_loss_history, validation_results, strategy_results

# how strategy results are handled requires adding new code for each strategy
def plot_results(loss_history, task_loss_history, validation_results, strategy_results, name=None, plot=False, save=False, plot_strategy_results=False):
    total_loss_figure = plot_total(loss_history, xlabel="Steps", ylabel="Loss")
    task_loss_figure = plot_task_specific(task_loss_history, xlabel="Steps", ylabel="Loss")
    val_figures = plot_all_validation_results(validation_results)

    strategy_figures = {}
    for strategy_name, results in strategy_results.items():
        if strategy_name == "loss_weight_history":
            loss_weight_figure = plot_task_specific(results, xlabel="Steps", ylabel="Loss Weights")
            strategy_figures[strategy_name] = loss_weight_figure
        elif strategy_name == "sample_weight_history":
            sample_weight_figure = plot_task_specific(results, xlabel="Steps", ylabel="Sample Weights")
            strategy_figures[strategy_name] = sample_weight_figure

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
            fig.savefig(f"{dir_path}/{fig_name}.png")

    for fig_name, fig in strategy_figures.items():
        if plot:
            fig.show()
        if save:
            fig.savefig(f"{dir_path}/{fig_name}.png")

    if save:
        print(f"All plots saved to {dir_path}")

# something like this?
# def prepare_results_for_plotting(**kwargs):
#     prepared_result_dict = {}
#     for name, result in kwargs.items():
#         if type(result) is dict:
#             prepared_result = {key: value.detach().cpu() for key, value in result.items()}
#         else:
#             prepared_result = result.detach().cpu()
#         prepared_result_dict[name] = prepared_result
#     return prepared_result_dict


def unpickle_and_plot(pickle_path, name=None, plot=False, save=False):
    loss_history, task_loss_history, validation_results, strategy_results = unpickle(pickle_path)

    print(f"loss history {loss_history}")
    print(f"task loss history {task_loss_history}")
    print(f"validation results {validation_results}")
    print(f"strategy results {strategy_results}")

    plot_results(loss_history, task_loss_history, validation_results, strategy_results, name=name, plot=plot, save=save)

if __name__ == "__main__":
    # pickle_path = "/usuarios/arnold.moralem/IRESComicMischief/Results/Test/Test.pkl"
    # name = "Test"
    # unpickle_and_plot(pickle_path, name, plot=False, save=True)

    # pickle_path = "/usuarios/arnold.moralem/IRESComicMischief/Results/GradNorm10/GradNorm10.pkl"
    # name = "GradNorm10"
    # unpickle_and_plot(pickle_path, name, plot=False, save=True)

    # pickle_path = "/usuarios/arnold.moralem/IRESComicMischief/Results/Test2/Test2.pkl"
    # name = "Test2"
    # unpickle_and_plot(pickle_path, name, plot=False, save=True)

    # pickle_path = "/usuarios/arnold.moralem/IRESComicMischief/Results/mature/mature.pkl"
    # name = "mature"
    # unpickle_and_plot(pickle_path, name, plot=False, save=True)

    pickle_path = "/usuarios/arnold.moralem/IRESComicMischief/Results/DynamicDifficultySampling10/DynamicDifficultySampling10.pkl"
    name = "mature"
    unpickle_and_plot(pickle_path, name, plot=False, save=False)