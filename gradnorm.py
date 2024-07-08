import torch
import torch.nn as nn
import numpy as np

debug = True

def dprint(text):
    if debug:
        print(text)

# naively this seems to work - it runs and weights are indeed adjusted, losses are at a similar scale as expected
# be wary that there may be issues here related to the optimization
# also this code needs to be cleaned up a lot (e.g. how layer is handled)
def gradnorm(task_losses_dict, initial_task_losses_dict, model, layer, optimizer, loss_weights, alpha=1.5):
    # task_losses and inital_task_losses are dictionaries of task losses (pytorch tensors)
    # Convert to tensors to enable efficient computation
    # test below
    task_losses = torch.stack(list(task_losses_dict.values()))
    initial_task_losses = torch.stack(list(initial_task_losses_dict.values()))

    dprint(f"Task Losses {task_losses}")

    T = len(task_losses) # we normalize to this quantity

    dprint(f"INITIAL MODEL LOSS WEIGHTS: {loss_weights}")

    # first compute gradients for each task

    # Only 2 layers not involved are bert.pooler.dense.weight and bias (out of 288 layers)
    # check_task_dependencies(task_losses_dict, model, layer)

    gradient_norms = [] # G_W^i(t) in paper
    for i in range(len(task_losses)):
        # THE BELOW IS JUST FOR TESTING - THIS SHOULD BE HANDLED BY PASSING IN THE RIGHT STUFF
        parameters = list(layer.parameters())
        parameters = parameters[-10:]

        # need to handle layer as input
        dprint(f"Task {i}, Loss: {task_losses[i]}, Requires Grad: {task_losses[i].requires_grad}")
        task_gradient = torch.autograd.grad(loss_weights[i] * task_losses[i], parameters, retain_graph=True, create_graph=True, allow_unused=True)
        # the above is a list of tensors corresponding to every parameter

        # to use all gradients we can use below
        # We have gradients w.r.t. to every parameter that we will take the norm of, so I think this makes sese
        task_gradient = torch.cat([grad.view(-1) for grad in task_gradient if grad is not None])
        dprint(f"Gradient of weighted loss for task {i} w.r.t. layer: {task_gradient}, size: {task_gradient.shape}")

        task_gradient_norm = torch.norm(task_gradient)
        dprint(f"Gradient Norm: {task_gradient_norm}")

        gradient_norms.append(task_gradient_norm)

    gradient_norms = torch.stack(gradient_norms)

    dprint(f"Gradient Norms: {gradient_norms}")

    # compute average gradient
    gradients_norm_avg = gradient_norms.mean() # .detach() ?

    dprint(f"Gradient Norm Average: {gradients_norm_avg}")

    # compute these loss ratios using task_losses and initial task losses
    loss_ratios = task_losses / initial_task_losses # \hat{L_i(t)} in paper - this is coordinate wise division since we have the same sizes
    inverse_training_rates = loss_ratios / torch.mean(loss_ratios, dim=0) # take mean across tasks, presumably the first dimension but CHECK

    dprint(f"Loss Ratios: {loss_ratios}")
    dprint(f"Inverse Training Rates: {inverse_training_rates}")

    # compute the grad norm loss function
    constant = gradients_norm_avg * inverse_training_rates ** alpha # This should just be a single number
    grad_norm_loss = torch.abs(gradient_norms - constant).sum() # make sure this is across the right dimensions

    dprint(f"Grad Norm Loss {grad_norm_loss}")

    # per the implementation I found
    optimizer.zero_grad() # is this right to put here -> also not necessary if we initialize optimizer every time but this probably isn't right

    # Backward pass the gradient norm loss
    grad_norm_loss.backward(retain_graph=True) # POSSIBLE ISSUE: In original code, backwards on normal loss comes first, issues?

    # calling this backwards updates all gradients I believe
    # if we modularize we have to be careful about the effect on the other optimizer

    # Update model.loss_weights from backward pass
    optimizer.step() # we get gradients from grad norm here, and then we backprop them just to model weights
    
    dprint(f"LOSS WEIGHTS AFTER OPTIMIZATION:", loss_weights)

    with torch.no_grad():
        loss_weights = loss_weights / loss_weights.sum() * T

    dprint(f"LOSS WEIGHTS AFTER NORMALIZING {loss_weights}")

    # we might also want to return loss ratios to track progress later on
    return loss_weights

def check_task_dependencies(task_losses_dict, model, layer):
    for task_name, task_loss in task_losses_dict.items():
        dprint(f"Checking task: {task_name}")
        grads = torch.autograd.grad(task_loss, layer.parameters(), retain_graph=True, allow_unused=True)
        # Print information about each parameter and its gradient involvement
        for param, grad in zip(layer.named_parameters(), grads):
            param_name, param_tensor = param
            if grad is None:
                print(f"Layer parameter '{param_name}' with shape {param_tensor.shape} is not involved in the computation of {task_name}")
            else:
                print(f"Layer parameter '{param_name}' with shape {param_tensor.shape} is involved in the computation of {task_name}, gradient shape: {grad.shape}")


if __name__ == "__main__":
    class SimpleModel(nn.Module):
        def __init__(self, num_tasks):
            super(SimpleModel, self).__init__()
            self.loss_weights = nn.Parameter(torch.ones(num_tasks))
            self.layer0 = nn.Linear(10, 20) # just a sample layer
            self.layer1 = nn.Linear(20, 5)
        
        def forward(self, x):
            temp = self.layer0(x)
            return self.layer1(temp)

    num_tasks = 2
    model = SimpleModel(num_tasks)
    layer = model.layer1 

    # Initialize example inputs
    input0 = torch.randn(1, 10)
    input1 = torch.randn(1, 10)

    task_losses_dict = {
        'task1': torch.norm(model(input0) - torch.ones(5)),
        'task2': torch.norm(model(input0))
    }

    initial_task_losses_dict = {
        'task1': torch.norm(model(input1) - torch.ones(5)),
        'task2': torch.norm(model(input1))
    }

    # Perform the test
    updated_loss_weights = gradnorm(task_losses_dict, initial_task_losses_dict, model, layer)
    print("Updated loss weights:", updated_loss_weights)