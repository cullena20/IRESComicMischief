import torch
import torch.nn as nn
from torch.nn import functional as F
from Model.UnifiedModel import UnifiedModel
from Model.BaseModel import Bert_Model
from Model.TaskHeads import BinaryClassification, MultiTaskClassification
import os
import re


# NOTE ON PADDING AND MASKING:
# Text: pad 0s at beginning (so final result is of form [0, 0, ..., 12, 15]) and perform mask accordingly
# Audio and Image: Pad 0s at end and mask accordingly. Note that each token has an embedding dimension.
# this is because text is not yet encoded -> the padding will translate into the embedded output
# however audio and image are preencoded. So a 0 token for audio and image is really a row of 0s.

# the below is used in the training loop
# this might have issues since it really increases the loss (around 2 to around 530 in the beginning)
# this makes sense with how many parameters there are, but do we want this?
l2_regularize = True
l2_lambda = 0.1
def compute_l2_reg_val(model):
    if not l2_regularize:
        return 0.

    l2_reg = None

    for w in model.parameters():
        if l2_reg is None:
            l2_reg = w.norm(2)
        else:
            l2_reg = l2_reg + w.norm(2)

    return l2_lambda * l2_reg.item()

# the below are used in the dataloader to process the data
# this creates a mask with 1s at beginning and 0s at end
# this assumes that tokens are in the beginning
def mask_vector(max_size, arr):
    if arr.shape[0] > max_size:
        output = [1] * max_size
    else:
        len_zero_value = max_size - arr.shape[0]
        output = [1] * arr.shape[0] + [0] * len_zero_value
    return torch.tensor(output)

# this assumes tokens are at end and is used for text
def mask_vector_reverse(max_size, arr):
    if arr.shape[0] > max_size:
        output = [1] * max_size
    else:
        len_zero_value = max_size - arr.shape[0]
        output = [0] * len_zero_value + [1] * arr.shape[0]
    return torch.tensor(output)

# this is used for audio and image
# padding is done at end
# each padded token is a row because it has already been embedded
def pad_segment(feature, max_feature_len):
    S, D = feature.shape
    if S > max_feature_len:
        feature = feature[:max_feature_len]
    else:
        pad_l = max_feature_len - S
        pad_segment = torch.zeros((pad_l, D))
        feature = torch.concatenate((feature, pad_segment), axis=0)
    return feature

# this assumes padding at beginning and is used for text
def pad_features(docs_ints, text_pad_length=500):
    features = torch.zeros((len(docs_ints), text_pad_length), dtype=int)
    for i, row in enumerate(docs_ints):
        features[i, -len(row):] = row[:text_pad_length]
    return features

# TO DO:
# The loading pretraining code should really be put separately
# Also need to further adapt this to Sukruth's code (same big ideas, but the dictionary may need to be changed a little)
key_mapping = {
    'features.0': 'bert',
    'features.1': 'rnn_img',
    'features.2': 'rnn_img_drop_norm',
    'features.3': 'rnn_audio',
    'features.4': 'rnn_audio_drop_norm',
    'features.5': 'sequential_audio',
    'features.6': 'sequential_image',
    'features.7': 'att1',
    'features.8': 'att1_drop_norm1',
    'features.9': 'att1_drop_norm2',
    'features.10': 'att2',
    'features.11': 'att2_drop_norm1',
    'features.12': 'att2_drop_norm2',
    'features.13': 'att3',
    'features.14': 'att3_drop_norm1',
    'features.15': 'att3_drop_norm2',
    'features.16': 'attention',
    'features.17': 'attention_drop_norm',
    'features.18': 'attention_audio',
    'features.19': 'attention_audio_drop_norm',
    'features.20': 'attention_image',
    'features.21': 'attention_image_drop_norm'
}  

# handles transforming the keys from having a ModuleList (e.g. calling self.features[i])
# to our new more readable version where we call the modules directly
def transform_key(input_key):
    pattern = r'^features\.(\d+)\.(.*)$'
    match = re.match(pattern, input_key)
    if match:
        x = match.group(1)
        y = match.group(2)
        base_key = f'features.{x}'
        if base_key in key_mapping:
            return f'{key_mapping[base_key]}.{y}'
    return input_key

binary_tasks = ["binary"]
multi_tasks = ["mature", "gory", "sarcasm", "slapstick"]

multi_task_heads = {
        "mature": BinaryClassification(),
        "gory": BinaryClassification(),
        "sarcasm": BinaryClassification(),
        "slapstick": BinaryClassification()
    }

def initiate_pretrained_model(task_heads, debug=False):
    base_path = os.path.dirname(os.path.abspath(__file__))
    pretrained_weights_path = os.path.join(base_path, "checkpoint-pretraining/best_pretrain_matching.pth")
    base_model = Bert_Model()
    pretrained_state_dict = torch.load(pretrained_weights_path)
    pretrained_model_state = {transform_key(k): v for k, v in pretrained_state_dict["model_state"].items()}

    if debug:
        print(pretrained_state_dict.keys())
        print(pretrained_state_dict['epoch']) # this is just a number, I assume the epoch where the best training occurs at
        print(pretrained_state_dict['score']) # this is a number, it is 0.1 ?
        print(pretrained_state_dict['optimizer'].keys()) # State and Param Groups
        print(pretrained_state_dict['optimizer']["state"].keys()) # numbers 0 to 335
        print(pretrained_state_dict['optimizer']["state"][0])
        print(pretrained_state_dict['optimizer']["param_groups"]) # contains learning rate, betas, eps, weight decay, some other hyprparameters, and params (which is just numbers 0 to 335)
        print()

        # print the parameters and their shapes saved in the pretrained parameters
        # the model has extra task specific heads
        # simply set strict=False to deal with (this would also work if we had new task specific heads in our model - but some names might be different if we were to used UnifiedModel directly)
        # for param_tensor in pretrained_state_dict["model_state"]:
        #     print(f"{param_tensor}  {pretrained_state_dict["model_state"][param_tensor].size()}") # we need to replace things like features.0 with bert - use regex
        # print()

        # print the parameters and their shapes in our new base model
        for param_tensor in base_model.state_dict():
            print(f"{param_tensor}  {base_model.state_dict()[param_tensor].size()}")
        print()

        # print the converted parameters in the pretrained parameters - should work in our model now (and will have some extra parameters)
        for param_tensor in pretrained_model_state:
            print(f"{param_tensor}  {pretrained_model_state[param_tensor].size()}") # we need to replace things like features.0 with bert - use regex

    # Load the state dictionary into the model with strict=False
    missing_keys, unexpected_keys = base_model.load_state_dict(pretrained_model_state, strict=False)
    
    # Print missing and unexpected keys for verification
    if debug:
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
    
    print(f"Successfully loaded base model weights from {pretrained_weights_path}")

    unified_model = UnifiedModel(base_model, task_heads)
    return unified_model, base_model

# initiate model from scratch
def initiate_model_new(task_heads):
    base_model = Bert_Model()
    unified_model = UnifiedModel(base_model, task_heads)
    return unified_model, base_model


# THE BELOW IS CURRENTLY NOT USED BUT COULD NEED HELP


import sys
import time
import torch
import numpy as np

__all__ = ['TorchHelper']


class TorchHelper:
    checkpoint_history = []
    early_stop_monitor_vals = []
    best_score = 0
    best_epoch = 0

    def __init__(self):
        self.USE_GPU = torch.cuda.is_available()

    def show_progress(self, current_iter, total_iter, start_time, training_loss, additional_msg=''):
        bar_length = 50
        ratio = current_iter / total_iter
        progress_length = int(ratio * bar_length)
        percents = int(ratio * 100)
        bar = '[' + '=' * (progress_length - 1) + '>' + '-' * (bar_length - progress_length) + ']'

        current_time = time.time()
        # elapsed_time = time.gmtime(current_time - start_time).tm_sec
        elapsed_time = round(current_time - start_time, 0)
        estimated_time_needed = round((elapsed_time / current_iter) * (total_iter - current_iter), 0)

        sys.stdout.write(
            'Iter {}/{}: {} {}%  Loss: {} ETA: {}s, Elapsed: {}s, TLI: {} {} \r\r'.format(current_iter, total_iter, bar,
                                                                                       percents,
                                                                                       round(training_loss, 4),
                                                                                       estimated_time_needed,
                                                                                       elapsed_time,
                                                                                       np.round(
                                                                                           elapsed_time / current_iter,
                                                                                           3), additional_msg))

        if current_iter < total_iter:
            sys.stdout.flush()
        else:
            sys.stdout.write('\n')

    def checkpoint_model(self, model_to_save, optimizer_to_save, path_to_save="", current_score=0, epoch=0, mode='max'):
        """
        Checkpoints models state after each epoch.

        :param model_to_save:
        :param optimizer_to_save:
        :param path_to_save:
        :param current_score:
        :param epoch:
        :param n_epoch:
        :param mode:
        :return:
        """
        # if scheduler_to_save == None:
        model_state = {'epoch'      : epoch + 1,
                      'model_state': model_to_save.state_dict(),
                      'score'      : current_score,
                      'optimizer'  : optimizer_to_save.state_dict(),
                      'scheduler'  : None}
        # else:
        #   model_state = {'epoch'      : epoch + 1,
        #                 'model_state': model_to_save.state_dict(),
        #                 'score'      : current_score,
        #                 'optimizer'  : optimizer_to_save.state_dict(),
        #                 'scheduler'  : scheduler_to_save.state_dict()}

        # Save the model as a regular checkpoint
        torch.save(model_state, path_to_save + 'last.pth'.format(epoch))

        self.checkpoint_history.append(current_score)
        is_best = False

        # If the model is best so far according to the score, save as the best model state
        if ((np.max(self.checkpoint_history) == current_score and mode == 'max') or
                (np.min(self.checkpoint_history) == current_score and mode == 'min')):
            is_best = True
            self.best_score = current_score
            self.best_epoch = epoch
            # print('inside checkpoint', current_score, np.max(self.checkpoint_history))
            # torch.save(model_state, path_to_save + '{}_best.pth'.format(n_epoch))
            torch.save(model_state, path_to_save + 'best.pth')
            print('BEST saved')

        print('Current best', round(max(self.checkpoint_history), 4), 'after epoch {}'.format(self.best_epoch))

        return is_best

    def checkpoint_model_pretrain(self, model_to_save, optimizer_to_save, scheduler_to_save=None, path_to_save="", current_score=0, epoch=0, mode='max'):
        """
        Checkpoints models state after each epoch.

        :param model_to_save:
        :param optimizer_to_save:
        :param path_to_save:
        :param current_score:
        :param epoch:
        :param n_epoch:
        :param mode:
        :return:
        """
        model_state = {'epoch'      : epoch + 1,
                      'model_state': model_to_save.state_dict(),
                      'score'      : current_score,
                      'optimizer'  : optimizer_to_save.state_dict()}

        # Save the model as a regular checkpoint
        torch.save(model_state, path_to_save + 'last_pretrain_Hybrid2.pth'.format(epoch))

        self.checkpoint_history.append(current_score)
        is_best = False

        # If the model is best so far according to the score, save as the best model state
        if ((np.max(self.checkpoint_history) == current_score and mode == 'max') or
                (np.min(self.checkpoint_history) == current_score and mode == 'min')):
            is_best = True
            self.best_score = current_score
            self.best_epoch = epoch
            # print('inside checkpoint', current_score, np.max(self.checkpoint_history))
            # torch.save(model_state, path_to_save + '{}_best.pth'.format(n_epoch))
            torch.save(model_state, path_to_save + 'best_pretrain_Hybrid2.pth')
            print('BEST saved')

        print('Current best', round(max(self.checkpoint_history), 4), 'after epoch {}'.format(self.best_epoch))

        return is_best

    def checkpoint_model_contrastive_loss(self, model_to_save, optimizer_to_save, path_to_save="", current_score=0, epoch=0, mode='max'):
        """
        Checkpoints models state after each epoch.

        :param model_to_save:
        :param optimizer_to_save:
        :param path_to_save:
        :param current_score:
        :param epoch:
        :param n_epoch:
        :param mode:
        :return:
        """
        model_state = {'epoch'      : epoch + 1,
                      'model_state': model_to_save.state_dict(),
                      'score'      : current_score,
                      'optimizer'  : optimizer_to_save.state_dict()}

        # Save the model as a regular checkpoint
        torch.save(model_state, path_to_save + 'last_contrastive_loss.pth'.format(epoch))

        self.checkpoint_history.append(current_score)
        is_best = False

        # If the model is best so far according to the score, save as the best model state
        if ((np.max(self.checkpoint_history) == current_score and mode == 'max') or
                (np.min(self.checkpoint_history) == current_score and mode == 'min')):
            is_best = True
            self.best_score = current_score
            self.best_epoch = epoch
            # print('inside checkpoint', current_score, np.max(self.checkpoint_history))
            # torch.save(model_state, path_to_save + '{}_best.pth'.format(n_epoch))
            torch.save(model_state, path_to_save + 'best_contrastive_loss.pth')
            print('BEST saved')

        print('Current best', round(max(self.checkpoint_history), 4), 'after epoch {}'.format(self.best_epoch))

        return is_best

    def load_saved_model(self, model, path, optimizer=None):
        """
        Load a saved model from dump
        :return:
        """
        # self.active_model.load_state_dict(self.best_model_path)['model_state']
        checkpoint = torch.load(path)
        #print(checkpoint['epoch'])
        #print(checkpoint['score'])
        #print(checkpoint['optimizer'])
        #print(checkpoint['model_state'])
        model.load_state_dict(checkpoint['model_state'])
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("optimizer loaded!")

    def load_saved_optimizer_scheduler(self, optimizer, scheduler, path):
        """
        Load a saved optimizer
        :return:
        """
        # self.active_model.load_state_dict(self.best_model_path)['model_state']
        checkpoint = torch.load(path)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("optimizer loaded")
        # scheduler.load_state_dict(checkpoint['scheduler'])
        

# can probably delete all of below now
#### Temporary code to pinpoint NaN errors

# Define a function to recursively check for NaNs
def check_for_nan(tensor, module_name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {module_name}")

# Define a recursive function to handle different types of outputs
def handle_output(output, module_name):
    if isinstance(output, torch.Tensor):
        check_for_nan(output, module_name)
    elif isinstance(output, (tuple, list)):
        for i, out in enumerate(output):
            handle_output(out, f"{module_name}[{i}]")
    elif isinstance(output, dict):
        for key, value in output.items():
            handle_output(value, f"{module_name}['{key}']")
    elif hasattr(output, '__dict__'):  # For objects with attributes
        for attr, value in output.__dict__.items():
            handle_output(value, f"{module_name}.{attr}")

# Define the hook function
def hook_fn(module, input, output):
    handle_output(output, module)

# Define a function to register hooks to a model
def register_nan_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Module):
            module.register_forward_hook(hook_fn)

######
