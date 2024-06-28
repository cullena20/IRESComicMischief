import torch
from torch.nn import functional as F

# the below is used in the training loop
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
def mask_vector(max_size, arr):
    if arr.shape[0] > max_size:
        output = [1] * max_size
    else:
        len_zero_value = max_size - arr.shape[0]
        output = [1] * arr.shape[0] + [0] * len_zero_value
    return torch.tensor(output)

def pad_segment(feature, max_feature_len):
    S, D = feature.shape
    if S > max_feature_len:
        feature = feature[:max_feature_len]
    else:
        pad_l = max_feature_len - S
        pad_segment = torch.zeros((pad_l, D))
        feature = torch.concatenate((feature, pad_segment), axis=0)
    return feature

# NOTE: BELOW MODIFIED FROM ORIGINAL CODE TO PAD ENDING
# BEFORE IT ADDED PAD VALUES TO THE BEGINNING
# def pad_features(feature, text_pad_length=500, pad_value=0):
#     current_length = feature.size(0)
#     if current_length >= text_pad_length:
#         return feature
#     pad_amount = text_pad_length - current_length
#     return F.pad(feature, (0, pad_amount), 'constant', pad_value)

# ORIGINAL PADS ENDING
# I think that this is done because the mask is handled very strangely in the model
def pad_features(docs_ints, seq_length=500):

    # getting the correct rows x cols shape
    features = torch.zeros((len(docs_ints), seq_length), dtype=int)

    for i, row in enumerate(docs_ints):
        features[i, -len(row):] = torch.tensor(row)[:seq_length]

    return features