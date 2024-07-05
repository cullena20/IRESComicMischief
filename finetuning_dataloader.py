import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time
import json
from helpers import pad_segment, mask_vector, pad_features, mask_vector_reverse

debug = False

def dprint(text):
    if debug:
        print(text)

# I think there's issues with the padding and masking (also doesn't padding set to -infinity)

# so here is the data we need to load
# sentences (tokenized)
# sentence mask
# image (processed by I3D)
# image mask
# audio (processed by VGG)
# audio_mask
# outputs (this is different binary or multi task)
# note that everything needs to be preprocessed

# this dataset will take in padding, sequence length, and directory and create dataset
# we are performing quite naive padding here, but our model seems to need it
# a better method would be to pad dynamically in the collate function, but this is a later concern

# everything is numpy arrays right now, should probably change to torch tensors

class CustomDataset(Dataset):
    def __init__(self, json_data, text_pad_length=500, img_pad_length=36, audio_pad_length=63):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        processed_data_dir = os.path.join(self.base_dir, "processed_data")
        full_json_data_path = os.path.join(processed_data_dir, json_data)
        # the above may be better to be modified

        self.data = json.load(open(full_json_data_path)) # data dictionary
        self.keys = list(self.data.keys()) # hack for __getitem__

        # define pad lengths for each modality
        # note that the pad length refers to number of tokens
        # so audio input is 128 dimensions for each token, but is padded to 36
        self.text_pad_length = text_pad_length 
        self.img_pad_length = img_pad_length
        self.audio_pad_length = audio_pad_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # here we process all the data accordingly
        key = self.keys[idx] # this is the name of the file
        item = self.data[key]
        
        # FIX THIS
        # NOTE THAT ORIGINAL CODE DOES ERROR HANDLING HERE
        image_path = os.path.join(self.base_dir, "i3D-vggish-features/i3d_vecs_extended_merged")

        # ERROR HANDLING BELOW NEEDS UNIFYING

        # Load image features
        image_path_rgb = os.path.join(image_path, f"{key}_rgb.npy")
        image_path_flow = os.path.join(image_path, f"{key}_flow.npy")
        # MODIFIED BELOW FOR TESTING, CAN CLEAN UP
        if os.path.isfile(image_path_rgb) and os.path.isfile(image_path_flow):
            try:
                a1 = np.load(image_path_rgb)
                a2 = np.load(image_path_flow)
                a1 = torch.tensor(a1)
                a2 = torch.tensor(a2)
                image_vec = a1 + a2
                masked_img = mask_vector(self.img_pad_length, image_vec)
                image_vec = pad_segment(image_vec, self.img_pad_length)
            except:
                dprint("Image not found")
                image_vec = torch.zeros((self.img_pad_length, 1024)) 
                masked_img = torch.zeros(self.img_pad_length)
        else:
            dprint("Image not found")
            image_vec = torch.zeros((self.img_pad_length, 1024)) 
            masked_img = torch.zeros(self.img_pad_length)

        dprint("PADDED IMAGE VEC")
        dprint(image_vec)
        dprint(f"IMAGE Masked {masked_img}")

        # Load audio features
        audio_path = os.path.join(self.base_dir, "i3D-vggish-features/vgg_vecs_extended_merged/")
        audio_path = audio_path+key+"_vggish.npy"

        try:
            audio_vec = np.load(audio_path)
            audio_vec = torch.tensor(audio_vec)
        except FileNotFoundError:
            dprint("Audio not found")
            audio_vec = torch.zeros((1, 128))

        masked_audio = mask_vector(self.audio_pad_length, audio_vec)
        audio_vec = pad_segment(audio_vec, self.audio_pad_length)

        dprint("PADDED AUDIO VEC")
        dprint(audio_vec)
        dprint(f"AUDIO Mask {masked_audio}")

        # Process text
        text = torch.tensor(item['indexes']) # tokenized text
        dprint(f"TEXT {text}")
        dprint(f"TEXT SHAPE {text.shape}")

        text_mask = mask_vector_reverse(self.text_pad_length, text)
        dprint(f"TEXT MASK {text_mask}")

        # NOTE: PADDING AT BEGINNING FOR SOME REASON
        text = pad_features([text], self.text_pad_length)[0]
        dprint(f"PADDED TEXT {text}")

        binary = torch.tensor(item['y']) 
        mature = torch.tensor(item["mature"])
        gory = torch.tensor(item["gory"])
        sarcasm = torch.tensor(item["sarcasm"])
        slapstick = torch.tensor(item["slapstick"])

        return {
            'text': text,
            'text_mask': text_mask,
            'image': image_vec.float(),
            'image_mask': masked_img,
            'audio': audio_vec.float(),
            'audio_mask': masked_audio,
            'binary': binary.float(),
            "mature": mature.float(),
            "gory": gory.float(),
            "sarcasm": sarcasm.float(),
            "slapstick": slapstick.float()
        }


if __name__ == "__main__":
    dataset = CustomDataset("test_features_lrec_camera.json")
    idx = 0
    for item in dataset:
        if idx == 1:
            break
        for key, value in item.items():
            print(key)
            print(value)
            print(value.shape)
            print()
        idx += 1

# POSSIBLE ISSUE
# TOKENS ARE ALL 0 UNTIL END OF TEXT
# BUT MASK HAS THE FIRST THINGS AS 1
# NOTE THIS IF FUTURE ERRORS

# BE WARY OF HOW PADDING AND MASKING IS DONE -> WILL NEED TO VERIFY LATER