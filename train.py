import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from finetuning_dataloader import CustomDataset
from BaseModel import Bert_Model
from TaskHeads import BinaryClassification, MultiTaskClassification
from UnifiedModel import UnifiedModel

import time

# Shapes and everything are working here, but not sure if this is actually working.

# currently this has no  lr_scheduler, or checkpoint modeling which the original does
# also does not have evaluation right now

# this functino maybe should be elsewhere
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

# MULTI TASK NOT IMPLEMENTED IN DATA LOADER
def train(model, optimizer, json_data, task, batch_size=32, num_epochs=1, text_pad_length=500, img_pad_length=36, audio_pad_length=63):
    # the dataset has x values: 
    # text, text_mask, image, image_mask, audio, audio_maxk
    # and y value (deal with depending on task)
    dataset = CustomDataset(json_data, text_pad_length, img_pad_length, audio_pad_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("CREATED DATASET AND DATALOADER")
    print()

    model.train()
    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        start_time = time.time() 
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # print(f"BATCH NUMBER: {batch_idx}")
            optimizer.zero_grad()

            batch_text = batch['text'].to(device)
            batch_text_mask = batch['text_mask'].to(device)
            batch_image = batch['image'].float().to(device)
            batch_mask_img = batch['image_mask'].to(device)
            batch_audio = batch['audio'].float().to(device)
            batch_mask_audio = batch['audio_mask'].to(device)

            batch_y = batch['label'].to(device).float()

            # print(f"batch_text shape: {batch_text.shape}")
            # print(f"batch_text_mask shape: {batch_text_mask.shape}")
            # print(f"batch_image shape: {batch_image.shape}")
            # print(f"batch_mask_img shape: {batch_mask_img.shape}")
            # print(f"batch_audio shape: {batch_audio.shape}")
            # print(f"batch_mask_audio shape: {batch_mask_audio.shape}")

            out = model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, task)
            # print(f"MODEL OUTPUT SHAPE: {out.shape}") batch_size by 2 for binary
            # print(f"OUTPUT: {out}")
            # print(out.dtype, batch_y.dtype) fixed this -> torch.float 32

            loss = F.binary_cross_entropy(out, batch_y) + compute_l2_reg_val(model)
            total_loss += loss.item()
            print(f"LOSS: {total_loss}")

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {total_loss / (batch_idx + 1):.4f}, Time: {time.time() - start_time:.2f}s')

    return model

# Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model = Bert_Model()
task_heads = {
    "binary": BinaryClassification(),
    "multi": MultiTaskClassification()
}
model = UnifiedModel(base_model, task_heads)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, optimizer, "train_features_lrec_camera.json", "binary", batch_size=8, num_epochs=10)