import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from finetuning_dataloader import CustomDataset

from helpers import compute_l2_reg_val

import time

# Shapes and everything are working here, but not sure if this is actually working.

# I also got an issue on one batch, so there are some datapoints with issues
# not all values were 0 or 1 - some were nan
# the other time same issue but it had an error that we are backpassing twice

# currently this has no  lr_scheduler, or checkpoint modeling which the original does
# also does not have evaluation right now

# where did these come from? supposed to be learnable
mature_w = 0.1
gory_w = 0.4
slap_w = 0.2
sarcasm_w = 0.2

# MULTI TASK NOT IMPLEMENTED IN DATA LOADER
# we should be able to mix up tasks here
# for now lets have the dataset return all labels
# then task will determine how to evaluate and train
# this can be further modularized
def train(model, optimizer, json_data, task, batch_size=32, num_epochs=1, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # the dataset has x values: 
    # text, text_mask, image, image_mask, audio, audio_maxk
    # and y value (deal with depending on task)
    dataset = CustomDataset(json_data, text_pad_length, img_pad_length, audio_pad_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
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

            # the output will depend on the task
            # in reality this isn't really necessary, but keep it for now and see what we need later
            out = model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, task)
                
            if task == "binary":
                batch_y = batch['binary'].to(device)

                # print(f"batch_text shape: {batch_text.shape}")
                # print(f"batch_text_mask shape: {batch_text_mask.shape}")
                # print(f"batch_image shape: {batch_image.shape}")
                # print(f"batch_mask_img shape: {batch_mask_img.shape}")
                # print(f"batch_audio shape: {batch_audio.shape}")
                # print(f"batch_mask_audio shape: {batch_mask_audio.shape}")

                # print(f"MODEL OUTPUT SHAPE: {out.shape}") batch_size by 2 for binary
                # print(f"OUTPUT: {out}")
                # print(out.dtype, batch_y.dtype) fixed this -> torch.float 32

                loss = F.binary_cross_entropy(out, batch_y) + compute_l2_reg_val(model)
                total_loss += loss.item()
                print(f"TOTAL LOSS: {total_loss} CURRENT LOSS: {loss}")

            elif task == "multi":
                mature_pred = out[:, 0, :]
                gory_pred = out[:, 1, :]
                slapstick_pred = out[:, 2, :]
                sarcasm_pred = out[:, 3, :]

                batch_mature = batch["mature"].to(device)
                batch_gory = batch["gory"].to(device)
                batch_sarcasm = batch["sarcasm"].to(device)
                batch_slapstick = batch["slapstick"].to(device)

                try:
                    loss1 = F.binary_cross_entropy(mature_pred, batch_mature) 
                    loss2 = F.binary_cross_entropy(gory_pred, batch_gory)
                    loss3 = F.binary_cross_entropy(sarcasm_pred, batch_sarcasm)
                    loss4 = F.binary_cross_entropy(slapstick_pred, batch_slapstick)
                except:
                    print(mature_pred)
                    print(batch_mature)

                # original code does below for total loss
                # total_loss += loss1.item() + loss2.item() + loss3.item() + loss4.item()
                
                # in original code they did slap_w ** loss - oops
                loss = mature_w*loss1 + gory_w*loss2 + sarcasm_w*loss3 + slap_w*loss4
                total_loss += loss

                print(f"TOTAL LOSS: {total_loss} CURRENT LOSS: {loss}")

                # print(f"OUTPUT SHAPE: {out.shape}")
                # print("Batch Mature Shape:", batch_mature.shape)
                # print("Batch Gory Shape:", batch_gory.shape)
                # print("Batch Sarcasm Shape:", batch_sarcasm.shape)
                # print("Batch Slapstick Shape:", batch_slapstick.shape)

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                # is this a normal way to report loss?
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {total_loss / (batch_idx + 1):.4f}, Time: {time.time() - start_time:.2f}s')

    return model

if __name__ == "__main__":
    from BaseModel import Bert_Model
    from TaskHeads import BinaryClassification, MultiTaskClassification
    from UnifiedModel import UnifiedModel
    # Usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model = Bert_Model()
    task_heads = {
        "binary": BinaryClassification(),
        "multi": MultiTaskClassification()
    }
    model = UnifiedModel(base_model, task_heads)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train(model, optimizer, "train_features_lrec_camera.json", "binary", batch_size=8, num_epochs=10)
    train(model, optimizer, "train_features_lrec_camera.json", "multi", batch_size=8, num_epochs=10, shuffle=False)