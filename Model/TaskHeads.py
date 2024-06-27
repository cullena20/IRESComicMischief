import torch
from torch import nn
from torch.nn import functional as F

class BinaryClassification(nn.Module):

    def __init__(self):
        super(BinaryClassification, self).__init__()

        # Input is text, audio, image embeddings concatenated
        self.text_audio_image_linear = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2) # why 2?
        )

    def forward(self, x):
        output = F.softmax(self.text_audio_image_linear(x), -1)
        return output


class MultiTaskClassification(nn.Module):

    def __init__(self):
        super(MultiTaskClassification, self).__init__()

        self.text_audio_image_linear_mature = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

 
        self.text_audio_image_linear_gory = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

        self.text_audio_image_linear_slapstick = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

        self.text_audio_image_linear_sarcasm = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

    def forward(self, x):
        output_mature = F.softmax(self.text_audio_image_linear_mature(x), -1)
        output_gory = F.softmax(self.text_audio_image_linear_gory(x), -1)
        output_slapstick = F.softmax(self.text_audio_image_linear_slapstick(x), -1)
        output_sarcasm = F.softmax(self.text_audio_image_linear_sarcasm(x), -1)
        
        # old return value is list: [output_mature, output_gory, output_slapstick, output_sarcasm]
        # I changed the order slightly to follow same order as data, hopefully no issues
        return torch.stack([output_mature, output_gory, output_sarcasm, output_slapstick], dim=1)