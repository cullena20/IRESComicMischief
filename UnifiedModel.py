import torch
from torch import nn
from torch.nn import functional as F
from BaseModel import Bert_Model
from TaskHeads import BinaryClassification, MultiTaskClassification

class UnifiedModel(nn.Module):

    def __init__(self, base_model, task_head):
        super(UnifiedModel, self).__init__()
        self.base_model = base_model
        self.task_head = task_head

    def forward(self, x):
        # using Bert_Model, this should be text, audio, image embeddings concatenated (768 * 3)
        embedding = self.base_model(x)
        output = self.task_head(embedding)
        return output

base_model = Bert_Model()
binary_head = BinaryClassification()
multi_head = MultiTaskClassification()

BinaryModel = UnifiedModel(base_model, binary_head)
MultiTaskModel = UnifiedModel(base_model, multi_head)
