import torch
from torch import nn

debug=False

def dprint(text):
    if debug:
        print(text)

# There may be an issue here since I can getting really high loss after making this change
class UnifiedModel(nn.Module):
    def __init__(self, base_model, task_heads):
        super(UnifiedModel, self).__init__()
        self.base_model = base_model
        self.task_heads = nn.ModuleDict(task_heads)

        # enable dynamic reweighting of losses
        # For simplicity: loss_weights[0]: binary, 1: mature, 2: gory, 3: sarcasm, 4: slapstick
        # Might change later on
        self.loss_weights = nn.Parameter(torch.ones(len(task_heads)))

    def forward(self, text_tokens, text_mask, image, image_mask, audio, audio_mask, tasks):
        shared_output = self.base_model(text_tokens, text_mask, image, image_mask, audio, audio_mask)
        dprint(shared_output)
        task_output = torch.stack([self.task_heads[task](shared_output) for task in tasks], dim=1)
        return task_output


# we don't do this in below paradigm anymore, instead using binary classification head for every task
if __name__ == "__main__":
    from BaseModel import Bert_Model
    from TaskHeads import BinaryClassification, MultiTaskClassification

    base_model = Bert_Model()
    task_heads = {
        "binary": BinaryClassification(),
        "multi": MultiTaskClassification()
    }
    unified_model = UnifiedModel(base_model, task_heads)

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model.to(device)
    unified_model.to(device)

    # Define input shapes 
    batch_size = 8
    sequence_length_text = 20 # not totally sure where numbers except batch_size and embedding_dim come from, check it
    sequence_length_image = 10
    sequence_length_audio = 15
    input_size_image = 1024 # I3D feature size
    input_size_audio = 128 # VGGish feature size
    embedding_dim = 768

    # Create random inputs and move them to the appropriate device
    # Sentences is going to be BERT tokenized sentences 
    text_tokens = torch.randint(0, 30522, (batch_size, sequence_length_text)).to(device)  # BERT vocab size is 30522 for 'bert-base-uncased' -> each number corresponds to a token
    text_mask = torch.randint(0, 2, (batch_size, sequence_length_text)).float().to(device) # 0 or 1 for size (batch_size, sequence_length_txt) -> determines which tokens are valid
    
    # Image is really going to be I3D video embeddings
    image = torch.randn(batch_size, sequence_length_image, input_size_image).to(device)
    image_mask = torch.randint(0, 2, (batch_size, sequence_length_image)).float().to(device)

    # Audio is going to be VGGish embeddings
    audio = torch.randn(batch_size, sequence_length_audio, input_size_audio).to(device)
    audio_mask = torch.randint(0, 2, (batch_size, sequence_length_audio)).float().to(device) 

    # Forward pass
    base_output = base_model(text_tokens, text_mask, image, image_mask, audio, audio_mask)
    binary_output = unified_model(text_tokens, text_mask, image, image_mask, audio, audio_mask, task="binary")
    multi_output = unified_model(text_tokens, text_mask, image, image_mask, audio, audio_mask, task="multi")

    # Print the output shape
    print("Base Output shape:", base_output.shape) # batch_size by 2304 (768 per modality * 3 modalities)
    print("Binary Output shape:", binary_output.shape) # batch size by 2 (one for each prediction ?, why not by 1)
    print("Multi Output shape:", multi_output.shape) # batch size by 4 by 2 (4 for 4 tasks and 2 for each task)

    print(binary_output[0]) 
    print(multi_output[0])