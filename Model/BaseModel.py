import sys
import torch
from torch import nn
from .attention import * 
import numpy as np
from transformers import BertModel
import math

debug=False

def dprint(text):
    if debug:
        print(text)

# Cullen: Deal with below
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TEMPORARY
import psutil

# Function to get the memory usage in MB
def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 ** 2  # Convert from bytes to MB


# pp = pprint.PrettyPrinter(indent=4).pprint
# debug = False

bidirectional = True
class BertOutAttention(nn.Module):
    def __init__(self, size, ctx_dim=None):
        super().__init__()
        if size % 12 != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (size, 12))
        self.num_attention_heads = 12
        self.attention_head_size = int(size / 12)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim =size
        self.query = nn.Linear(size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


# CULLEN:
# Currently this is sensitive to input and output size
# We might need to customize this somehow (probably inputs we can adjust somehow)
# Capability 1: Train one modality at a time
# Capability 2: Train on different output tasks
# Capability 3: Why should binary and multi task be separate - modularize better (closely related to 2)
class Bert_Model(nn.Module):

    def __init__(self):
        super(Bert_Model, self).__init__()

        self.rnn_units = 512 # rnn units used to further encode audio and video
        self.embedding_dim = 768 # output embedding for each modality
        
        dropout = 0.2

        # this is the attention module used to embed text, along with dropout layers
        self.att1 = BertOutAttention(self.embedding_dim)
        self.att1_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.att1_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 

        # attention to embed audio, along with dropout layers
        self.att2 = BertOutAttention(self.embedding_dim)
        self.att2_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.att2_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        
        # attention to embed audio, along with dropout layers
        self.att3 = BertOutAttention(self.embedding_dim)
        self.att3_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.att3_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 

        # FC layers to pass RNN embeded audio into (raw audio -> VGGish features elsewhere -> LSTM here -> FC here -> attention)
        self.sequential_audio = nn.Sequential(
            nn.Linear(self.rnn_units*2  , self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),

        )

        # FC layers to pass RNN embedded video into
        self.sequential_image = nn.Sequential(
            nn.Linear(self.rnn_units*2  , self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),

        )

        # CULLEN: added eager thing, meaning some manual implementation is being used to prevent future issues
        # Maybe want to address this later

        # Load BERT model from pretrained to embed sentence tokens
        # Might want to freeze for consistency, or not
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                                output_attentions=True, 
                                                                attn_implementation="eager")

        # LSTM: input_size_audio (128 VGG) dim inputs -> rnn_units (512)
        # Then norm will make it rnn_units * 2 (1024) - not sure why 
        input_size_audio = 128
        self.rnn_audio = nn.LSTM(input_size_audio, self.rnn_units, num_layers=2, bidirectional=True, batch_first = True)
        self.rnn_audio_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 

        # LSTM: input_size_image (1024 I3D) dim inputs -> rnn_units (512)
        # Then norm will make it rnn_units * 2 (1024) - not sure why  
        input_size_image = 1024
        self.rnn_img = nn.LSTM(input_size_image, self.rnn_units, num_layers=2, bidirectional=True, batch_first = True)
        self.rnn_img_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 

        #  Self attention layer for embedded audio
        self.attention_audio = Attention(self.embedding_dim)
        self.attention_audio_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 

        # Self attention layer for embedded video - input has to be self.embedding_dim too: embedding_dim -> embedding_dim
        self.attention_image = Attention(self.embedding_dim)
        self.attention_image_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 

        # Self attention layer for embedded audio
        self.attention = Attention(self.embedding_dim)
        self.attention_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        )
        
        # Got rid of below for easier readibility and just referenced names directly
        # self.features = nn.ModuleList([bert, rnn_img, rnn_img_drop_norm, rnn_audio, rnn_audio_drop_norm, sequential_audio, sequential_image, att1, att1_drop_norm1, att1_drop_norm2, att2, att2_drop_norm1, att2_drop_norm2, att3, att3_drop_norm1, att3_drop_norm2, attention, attention_drop_norm, attention_audio, attention_audio_drop_norm, attention_image, attention_image_drop_norm])

    # NOTE: it is possible that there are size issues below
    # CHECK LOSER
    def forward(self, text, text_mask, image, image_mask, audio, audio_mask, mode='pre_train'):
        """
        Forward pass for the Bert_Model.

        Parameters:
        -----------
        text : torch.Tensor
            Input tensor representing the tokenized text, with shape (batch_size, sequence_length).
        text_mask : torch.Tensor
            Attention mask for the text, with shape (batch_size, sequence_length). This mask differentiates between valid tokens (1) and padding tokens (0).
        image : torch.Tensor
            Input tensor representing the image data, with shape (batch_size, sequence_length, 1024). This data is processed by an LSTM.
        image_mask : torch.Tensor
            Attention mask for the image data, with shape (batch_size, sequence_length). This mask differentiates between valid image segments (1) and padding segments (0).
        audio : torch.Tensor
            Input tensor representing the audio data, with shape (batch_size, sequence_length, 128). This data is processed by an LSTM.
        audio_mask : torch.Tensor
            Attention mask for the audio data, with shape (batch_size, sequence_length). This mask differentiates between valid audio segments (1) and padding segments (0).
        mode : str, optional
            Mode for the model, default is 'pre_train'.

        Returns:
        --------
        torch.Tensor
            The concatenated tensor of processed text, audio, and image representations with shape (batch_size, embedding_dim * 3).
        """

        dprint(f"text shape: {text.shape}") # batch_size by 500
        dprint(f"text_mask shape: {text_mask.shape}") # batch_size by 500
        dprint(f"Image shape: {image.shape}") # batch_size by 36 by 1024 (tokens and embedding dimension)
        dprint(f"Image_mask shape: {image_mask.shape}") # batch_size by 36
        dprint(f"Audio shape: {audio.shape}") # batch_size by 63 by 128 (tokens and embedding dimension)
        dprint(f"Audio_mask shape: {audio_mask.shape}") # batch_size by 63

        dprint("INSIDE TRAINING LOOP")
        dprint(f'CPU Memory Usage: {get_memory_usage()} MB') # started at 232, now is 916
        # encode text tokens using BERT
        hidden, _ = self.bert(text)[-2:] 
        text_encoded = hidden[-1] # MOVED [-1] here because it is the only thing used

        dprint("Text Encoded Is NaN?")
        dprint(torch.isnan(text_encoded.sum()))
        # print(text_encoded)

        dprint(f'CPU Memory Usage After BERT: {get_memory_usage()} MB') # went to 3000
        dprint(f"BERT Embedded text shape {text_encoded.shape}") 
        
        # encode video using LSTM and FC layers (I3D done beforehand)
        rnn_img_encoded, _ = self.rnn_img(image)
        rnn_img_encoded = self.rnn_img_drop_norm(rnn_img_encoded)
        img_encoded = self.sequential_image(rnn_img_encoded)

        dprint(f"Image Embedded shape {img_encoded.shape}")
        # print(img_encoded)

        # encode audio using LSTM and FC layers (VGG done beforehand)
        rnn_audio_encoded, _ = self.rnn_audio(audio)
        rnn_audio_encoded = self.rnn_audio_drop_norm(rnn_audio_encoded)
        audio_encoded = self.sequential_audio(rnn_audio_encoded)

        dprint(f"Audio Embedded shape {audio_encoded.shape}")
        # print(audio_encoded)

        # Every attention mask goes from batch_size by modality_token_size to batch_size by 1 by 1 by modality_token_size
        # Bring 0s to -10000, bring 1s to 0. Huh?

        extended_text_attention_mask = text_mask.float().unsqueeze(1).unsqueeze(2)
        extended_text_attention_mask = extended_text_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_text_attention_mask = (1.0 - extended_text_attention_mask) * -10000.0

        dprint(f"Extended text mask shape {extended_text_attention_mask.shape}")
        
        extended_audio_attention_mask = audio_mask.float().unsqueeze(1).unsqueeze(2)
        extended_audio_attention_mask = extended_audio_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0
      
        dprint(f"Extended audio mask shape {extended_audio_attention_mask.shape}")

        extended_image_attention_mask = image_mask.float().unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = extended_image_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        dprint(f"Extended image mask shape {extended_image_attention_mask.shape}")
        dprint(f"Image Mask One Example {image_mask[0]}")
        dprint(f"Extended Image Mask One Example {extended_image_attention_mask[0]}")
 
        output_text = self.att1(text_encoded, img_encoded, extended_image_attention_mask)
        output_text = self.att1_drop_norm1(output_text)
        output_text = self.att1(output_text, audio_encoded, extended_audio_attention_mask)
        output_text = self.att1_drop_norm2(output_text)
        
        output_text = output_text + text_encoded
        dprint(f"Output Text")
        dprint(f"NaN? : {torch.isnan(output_text.sum())}")

        output_audio = self.att2(audio_encoded, img_encoded, extended_image_attention_mask)
        output_audio = self.att2_drop_norm1(output_audio)
        output_audio = self.att2(output_audio, text_encoded, extended_text_attention_mask)   
        output_audio = self.att2_drop_norm2(output_audio)
        
        output_audio = output_audio + audio_encoded
        dprint(f"Output Audio")
        dprint(f"NaN? : {torch.isnan(output_audio.sum())}")

        output_image = self.att3(img_encoded, text_encoded, extended_text_attention_mask)
        output_image = self.att3_drop_norm1(output_image)
        output_image = self.att3(output_image, audio_encoded, extended_audio_attention_mask)
        output_image = self.att3_drop_norm2(output_image)
        
        output_image = output_image + img_encoded
        dprint(f"Output Image")
        dprint(f"NaN? : {torch.isnan(output_image.sum())}")

        # why are there new masks here?
        mask = torch.tensor(np.array([1]*output_text.size()[1])).to(next(self.parameters()).device) # cuda()
        audio_mask = torch.tensor(np.array([1]*output_audio.size()[1])).to(next(self.parameters()).device) # cuda()
        image_mask = torch.tensor(np.array([1]*output_image.size()[1])).to(next(self.parameters()).device) # cuda()

        #dprint("TEXT BEFORE SELF ATTENTION:", output_text.shape)
        output_text, attention_weights = self.attention(output_text, mask.float())
        output_text = self.attention_drop_norm(output_text)

        dprint(f"Post Attetion Output Text NaN? : {torch.isnan(output_text.sum())}")

        #dprint("TEXT AFTER SELF ATTENTION:", output_text.shape)
        output_audio, attention_weights = self.attention_audio(output_audio, audio_mask.float())
        output_audio = self.attention_audio_drop_norm(output_audio)

        dprint(f"Post Attetion Output Audio NaN? : {torch.isnan(output_audio.sum())}")

        #dprint("IMAGE BEFORE SELF ATTENTION", output_image.shape)
        output_image, attention_weights = self.attention_image(output_image, image_mask.float())
        output_image = self.attention_image_drop_norm(output_image)

        dprint(f"Post Attetion Output Image NaN? : {torch.isnan(output_image.sum())}")
        #dprint("IMAGE AFTER SELF ATTENTION", output_image.shape)

        dprint("Final Concat")

        text_audio_image_cat = torch.cat([output_text, output_audio, output_image], dim=-1)
        dprint(f"Is final cat NaN? {torch.isnan(text_audio_image_cat.sum())}")

        return text_audio_image_cat