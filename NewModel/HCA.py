import torch
from torch import nn
import math
from transformers import BertTokenizer, BertModel


class Attention(nn.Module):
    def __init__(self, dimension):
        super(Attention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.v = nn.Parameter(torch.rand(dimension), requires_grad=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-10

    def forward(self, h, mask, audio_mask, image_mask):
        u_it = self.u(h)
        u_it = self.tanh(u_it)
        
        alpha = torch.exp(torch.matmul(u_it, self.v))
        alpha = mask * alpha + self.epsilon
        denominator_sum = torch.sum(alpha, dim=-1, keepdim=True)
        alpha = mask * (alpha / denominator_sum)
        output = h * alpha.unsqueeze(2)
        output = torch.sum(output, dim=1)

        extended_attention_mask = mask.float().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        self.extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_audio_attention_mask = audio_mask.float().unsqueeze(1).unsqueeze(2)
        extended_audio_attention_mask = extended_audio_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        self.extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0

        extended_image_attention_mask = image_mask.float().unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = extended_image_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        self.extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        return output, alpha

class BertOutAttention(nn.Module):
    def __init__(self, size, ctx_dim=None):
        super().__init__()
        if size % 12 != 0:
            raise ValueError(
                "The encoded_text size (%d) is not a multiple of the number of attention "
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

    def forward(self, encoded_text_states, context, attention_mask=None):
        mixed_query_layer = self.query(encoded_text_states)
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

class HCA(nn.Module):
    def __init__(self):
        super(HCA, self).__init__()
        self.rnn_units = 512
        self.embedding_dim = 768
        dropout = 0.2

        self.att1 = BertOutAttention(self.embedding_dim)
        self.att1_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.att1_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.att2 = BertOutAttention(self.embedding_dim)
        self.att2_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.att2_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        
        self.att3 = BertOutAttention(self.embedding_dim)
        self.att3_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.att3_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 

        self.bert = BertModel.from_pretrained('bert-base-uncased', output_encoded_text_states=True,
                                                                output_attentions=True)

        self.rnn_audio = nn.LSTM(128, self.rnn_units, num_layers=2, bidirectional=True, batch_first = True)
        self.rnn_audio_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 
        self.rnn_img = nn.LSTM(1024, self.rnn_units, num_layers=2, bidirectional=True, batch_first = True)
        self.rnn_img_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 

        self.attention_audio = Attention(768)
        self.attention_audio_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.attention_image = Attention(768)
        self.attention_image_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.attention = Attention(768)
        self.attention_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        )
        
    def forward(self, encoded_text, encoded_img, extended_image_attention_mask, 
                encoded_audio, extended_audio_attention_mask, extended_attention_mask):
    
        output_text = self.att1(encoded_text[-1], encoded_img, extended_image_attention_mask)
        output_text = self.att1_drop_norm1(output_text)
        output_text = self.att1(output_text, encoded_audio, extended_audio_attention_mask)
        output_text = self.att1_drop_norm2(output_text)
        
        output_text = output_text + encoded_text[-1]

        output_audio = self.att2(encoded_img, encoded_img ,extended_image_attention_mask)
        output_audio = self.att2_drop_norm1(output_audio)
        output_audio = self.att2(output_audio, encoded_text[-1], extended_attention_mask)   
        output_audio = self.att2_drop_norm2(output_audio)
        
        output_audio = output_audio + encoded_audio

        output_image = self.att3(encoded_img, encoded_text[-1], extended_attention_mask)
        output_image = self.att3_drop_norm1(output_image)
        output_image = self.att3(output_image, encoded_audio ,extended_audio_attention_mask)
        output_image = self.att3_drop_norm2(output_image)
        
        output_image = output_image + encoded_img

        return output_text, output_audio, output_image