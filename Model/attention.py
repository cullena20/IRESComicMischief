import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dimension):
        super(Attention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.v = nn.Parameter(torch.rand(dimension), requires_grad=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-10

    def forward(self, h, mask, debug=False):
        # Temporary debug variable
        def dprint(text):
            if debug:
                print(text)

        # temporary helper to debug NaNs
        def dprint_nan_percentage(tensor, tensor_name="tensor"):
            if debug == True:
                num_elements = tensor.numel()
                num_nans = torch.isnan(tensor).sum().item()
                nan_percentage = (num_nans / num_elements) * 100
                print(f"{tensor_name} contains {nan_percentage:.2f}% NaNs")

        # batch_size by max sequence length (500 text) by embedding dim (768)
        # mask size just max sequence length
        dprint(f"Self Attention Input Shape: {h.shape}, Mask Shape {mask.shape}")
        u_it = self.u(h)
        u_it = self.tanh(u_it)

        dprint_nan_percentage(u_it, "Linear + Tanh Input")
        
        alpha = torch.exp(torch.matmul(u_it, self.v))
        dprint_nan_percentage(alpha, "Input Multiplied by Values Exponent")
        dprint(f"Alpha initial shape: {alpha.shape}")

        alpha = mask * alpha + self.epsilon
        dprint_nan_percentage(alpha, "Applying above to mask")
        dprint(f"Alpha on Mask Shape: {alpha.shape}")
        # both alphas will be batch size by max sequence length

        # this is batch_size by 1, and is this source of NaNs -> it quickly explodes
        # even on first batch this is a very large number
        denominator_sum = torch.sum(alpha, dim=-1, keepdim=True)
        dprint(f"Denominator: {denominator_sum}")

        alpha = mask * (alpha / denominator_sum)
        output = h * alpha.unsqueeze(2)
        dprint_nan_percentage(output, "Temp Output")
        dprint(f"Temp Output Shape: {output.shape}")
        # the shape here is batch size by max sequence length by embedding dimension again

        output = torch.sum(output, dim=1)
        dprint_nan_percentage(output, "Final Output")
        dprint(f"Final Output Shape: {output.shape}")
        # the final output is just batch size by embedding dimension
        return output, alpha