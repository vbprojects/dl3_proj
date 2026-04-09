from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class MonteCarloDropoutHead(nn.Module):
    """
    Monte Carlo Dropout uses a dropout layer and a linear layer at the end of an embedding method. 
    While they can be used as a powerful form of regularization, the intent is to use dropout 
    in inference and sample multiple outputs to use as multiple draws from a single language model.
    """
    def __init__(self, input_dim: int, output_dim: int, num_categories :int, dropout_prob: float = 0.1):
        """
        Initializes the Monte Carlo Dropout Head.
        
        Args:
            input_dim (int): The dimensionality of the input embeddings.
            output_dim (int): The dimensionality of the output projections.
            dropout_prob (float): The probability of an element to be zeroed. Default: 0.1.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        
        # Standard dropout and linear projection for the standard forward pass
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(input_dim, output_dim)
        self.final_layer = nn.Linear(output_dim, num_categories)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass. 
        Note: This obeys the module's current mode (train() vs eval()). 
        If the model is in eval() mode, dropout will be bypassed.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (..., output_dim)
        """
        x = F.selu(x)
        x = self.dropout(x)
        x = self.linear(x)
        embeddings = F.selu(x)
        x = self.final_layer(embeddings)
        # x = F.softmax(x, dim=-1)
        return x, embeddings