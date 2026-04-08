import torch
import torch.nn as nn
import torch.nn.functional as F

class MonteCarloDropoutHead(nn.Module):
    """
    Monte Carlo Dropout uses a dropout layer and a linear layer at the end of an embedding method. 
    While they can be used as a powerful form of regularization, the intent is to use dropout 
    in inference and sample multiple outputs to use as multiple draws from a single language model.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout_prob: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass. 
        Note: This obeys the module's current mode (train() vs eval()). 
        If the model is in eval() mode, dropout will be bypassed.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (..., output_dim)
        """
        x = self.dropout(x)
        return self.linear(x)

    def mc_forward(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Executes Monte Carlo Dropout inference by forcing dropout activation, 
        yielding multiple stochastic draws for the same input.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, input_dim) 
                              or (..., input_dim).
            num_samples (int): Number of stochastic forward passes to execute.
            
        Returns:
            torch.Tensor: Stacked outputs of shape (..., num_samples, output_dim).
        """
        mc_outputs = []
        
        for _ in range(num_samples):
            # F.dropout with training=True forces the stochastic mask generation 
            # and weight scaling, completely ignoring if the model is in .eval() mode.
            dropped_x = F.dropout(x, p=self.dropout_prob, training=True)
            
            # Project the stochastically masked embedding
            out = self.linear(dropped_x)
            mc_outputs.append(out)
            
        # Stack the outputs along the penultimate dimension.
        # If input x is (B, D), output is (B, N, O) where N=num_samples.
        return torch.stack(mc_outputs, dim=-2)

    def get_mc_moments(self, x: torch.Tensor, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Utility method to calculate the predictive mean and predictive variance 
        (epistemic uncertainty) from the MC draws.
        
        Args:
            x (torch.Tensor): Input embeddings.
            num_samples (int): Number of stochastic draws.
            
        Returns:
            tuple:
                - Predictive mean of shape (..., output_dim)
                - Predictive variance of shape (..., output_dim)
        """
        # Shape: (..., num_samples, output_dim)
        mc_samples = self.mc_forward(x, num_samples)
        
        # Calculate moments over the num_samples dimension (dim=-2)
        predictive_mean = mc_samples.mean(dim=-2)
        predictive_variance = mc_samples.var(dim=-2, unbiased=True)
        
        return predictive_mean, predictive_variance