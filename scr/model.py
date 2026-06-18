import torch
import torch.nn as nn
import torch.nn.init as init

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        init.zeros_(self.linear.bias)
        init.normal_(self.linear.weight, 0, 6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm.forward(x)
        # hn = [D,N,H_out] D = (2 if bidirectional else 1) * num_layers
        h = hn.squeeze(0) # [N,H_out]
        return torch.sin(self.linear.forward(h))