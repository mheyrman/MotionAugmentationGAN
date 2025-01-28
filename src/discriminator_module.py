import torch
import torch.nn as nn

class AugmentationDiscriminator(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            device='cuda'
        ):
        super(AugmentationDiscriminator, self).__init__()
        self.seq_len = input_dim // 31
        self.input_dim = 31
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm([self.seq_len, self.input_dim], device=device)

        self.conv = nn.Sequential(
            nn.Conv1d(
                self.input_dim,
                hidden_dim,
                5,
                stride=1,
                padding=2,
                device=device
            ),
            nn.LayerNorm([hidden_dim, self.seq_len], device=device),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(
                self.hidden_dim,
                hidden_dim,
                5,
                stride=1,
                padding=2,
                device=device
            ),
            nn.LayerNorm([hidden_dim, self.seq_len], device=device),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(hidden_dim, 1, device=device)

    def forward(self, x):
        # normalize the noise input for each batch
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # apply conv1d to the sequence dimension
        x = self.conv(x)
        x = x.squeeze(-1)
        x = self.fc(x)

        return x