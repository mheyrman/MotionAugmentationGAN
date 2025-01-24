import torch
import torch.nn as nn

class AugmentationGenerator(nn.Module):
    def __init__(
            self,
            noise_input_dim,
            output_dim,
            hidden_dim,
            seq_len,
            # d_model,
            # nhead,
            num_layers,
            device='cuda'
        ):
        super(AugmentationGenerator, self).__init__()
        self.noise_input_dim = noise_input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.norm = nn.LayerNorm(noise_input_dim, device=device)

        self.fc_init = nn.Linear(noise_input_dim, hidden_dim * num_layers * 2, device=device)
        self.lstm_generator = nn.LSTM(
            input_size=noise_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            device=device
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim, device=device)
        # self.tanh = nn.Tanh()

    def forward(self, x):
        # x: [batch_size, seq_len, noise_input_dim]
        batch_size = x.size(0)
        x = self.norm(x)
        hc_init = self.fc_init(x)
        h_init, c_init = torch.split(hc_init, self.hidden_dim * self.num_layers, dim=1)
        h_init = h_init.reshape(self.num_layers, batch_size, self.hidden_dim)
        c_init = c_init.reshape(self.num_layers, batch_size, self.hidden_dim)

        noise_seq = x.unsqueeze(1).repeat(1, self.seq_len, 1)

        lstm_out, _ = self.lstm_generator(noise_seq, (h_init.contiguous(), c_init.contiguous()))

        y = self.fc_out(lstm_out)

        return y
