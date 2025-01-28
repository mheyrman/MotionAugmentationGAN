import torch
import torch.nn as nn

class LSTMAugmentationGenerator(nn.Module):
    def __init__(
            self,
            noise_input_dim,
            output_dim,
            hidden_dim,
            history_len,
            # d_model,
            # nhead,
            num_layers,
            device='cuda'
        ):
        super(AugmentationGenerator, self).__init__()
        self.noise_input_dim = noise_input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.history_len = history_len
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
        # x: [batch_size, history_len, noise_input_dim]
        batch_size = x.size(0)
        x = self.norm(x)
        hc_init = self.fc_init(x)
        h_init, c_init = torch.split(hc_init, self.hidden_dim * self.num_layers, dim=1)
        h_init = h_init.reshape(self.num_layers, batch_size, self.hidden_dim)
        c_init = c_init.reshape(self.num_layers, batch_size, self.hidden_dim)

        noise_seq = x.unsqueeze(1).repeat(1, self.history_len, 1)

        lstm_out, _ = self.lstm_generator(noise_seq, (h_init.contiguous(), c_init.contiguous()))

        y = self.fc_out(lstm_out)

        return y

# class AugmentationGenerator(nn.Module):
#     """
#     Test a DAGAN:
#     - input: real information, noise
#     - encode input data, add noise, decode to generate fake data
#     """
#     def  __init__(
#             self,
#             input_dim,
#             hidden_noise_dim,
#             history_len,
#             encoder_dim,
#             decoder_dim,
#             device='cuda'
#         ):
#         super(AugmentationGenerator, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_noise_dim = hidden_noise_dim
#         self.history_len = history_len
#         self.encoder_dim = encoder_dim
#         self.decoder_dim = decoder_dim

#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, encoder_dim, device=device),
#             nn.LeakyReLU(),
#             nn.Linear(encoder_dim, encoder_dim, device=device),
#             nn.LeakyReLU(),
#             nn.Linear(encoder_dim, hidden_noise_dim, device=device),
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_noise_dim, decoder_dim, device=device),
#             nn.LeakyReLU(),
#             nn.Linear(decoder_dim, decoder_dim, device=device),
#             nn.LeakyReLU(),
#             nn.Linear(decoder_dim, input_dim, device=device),
#         )

#     def forward(self, x):
#         # x: [batch_size, history_len, input_dim]
#         batch_size = x.size(0)
#         noise = torch.normal(
#             mean=0.0,
#             std=1.0,
#             size=(batch_size, self.history_len, self.hidden_noise_dim)
#         ).to(device=x.device)
#         x = self.encoder(x)
#         x = x + noise
#         x = self.decoder(x)
#         x = x.reshape(batch_size, self.history_len, self.input_dim)

#         return x

class AugmentationGenerator(nn.Module):
    """
    Test a DAGAN:
    - input: real information, noise
    - encode input data, add noise, decode to generate fake data
    """
    def  __init__(
            self,
            input_dim,
            hidden_noise_dim,
            history_len,
            encoder_dim,
            decoder_dim,
            device='cuda'
        ):
        super(AugmentationGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_noise_dim = hidden_noise_dim
        self.history_len = history_len
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim * history_len, encoder_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, encoder_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, hidden_noise_dim, device=device),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_noise_dim, decoder_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(decoder_dim, decoder_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(decoder_dim, input_dim, device=device),
        )

    def forward(self, x):
        # x: [batch_size, history_len, input_dim]
        batch_size = x.size(0)
        # flatten the input tensor
        x = x.reshape(batch_size, self.input_dim * self.history_len)
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, self.hidden_noise_dim)
        ).to(device=x.device)
        x = self.encoder(x)
        x = x + noise
        x = self.decoder(x)
        x = x.reshape(batch_size, self.input_dim)

        return x.unsqueeze(1)