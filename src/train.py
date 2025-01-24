import torch
from discriminator_module import AugmentationDiscriminator
from generator_module import AugmentationGenerator

import json
import os

import time


lambda_gp = 10  # Gradient penalty lambda hyperparameter
n_critic = 3  # Number of critic iterations per generator iteration
seq_len = 250  # Sequence length
# d_model = 128  # Transformer model dimension
# nhead = 4  # Number of heads in the multiheadattention models
num_layers = 4  # Number of layers
noise_input_dim = 128  # Noise input dimension
hidden_dim = 64  # Hidden dimension
n_batches = 128  # Number of batches


def parse_data(data):
    parsed_data = []
    for point in data:
        point = point[1]

        LeftFoot = point['LeftFoot']
        LeftHand = point['LeftHand']
        LeftShoulder = point['LeftShoulder']
        LeftUpLeg = point['LeftUpLeg']
        RightFoot = point['RightFoot']
        RightHand = point['RightHand']
        RightShoulder = point['RightShoulder']
        RightUpLeg = point['RightUpLeg']
        base_pos = point['base_pos']
        base_quat = point['base_quat']

        step_data = [
            LeftFoot[0], LeftFoot[1], LeftFoot[2],
            LeftHand[0], LeftHand[1], LeftHand[2],
            LeftShoulder[0], LeftShoulder[1], LeftShoulder[2],
            LeftUpLeg[0], LeftUpLeg[1], LeftUpLeg[2],
            RightFoot[0], RightFoot[1], RightFoot[2],
            RightHand[0], RightHand[1], RightHand[2],
            RightShoulder[0], RightShoulder[1], RightShoulder[2],
            RightUpLeg[0], RightUpLeg[1], RightUpLeg[2],
            base_pos[0], base_pos[1], base_pos[2],
            base_quat[0], base_quat[1], base_quat[2], base_quat[3]
        ]

        parsed_data.append(step_data)

    return torch.Tensor(parsed_data).to(device="cuda")


def load_data():
    real_data = {}
    for i, file in enumerate(os.listdir('traj_data')):
        with open(f'traj_data/{file}', 'r') as f:
            data = json.load(f)

            parsed_data = parse_data(data)
            real_data[i] = parsed_data

    return real_data


def calculate_gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.rand(
        real_data.size(0),
        *([1]*(len(real_data.shape)-1))
    ).to(device="cuda")

    interpolated = alpha * real_data + ((1 - alpha) * fake_data.detach())
    interpolated = interpolated.cuda()
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

    disc_interpolated = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(disc_interpolated.size(), device='cuda'),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


def train(num_epochs):
    generator = AugmentationGenerator(
        noise_input_dim=noise_input_dim,
        output_dim=31,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        # d_model=d_model,
        # nhead=nhead,
        num_layers=num_layers
    )
    discriminator = AugmentationDiscriminator(
        input_dim=31 * seq_len,
        hidden_dim=hidden_dim
    )

    # create log dir with the current date and time
    log_dir = os.path.join("logs", time.strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, exist_ok=True)


    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-5, betas=(0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0, 0.9))

    real_data = load_data()

    for epoch in range(num_epochs):
        timer_start = time.time()
        mean_loss_D = 0
        mean_loss_G = 0
        for real_data_batch in real_data.values():
            # Train Discriminator
            for n in range(n_critic):
                noise = torch.randn(n_batches, noise_input_dim, device='cuda')

                fake_data = generator(noise)
                real_data_expanded = real_data_batch.unsqueeze(0)

                optimizer_D.zero_grad()

                # select a random window of seq_len from the real data
                if real_data_expanded.size(1) < seq_len:
                    real_data_input = torch.zeros(n_batches, seq_len, 31, device='cuda')
                    real_data_input[:, :real_data_expanded.size(1), :] = real_data_expanded
                else:
                    start = torch.randint(
                        0,
                        real_data_expanded.size(1) - seq_len,
                        (n_batches,)
                    )
                    
                    # select a random window of seq_len from the real data
                    real_data_input = torch.zeros(n_batches, seq_len, 31, device='cuda')
                    for i, start in enumerate(start):
                        real_data_input[i] = real_data_expanded[:, start:start + seq_len, :]

                real_output = discriminator(real_data_input)
                fake_output = discriminator(fake_data)

                gradient_penalty = calculate_gradient_penalty(discriminator, real_data_input, fake_data)
                loss_D = -torch.mean(real_output) + torch.mean(fake_output) + gradient_penalty

                loss_D.backward()
                optimizer_D.step()
            
            # Train Generator
            noise = torch.randn(n_batches, noise_input_dim, device='cuda')
            fake_data = generator(noise)
            fake_valid = discriminator(fake_data)
            loss_G = -torch.mean(fake_valid)

            # Update Generator
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            mean_loss_D += loss_D.item()
            mean_loss_G += loss_G.item()
        
        mean_loss_D /= len(real_data)
        mean_loss_G /= len(real_data)
        
        print(f"Epoch {epoch}")
        print(f"Mean Loss D: {mean_loss_D}")
        print(f"Mean Loss G: {mean_loss_G}")
        print(f"Time taken: {time.time() - timer_start}")
        print("==================================================")

        # save model every 1000 epochs to log_dir
        if epoch % 1000 == 0:
            torch.save(generator.state_dict(), os.path.join(log_dir, f"generator_{epoch}.pt"))
            torch.save(discriminator.state_dict(), os.path.join(log_dir, f"discriminator_{epoch}.pt"))
   

if __name__ == "__main__":
    train(10000)
