import torch

import json
import os
import argparse


from generator_module import AugmentationGenerator

lambda_gp = 10  # Gradient penalty lambda hyperparameter
n_critic = 3  # Number of critic iterations per generator iteration
seq_len = 250  # Sequence length
# d_model = 128  # Transformer model dimension
# nhead = 4  # Number of heads in the multiheadattention models
num_layers = 4  # Number of layers
noise_input_dim = 128  # Noise input dimension
hidden_dim = 64  # Hidden dimension
n_batches = 128  # Number of batches

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="generator_6000.pt")
parser.add_argument("-s", "--seed", type=int, default=17)
parser.add_argument("-n", "--num_generated_samples", type=int, default=4)

args = parser.parse_args()

model_name = args.model
seed = args.seed
num_generated_samples = args.num_generated_samples


def parse_generated_data(data):
    parsed_data = []
    contacts_dict = {}
    contacts_data = []
    timestep = 1 / 60
    t = 0
    for _, point in enumerate(data):
        step_data = []
        step_data.append(t)
        c_fdict = {}

        # if foot z < 0, set foot z = 0
        point_copy = point.clone()
        for j in [2, 5, 14, 17]:
            if point_copy[j] < 0:
                point_copy[j] = 0
        point = point_copy

        c_fdict['LeftFoot'] = True if point[2] < 0.01 else False
        c_fdict['LeftHand'] = True if point[5] < 0.01 else False
        c_fdict['RightFoot'] = True if point[14] < 0.01 else False
        c_fdict['RightHand'] = True if point[17] < 0.01 else False
        c_info = [t, c_fdict]
        contacts_data.append(c_info)

        step_dict = {
            'LeftFoot': [point[0].item(), point[1].item(), point[2].item()],
            'LeftHand': [point[3].item(), point[4].item(), point[5].item()],
            'LeftShoulder': [point[6].item(), point[7].item(), point[8].item()],
            'LeftUpLeg': [point[9].item(), point[10].item(), point[11].item()],
            'RightFoot': [point[12].item(), point[13].item(), point[14].item()],
            'RightHand': [point[15].item(), point[16].item(), point[17].item()],
            'RightShoulder': [point[18].item(), point[19].item(), point[20].item()],
            'RightUpLeg': [point[21].item(), point[22].item(), point[23].item()],
            'base_pos': [point[24].item(), point[25].item(), point[26].item()],
            'base_quat': [point[27].item(), point[28].item(), point[29].item(), point[30].item()]
        }
        step_data.append(step_dict)
        parsed_data.append(step_data)

        t += timestep
        
    contacts_dict['data'] = (contacts_data)
    
    return parsed_data, contacts_dict

def play():
    # log_dir is newest dir in logs
    log_dir = "logs"
    log_dir = os.path.join(log_dir, sorted(os.listdir(log_dir))[-1])
    model = AugmentationGenerator(
        noise_input_dim=noise_input_dim,
        output_dim=31,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device='cuda'
    )
    state_dict = torch.load(os.path.join(log_dir, model_name), weights_only=True)
    model.load_state_dict(state_dict=state_dict)

    torch.manual_seed(seed)

    noise_inputs = torch.randn(num_generated_samples, noise_input_dim).cuda()

    generated_data = model(noise_inputs)

    for i, data in enumerate(generated_data):
        parsed_data, contacts_data = parse_generated_data(data)
        
        os.makedirs("data_gen", exist_ok=True)
        
        traj_file_name = f"data_gen/G_{seed}_{i}_traj_processed.json"
        contact_file_name = f"data_gen/G_{seed}_{i}_contact_cropped.json"

        with open(traj_file_name, 'w') as f:
            json.dump(parsed_data, f)
        with open(contact_file_name, 'w') as f:
            json.dump(contacts_data, f)
        print(f"Generated data saved to {traj_file_name} and {contact_file_name}")
    

if __name__ == "__main__":
    play()