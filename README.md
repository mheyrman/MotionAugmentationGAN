# Motion Augmentation GAN
## About
This is my first go at developing a GAN.

Developing a GAN to automatically generated formatted trajectory data based on motion capture data. The original purpose of this package is to augment motion capture datasets to enable richer visualization and training data for other purposes.

It is using Wasserstein loss with an LSTM-based generator and a Conv1d based discriminator. It is still heavily in development and needs to be tested, validated, etc...

## Training
All relevant code is in `src`. Training can be done via `train.py` and requires `pytorch`. Motion data should be placed in `traj_data`. The data that I am using cannot be publically pushed. It is processed data from the [MANN dataset](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018) formatted as follows:

- `index`:
    - `0`: Time Step
    - `1`: Position Information
        - `LeftFoot`: [x, y, z]
        - `LeftHand`: [x, y, z]
        - `LeftShoulder`: [x, y, z]
        - `LeftUpLeg`: [x, y, z]
        - `RightFoot`: [x, y, z]
        - `RightHand`: [x, y, z]
        - `RightShoulder`: [x, y, z]
        - `RightUpLeg`: [x, y, z]
        - `base_pos`: [x, y, z]
        - `base_quat`: [w, x, y, z]

