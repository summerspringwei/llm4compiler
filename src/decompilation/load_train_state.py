import torch

state_path = ""
state_dict = torch.load(state_path)
for k, v in state_dict.items():
    print(k, v)
