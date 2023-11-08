
import torch

def print_torch_bin(file_path):
    torch_bin = torch.load(file_path)
    # print(torch_bin.keys())
    for key in torch_bin.keys():
        print(key, torch_bin[key].shape, torch_bin[key].dtype)
    # print(torch_bin['model'].keys())
    # print(torch_bin['model']['transformer'].keys())
    # print(torch_bin['model']['transformer']['encoder'].keys())
    # print(torch_bin['model']['transformer']['encoder']['layer'].keys())
    # print(torch_bin['model']['transformer']['encoder']['layer'][0].keys())
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn'].keys())
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['in_proj_weight'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['in_proj_bias'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['out_proj'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['attn_mask'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['bias_k'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['bias_v'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['q_proj_weight'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['k_proj_weight'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['v_proj_weight'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['in_proj_weight'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['in_proj_bias'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['out_proj'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['attn_mask'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['bias_k'].shape)
    # print(torch_bin['model']['transformer']['encoder']['layer'][0]['self_attn']['bias_v'].shape)

if __name__ == "__main__":
    
    # print_torch_bin("/home/xiachunwei/Dataset/HW-cost-model-dataset/cbench_sft_output/sft_lora_model/adapter_model.bin")
    print_torch_bin("/home/xiachunwei/Dataset/cbench_sft_output/perf_model_state_dict.pt")
    # print("*"*100)
    # print_torch_bin("/home/xiachunwei/Dataset/cbench_sft_output/codellama_old_state_dict.pt")

