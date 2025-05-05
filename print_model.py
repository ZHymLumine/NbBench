import torch
from safetensors.torch import load_file
# from transformers import RobertaForMaskedLM, AutoModel, AutoConfig

# model_path = "/home/yzhang/research/nanobody_benchmark/checkpoint/opensource/vhhbert/model.safetensors"
# model = load_file(model_path)
# for key in model.keys():
#     print(key)

# # model = RobertaForMaskedLM.from_pretrained('multimolecule/rnabert')
# # print(model)

# print('='*100)
# # model_path_bin = "/home/yzhang/research/nanobody/checkpoint/opensource/nanobert/pytorch_model.bin"
# # model_bin = torch.load(model_path_bin, map_location=torch.device('cpu'))
# # for key in model_bin.keys():
# #     print(key)

from transformers import AutoModel

model_name = "Exscientia/IgBert"
model_path = "/home/yzhang/research/nanobody_benchmark/checkpoint/opensource/vhhbert/model.safetensors"
# 加载safetensors格式的模型
model_safetensors = load_file(model_path)
total_params_safetensors = sum(p.numel() for p in model_safetensors.values())
print(f"safetensors格式模型的总参数量: {total_params_safetensors/1e6:.1f} M")

# 加载pytorch格式的模型
model_path_bin = "/home/yzhang/research/nanobody_benchmark/checkpoint/opensource/nanobert/pytorch_model.bin"
model_bin = torch.load(model_path_bin, map_location=torch.device('cpu'))
total_params_bin = sum(p.numel() for p in model_bin.values())
print(f"pytorch格式模型的总参数量: {total_params_bin/1e6:.1f} M")


# 加载模型（首次运行会联网下载并缓存到 ~/.cache/huggingface）
# model = AutoModel.from_pretrained(model_name)


# total_params = sum(p.numel() for p in model.parameters())

# print(f"Total parameters: {total_params/1e6:.1f} M")
