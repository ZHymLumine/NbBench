import torch
from safetensors.torch import load_file
from transformers import RobertaForMaskedLM, AutoModel, AutoConfig

model_path = "/home/yzhang/research/nanobody/outputs/ft/sdabtype/opensource/nanobert/checkpoint-200/model.safetensors"
model = load_file(model_path)
for key in model.keys():
    print(key)

# model = RobertaForMaskedLM.from_pretrained('multimolecule/rnabert')
# print(model)

print('='*100)
model_path_bin = "/home/yzhang/research/nanobody/checkpoint/opensource/nanobert/pytorch_model.bin"
model_bin = torch.load(model_path_bin, map_location=torch.device('cpu'))
for key in model_bin.keys():
    print(key)