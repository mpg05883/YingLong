import torch
from transformers import AutoModelForCausalLM

# load pretrain model
model = AutoModelForCausalLM.from_pretrained(
    'qcw2333/YingLong_6m',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()

# prepare input
batch_size, lookback_length = 1, 2880
seqs = torch.randn(batch_size, lookback_length).bfloat16().cuda()

# generate forecast
prediction_length = 96
output = model.generate(seqs, future_token=prediction_length)

print(output.shape)