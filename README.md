# YingLong# YingLong

YingLong model is introduced in this [paper](xxxxxxxx) (coming soon). This version is pre-trained on 78B time points. We provide 4 differnet sizes ([6m](https://huggingface.co/qcw2333/YingLong_6m), [50m](https://huggingface.co/qcw2333/YingLong_50m), [110m](https://huggingface.co/qcw2333/YingLong_110m) and [300m](https://huggingface.co/qcw2333/YingLong_300m)).

 
## Quickstart

```
pip install xformers transformers
pip install flash-attn --no-build-isolation
git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
```
The flash attention is not required. If you use V100 or other GPU doesn't support flash attention, just change the FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1") to
FlashAttention2Available = False in the model.py file. It should be able to run. 

```
import torch
from transformers import AutoModelForCausalLM

# load pretrain model
model = AutoModelForCausalLM.from_pretrained('qcw2333/YingLong_6m', trust_remote_code=True,torch_dtype=torch.bfloat16).cuda()

# prepare input
batch_size, lookback_length = 1, 2880
seqs = torch.randn(batch_size, lookback_length).bfloat16().cuda()

# generate forecast
prediction_length = 96
output = model.generate(seqs, future_token=prediction_length)

print(output.shape)
```

A notebook example is also provided [here](xxxxxxx). Try it out!


In order to replicate the long-term forecasting results in the paper, please run

```
pip install -r Long_Term_Forecasting/requirement.txt

fabric run \
--accelerator=cuda \
--devices=1 \
--num-nodes=1 \
--main-port=1145 \
Long_Term_Forecasting/main.py \
--batch_size 32 \
--seq_len 4096 \
--future_token 4096 \
--model_name qcw1314/YingLong_300m \
--num_gpus 1 \
-t ETTh1 \
-t ETTh2 \
-t ETTm1 \
-t ETTm2 \
-t Weather

```

<!-- ## Specification -->


<!-- ## Acknowledgments -->

## Citation

coming soon...

## Contact

If you have any questions or want to use the code, feel free to contact:

Xue Wang (xue.w@alibaba-inc.com)

Tian Zhou (an.zt@alibaba-inc.com)

 
## License

This model is licensed under the cc-by-4.0 License.