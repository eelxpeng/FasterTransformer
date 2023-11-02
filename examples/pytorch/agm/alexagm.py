import os
import sys
ROOT_DIR = os.path.abspath("../../../")
sys.path.append(ROOT_DIR)
lib_path = os.path.join(ROOT_DIR, './build/lib/libth_transformer.so')

import configparser
import numpy as np
import torch
import os
import numpy as np
import time
import math

from examples.pytorch.agm.utils.ft_decoding import FTAgmDecodingWeight, FTAgmDecoding, FTAgm

from examples.pytorch.agm.atm_decoder.modeling_atm_decoder import ATMT5CE4ForConditionalGeneration, ATMT5Config
from examples.pytorch.agm.atm_decoder.test import get_model_config

# specify model name or checkpoint path
model_name = 'test'
config = ATMT5Config(**get_model_config(model_name))
model = ATMT5CE4ForConditionalGeneration(config)

# perturb random weight
for p in model.parameters():
    if len(p.shape) == 1:
        torch.nn.init.normal_(p, mean=1.5, std=2)
    elif len(p.shape) == 2:
        torch.nn.init.xavier_normal_(p, gain=2)
    # torch.nn.init.normal_(p, mean=0.5, std=2)
    # print(p)

## fixed random weight model for debugging
# model.save_pretrained('/tmp/alexa')
# model = ATMT5CE4ForConditionalGeneration.from_pretrained('/tmp/alexa')

model = model.eval().to('cuda')

# print all weights
# for name, para in model.named_parameters():
#     print(f'{name}: {para.shape}') 
#     if "relative_attention_bias" in name:
#         print(para)

activation_type = config.activation_function
config.decoder_start_token_id = 2 # doesn't matter since decoding start after context phase

# note: alexatm LN eps=1e-12
# single-gpu so set TP=1, PP=1
tensor_para_size = 1
pipeline_para_size = 1
agm_with_bias = True
use_gated_activation = False
position_embedding_type = 0 # relative positional embedding
encoder_head_size = config.d_model // config.encoder_attention_heads
decoder_head_size = config.d_model // config.decoder_attention_heads
remove_padding = False
use_bf16 = True
weight_data_type = "bf16" if use_bf16 else "fp32"
q_scaling = None
if config.scale_attention:
    if config.mup_scale:
        q_scaling = 1.0 / np.sqrt(decoder_head_size) * decoder_head_size / 8  # --> HF q scaling 8 / head_size
    else:
        q_scaling =  1.0 # --> HF q scaling 1/sqrt(head_size)
else:
    q_scaling = 1.0 / np.sqrt(decoder_head_size) # --> HF q scaling 1.0
use_cache = True
pad_embedding = False

vocab_size = config.vocab_size
if pad_embedding:
    # same logic in AgmDecoding.cc for computing vocab_size_padded_
    raw_vocab_size = config.vocab_size
    local_vocab_size = np.ceil(raw_vocab_size / tensor_para_size)
    if weight_data_type == "fp16" or weight_data_type == "bf16":
        local_vocab_size = np.ceil(local_vocab_size / 8) * 8
    new_vocab_size = int(local_vocab_size * tensor_para_size)
    vocab_size = new_vocab_size

ft_decoding_weight = FTAgmDecodingWeight(
    config,
    tensor_para_size,
    pipeline_para_size,
    agm_with_bias=agm_with_bias,
    use_gated_activation=use_gated_activation,
    position_embedding_type=position_embedding_type,
    weight_data_type=weight_data_type,
    pad_embedding=pad_embedding
)
ft_decoding_weight.load_from_model(model.bfloat16() if use_bf16 else model.float())

if use_bf16:
    ft_decoding_weight.to_bfloat16()#to_half()

ft_decoding = FTAgmDecoding(ft_decoding_weight.w, lib_path,
                        config.decoder_attention_heads, decoder_head_size,
                        config.decoder_ffn_dim, config.d_model,
                        config.d_model, config.decoder_layers,
                        config.decoder_start_token_id, config.eos_token_id, vocab_size=vocab_size,
                        q_scaling=q_scaling, num_bucket=config.relative_attention_num_buckets, max_distance=config.relative_attention_max_distance,
                        tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size, 
                        agm_with_bias=agm_with_bias,
                        position_embedding_type=position_embedding_type, 
                        activation_type=activation_type, use_bf16=use_bf16)

ft_agm = FTAgm(ft_decoding)

## Test
batch_size = 1
input_len = 12
inputs = {
    'input_ids': torch.randint(0, config.vocab_size, size=(batch_size, input_len)).to("cuda"),
    'attention_mask': torch.ones(size=(batch_size, input_len)).to("cuda")    
}

#triton check
batch_size = 1
input_len = 12
inputs = {
    'input_ids': torch.tensor([[176832,  15500, 117023, 113787,  79942,  88278, 224053,  81494,  16261,
            145410,  17548,  14670]]).to("cuda"),
    'attention_mask': torch.ones(size=(batch_size, input_len)).to("cuda")    
}

print("input ids", inputs['input_ids'].tolist())

# forced_decoder_ids = None
forced_decoder_ids = inputs['input_ids'].type(torch.int32).to("cuda").contiguous()
max_output_len = 32
ft_max_output_len = max_output_len # to achieve identical results w/ HF, exclude start & end tokens
num_beams = 1
beam_search_diversity_rate = 0.0
topk = None
topp = None
measurement_iters = 1

# PyT test
if use_bf16:
    model.bfloat16() #half()
else:
    model.float()
hf_outputs = model.generate(inputs['input_ids'], max_length=max_output_len, num_beams=num_beams, use_cache=use_cache)
print("HF output ids",hf_outputs.tolist())

hf_latencies = []
for _ in range(measurement_iters):
    start_time = time.time()
    model.generate(inputs['input_ids'], max_length=max_output_len, num_beams=num_beams, use_cache=use_cache)
    end_time = time.time()
    hf_latencies.append(end_time - start_time)
hf_p50 = np.percentile(hf_latencies, 50)
hf_p99 = np.percentile(hf_latencies, 99)
print(f"HF p50: {hf_p50*1000:.2f} ms, p99: {hf_p99*1000:.2f} ms ")

# FT test
return_dict = ft_agm(inputs['input_ids'],
                      inputs['attention_mask'],
                      inputs_embeds=None,
                      beam_size=num_beams,
                      max_seq_len=ft_max_output_len,
                      top_k=topk,
                      top_p=topp,
                      beam_search_diversity_rate=beam_search_diversity_rate,
                      is_return_output_log_probs=False,
                      is_return_cum_log_probs=False,
                      forced_decoder_ids=forced_decoder_ids)

# ft_agm returns output_ids of shape [batch_size, beam_width, max_output_seq_len]
# ft_agm returns sequence_length of shape [batch_size, beam_width]
ft_output_ids = return_dict['output_ids']
ft_sequence_length = return_dict['sequence_lengths']

ft_outputs = []
for i in range(batch_size):
    # selecting the top sequence from beam width number of sequences
    ft_outputs.append([inputs['input_ids'][i,0].item()] + list(ft_output_ids[i, 0, :][:ft_sequence_length[i , 0]])) # 1st token is not reported by FT, add it back
print("FT output ids", ft_outputs)

ft_latencies = []
for _ in range(measurement_iters):
    start_time = time.time()
    return_dict = ft_agm(inputs['input_ids'],
                          inputs['attention_mask'],
                          inputs_embeds=None,
                          beam_size=num_beams,
                          max_seq_len=ft_max_output_len,
                          top_k=topk,
                          top_p=topp,
                          beam_search_diversity_rate=beam_search_diversity_rate,
                          is_return_output_log_probs=False,
                          is_return_cum_log_probs=False,
                          forced_decoder_ids=forced_decoder_ids)
    end_time = time.time()
    ft_latencies.append(end_time - start_time)
ft_p50 = np.percentile(ft_latencies, 50)
ft_p99 = np.percentile(ft_latencies, 99)
print(f"FT p50: {ft_p50*1000:.2f} ms, p99: {ft_p99*1000:.2f} ms ")

print(f"Precision: {'BF16' if use_bf16 else 'FP32'}")
print(f"Input length: {input_len}, Output length: {max_output_len}")
print(f"HF p50: {hf_p50*1000:.2f} ms, p99: {hf_p99*1000:.2f} ms ")
print(f"FT p50: {ft_p50*1000:.2f} ms, p99: {ft_p99*1000:.2f} ms ")