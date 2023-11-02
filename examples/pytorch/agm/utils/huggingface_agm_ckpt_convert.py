# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python utils/huggingface_agm_ckpt_convert.py -i test -o agm -i_g 1 -w bf16 --pad_embedding

import argparse
import configparser
import multiprocessing
from datetime import datetime
import logging
from pathlib import Path

import sys
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../../../../"
sys.path.append(ROOT_DIR)

from examples.pytorch.agm.atm_decoder.modeling_atm_decoder import ATMT5CE4ForConditionalGeneration, ATMT5Config
from examples.pytorch.agm.atm_decoder.test import get_model_config

import numpy as np
import torch  # pytype: disable=import-error

LOGGER = logging.getLogger(__name__)

def np_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    elif data_type == "bf16":
        return np.float16
        # numpy has no bf16 type, need to convert from torch.bfloat16 via torch.int16
    else:
        assert False, f"Invalid weight data type {data_type}"

def pyt_weight_data_type(data_type):
    if data_type == "fp32":
        return torch.float32
    elif data_type == "fp16":
        return torch.float16
    elif data_type == "bf16":
        return torch.float16 # still use fp16
    else:
        assert False, f"Invalid weight data type {data_type}"

def fuse_decoder_qkv(model, factor, saved_dir, weight_data_type):
    model_dict = {}
    for name, param in model.named_parameters():
        if name.find("decoder") != -1 and name.find("self_attn") != -1:
            model_dict[name.replace("model.", "")] = param

    for i in range(model.config.decoder_layers):
        shape = model_dict[f"decoder.layers.{i}.self_attn.q.weight"].T.shape
        qkv = torch.cat([model_dict[f"decoder.layers.{i}.self_attn.q.weight"].T,
                         model_dict[f"decoder.layers.{i}.self_attn.k.weight"].T,
                         model_dict[f"decoder.layers.{i}.self_attn.v.weight"].T], dim=-1)

        qkv = qkv.reshape([shape[0], 3, shape[1]])
        # if weight_data_type == "bf16":
        #     qkv = qkv.view(torch.int16)
        #     qkv = qkv.detach().cpu().numpy()
        # else: 
            # qkv = qkv.to(pyt_weight_data_type(weight_data_type))
            # qkv = qkv.detach().cpu().numpy().astype(np_weight_data_type(weight_data_type))
        qkv = qkv.to(pyt_weight_data_type(weight_data_type))
        qkv = qkv.detach().cpu().numpy().astype(np_weight_data_type(weight_data_type))

        split_vals = np.split(qkv, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"decoder.layers.{i}.self_attn.qkv.weight.{j}.bin"
            split_vals[j].tofile(saved_path.as_posix())

def split_and_convert_process(model, factor, saved_dir, weight_data_type, pad_embedding, pad_length):
    for key, val in model.state_dict().items():

        saved_key = key.replace("model.", "")

        if val.dim() == 2:
            val = val.transpose(1, 0)
        
        # embedding tables (FT use TN gemm to compute, no transpose) and bias
        # do padding accordingly
        if key.find("shared.weight") != -1 or key.find("lm_head.decoder.weight") != -1:
            val = val.transpose(1, 0)
            if pad_embedding and pad_length > 0:
                val = torch.cat((val, torch.zeros(pad_length, val.shape[1]).to(val.device)))
        if key.find("final_logits_bias") != -1:
            if pad_embedding and pad_length > 0:
                val = torch.cat((val, torch.zeros(pad_length).to(val.device)))
        
        LOGGER.debug(f"key: {key}, val.shape: {val.shape}")

        # special handling for bf16 data type (actually, NO, buggy, please use fp32/fp16 and specify bf16 in triton instead)
        # if weight_data_type == "bf16":
        #     val = val.view(torch.int16)
        #     val = val.detach().cpu().numpy()
        # else:
        #     val = val.to(pyt_weight_data_type(weight_data_type))
        #     val = val.detach().cpu().numpy().astype(np_weight_data_type(weight_data_type))
        
        val = val.to(pyt_weight_data_type(weight_data_type))
        val = val.detach().cpu().numpy().astype(np_weight_data_type(weight_data_type))

        if key.find("shared.weight") != -1:
            # shared weights, only need to convert the weights of rank 0
            saved_path = saved_dir / f"{saved_key}.bin"
            val.tofile(saved_path.as_posix())
        elif key.find("lm_head.decoder.weight") != -1:
            # lm_head weights, only need to convert the weights of rank 0
            saved_path = saved_dir / f"{saved_key}.bin"
            val.tofile(saved_path.as_posix())
        elif key.find("final_logits_bias") != -1:
            # lm_head bias, only need to convert the weights of rank 0
            saved_path = saved_dir / f"{saved_key}.bin"
            val.tofile(saved_path.as_posix())

        elif key.find("layer_norm") != -1 or key.find("layernorm") != -1:
            # shared weights, only need to convert the weights of rank 0
            saved_path = saved_dir / f"{saved_key}.bin"
            val.tofile(saved_path.as_posix())

        # output linear layer weight & FC2 weight, split on first dim
        elif (
                key.find("self_attn.o.weight") != -1
                or key.find("fc2.weight") != -1
        ):
            split_vals = np.split(val, factor, axis=0)
            for j in range(factor):
                saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
                split_vals[j].tofile(saved_path.as_posix())
        
        # output linear layer bias & FC2 bias, share, no split
        elif (
                key.find("self_attn.o.bias") != -1
                or key.find("fc2.bias") != -1
        ):
            # shared weights, only need to convert the weights of rank 0
            saved_path = saved_dir / f"{saved_key}.bin"
            val.tofile(saved_path.as_posix())

        # fc1 weight and bias, split on last dim
        elif key.find("fc1") != -1:
            split_vals = np.split(val, factor, axis=-1)
            for j in range(factor):
                saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
                split_vals[j].tofile(saved_path.as_posix())
        
        # relative attention bias [num_heads, num_buckets], split on fist dim
        elif key.find("relative_attention_bias") != -1:
            # split_vals = np.split(val, factor, axis=0)
            # for j in range(factor):
            #     saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            #     split_vals[j].tofile(saved_path.as_posix())
            # don't split
            saved_path = saved_dir / f"{saved_key}.bin"
            val.tofile(saved_path.as_posix())
        
        # skip decoder self attn, fuse later
        elif (
                key.find("decoder") != -1 and
                (
                        key.find("self_attn.q.weight") != -1
                        or key.find("self_attn.k.weight") != -1
                        or key.find("self_attn.v.weight") != -1
                )
        ):
            pass
        elif key.find("decoder.embed_tokens.weight") != -1:
            LOGGER.warning(f"Not save {key}, using shared.weight directly.")
        else:
            LOGGER.warning(f"cannot find key '{key}' with shape {val.shape}")

def convert_checkpoint(args):

    saved_dir = Path(args.saved_dir) / f"{args.inference_tensor_para_size:d}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.in_file
    # Load the exact config from the 13B Checkpoint
    # CFG_PATH = "atm-13b/config.json"
    # with open(CFG_PATH) as f:
    #     config_dict = json.load(f)
    #     config = ATMT5Config(**config_dict)
    # cfg = ATMT5Config(**get_model_config(model_name))
    # agm_model = ATMT5CE4ForConditionalGeneration(cfg)
    agm_model = ATMT5CE4ForConditionalGeneration.from_pretrained(model_name)
    agm_model.eval().to(pyt_weight_data_type(args.weight_data_type)).to("cuda")

    # example input for correctness check
    batch_size = 1
    input_len = 9
    inputs = {
        'input_ids': torch.tensor([[108, 588, 4153, 407, 3734, 447, 400, 2030, 407]]).to("cuda"),
        'attention_mask': torch.ones(size=(batch_size, input_len)).to("cuda")
    }
    hf_outputs = agm_model.generate(inputs['input_ids'], max_length=32, num_beams=1)
    print("Precision:", args.weight_data_type)
    print("input ids", inputs['input_ids'])
    print("HF output ids",hf_outputs)

    config = configparser.ConfigParser()

    # AGM reports illegal memory issue only for large model, and claims that do the vocab padding beforehand can solve the issue
    pad_length = 0
    if args.pad_embedding:
        # same logic in AgmDecoding.cc for computing vocab_size_padded_
        raw_vocab_size = agm_model.config.vocab_size
        local_vocab_size = np.ceil(raw_vocab_size / args.inference_tensor_para_size)
        if args.weight_data_type == "fp16" or args.weight_data_type == "bf16":
            local_vocab_size = np.ceil(local_vocab_size / 8) * 8
        agm_model.config.vocab_size = int(local_vocab_size * args.inference_tensor_para_size)
        pad_length = agm_model.config.vocab_size - raw_vocab_size
        LOGGER.warning(f"Vocab size is padded from {raw_vocab_size} to {agm_model.config.vocab_size}")

    agm_model.config.decoder_start_token_id = 2 # suppress warning, doesn't really being used

    config["decoder"] = {}
    if not args.encoder_only:
        for key, val in agm_model.config.to_dict().items():
            config["decoder"][key] = f"{val}"
        config["decoder"]["weight_data_type"] = "fp32" if args.weight_data_type == "fp32" else "fp16" # config.ini should NOT say bf16


    # structure info
    config["structure"] = {}
    config["structure"]["agm_with_bias"] = "true"
    config["structure"]["activation_function"] = f"{agm_model.config.activation_function}"
    if config["structure"]["activation_function"].find("gated") != -1:
        config["structure"]["use_gated_activation"] = "true"
    config["structure"]["position_embedding_type"] = "relative"
    config["structure"]["scale_attention"] = f"{agm_model.config.scale_attention}"
    config["structure"]["mup_scale"] = f"{agm_model.config.mup_scale}"
    config["structure"]["pad_embedding"] = f"{args.pad_embedding}"

    with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)

    i_gpu_num = args.inference_tensor_para_size

    split_and_convert_process(agm_model, i_gpu_num, saved_dir, args.weight_data_type, args.pad_embedding, pad_length)

    if not args.encoder_only:
        fuse_decoder_qkv(agm_model, i_gpu_num, saved_dir, args.weight_data_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file", required=True)
    parser.add_argument("-inference_tensor_para_size", "-i_g", type=int, help="How many gpus for inference",
                        required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 4)",
                        default=4)
    parser.add_argument("-weight_data_type", "-w", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--pad_embedding", action="store_true")
    parser.add_argument("--encoder_only", "-e", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Provide verbose messages")
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=log_format)
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))
