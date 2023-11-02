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

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import os


class FTAgmDecodingWeight(object):
    def __init__(
            self,
            config,
            tensor_para_size,
            pipeline_para_size,
            *,
            agm_with_bias=False,
            use_gated_activation=False,
            agm_with_moe=False,
            position_embedding_type=0,
            weight_data_type,
            pad_embedding=True
    ):
        self.config = config
        self.num_layer = config.decoder_layers
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.agm_with_bias = agm_with_bias
        self.use_gated_activation = use_gated_activation
        self.agm_with_moe = agm_with_moe
        self.position_embedding_type = position_embedding_type
        self.real_weights_num = 30  # assume all weights are allocated and converted to specific data type
        self.weight_data_type = weight_data_type
        self.adapter_inter_size = config.adapter_inter_size if hasattr(config, "adapter_inter_size") else 0
        self.w = []
        self.use_mpi = dist.is_mpi_available()
        self.pad_embedding = pad_embedding

        if self.use_mpi:
            try:
                dist.init_process_group(backend='mpi')
            except:
                print("[INFO] WARNING: Exception occurred in dist.init_process_group(backend = 'mpi'). Maybe the process group has been initialized somewhere else.")
        else:
            print("[INFO] MPI is not available in this PyTorch build.")
            assert tensor_para_size == 1, "[FATAL] MPI is required for tensor_para_size > 1."
            assert pipeline_para_size == 1, "[FATAL] MPI is required for pipeline_para_size > 1."

        self.rank = dist.get_rank() if self.use_mpi else 0
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size() if self.use_mpi else 1
        assert world_size == tensor_para_size * \
            pipeline_para_size, "[ERROR] world_size != tensor_para_size * pipeline_para_size"
        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

    def load_from_model(self, model):
        start_layer = self.pipeline_para_rank * self.num_layer // self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer // self.pipeline_para_size

        torch_weight_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.weight_data_type]

        pad_length = 0
        if self.pad_embedding:
            # same logic in AgmDecoding.cc for computing vocab_size_padded_
            raw_vocab_size = self.config.vocab_size
            local_vocab_size = np.ceil(raw_vocab_size / self.tensor_para_size)
            if self.weight_data_type == "fp16" or self.weight_data_type == "bf16":
                local_vocab_size = np.ceil(local_vocab_size / 8) * 8
            new_vocab_size = int(local_vocab_size * self.tensor_para_size)
            pad_length = new_vocab_size - raw_vocab_size
            print(f"[INFO] Vocab size is padded from {raw_vocab_size} to {new_vocab_size}")
        
        weight_dict = {}
        qkv_weight_tmp = ["q", "k", "v"] # must respect this order
        qkv_weight_len = 0
        qkv_bias_tmp = ["q", "k", "v"]
        qkv_bias_len = 0
        for name, param in model.state_dict().items():
            name = name.replace("model.", "")
            if param.dim() == 2:
                param_t = param.transpose(1, 0)
            elif param.dim() == 1:
                param_t = param
            else:
                assert False, f"The dimension of param {name} should be 2"
            if name.find("decoder.layers") != -1:
                if name.find(".self_attn.q.weight") != -1 or name.find(".self_attn.k.weight") != -1 or name.find(".self_attn.v.weight") != -1:
                    qkv_weight_tmp[0 if "q" in name else 1 if "k" in name else 2] = param_t # qkv order in weight dict is not guaranteed
                    qkv_weight_len += 1
                    if qkv_weight_len == 3:
                        qkv_weight = torch.cat(qkv_weight_tmp, dim=-1)
                        weight_dict[name.partition("self_attn")[0] + "self_attn.qkv.weight"] = qkv_weight
                        qkv_weight_tmp = ["q", "k", "v"]
                        qkv_weight_len = 0
                elif name.find(".self_attn.q.bias") != -1 or name.find(".self_attn.k.bias") != -1 or name.find(".self_attn.v.bias") != -1:
                    qkv_bias_tmp[0 if "q" in name else 1 if "k" in name else 2] = param_t # qkv order in weight dict is not guaranteed
                    qkv_bias_len += 1
                    if qkv_bias_len == 3:
                        qkv_bias = torch.cat(qkv_bias_tmp, dim=-1)
                        weight_dict[name.partition("self_attn")[0] + "self_attn.qkv.bias"] = qkv_bias
                        qkv_bias_tmp = ["q", "k", "v"]
                        qkv_bias_len = 0
                else:
                    weight_dict[name] = param_t
            elif name.find("decoder.layernorm_output") != -1 or name.find("decoder.layer_norm") != -1 or name.find("final_logits_bias") != -1:
                weight_dict[name] = param_t
            elif name.find("shared") != -1 or name.find("decoder.embed_positions") != -1 or name.find("lm_head") != -1:
                weight_dict[name] = param

        # load by torch model directly
        # [0]
        t = torch.stack([weight_dict["decoder.layers.{}.self_attn_layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        # [1]
        t = torch.stack([weight_dict["decoder.layers.{}.self_attn.qkv.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.reshape([t.shape[0], t.shape[1], 3, t.shape[2] // 3])
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        # [2]
        t = torch.stack([weight_dict["decoder.layers.{}.self_attn.o.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        # [3]
        t = torch.stack([weight_dict["decoder.layers.{}.final_layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        # t = torch.stack([weight_dict["decoder.layers.{}.EncDecAttention.q.weight".format(i)]
        #                 for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        # t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        # self.w.append(t)
        # t = torch.stack([weight_dict["decoder.layers.{}.EncDecAttention.k.weight".format(i)]
        #                 for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        # t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        # self.w.append(t)
        # t = torch.stack([weight_dict["decoder.layers.{}.EncDecAttention.v.weight".format(i)]
        #                 for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        # t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        # self.w.append(t)
        # t = torch.stack([weight_dict["decoder.layers.{}.EncDecAttention.o.weight".format(i)]
        #                 for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        # t = t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous()
        # self.w.append(t)
        # t = torch.stack([weight_dict["decoder.layers.{}.layer.2.layer_norm.weight".format(i)]
        #                 for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        # self.w.append(t)
        # [4] to [8]
        self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
        self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
        self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
        self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
        self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
        # [9]
        t = torch.stack([weight_dict["decoder.layers.{}.fc1.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        ## [10] empty gated weight
        self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())
        # [11]
        t = torch.stack([weight_dict["decoder.layers.{}.fc2.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        # [12]
        t = weight_dict["decoder.layernorm_output.weight"].contiguous().cuda()
        self.w.append(t)
        # [13]
        t = weight_dict["shared.weight"].contiguous().cuda()
        if pad_length > 0:
            t = torch.cat((t, torch.zeros(pad_length, t.shape[1]).to(t.device)))
        self.w.append(t)
        # [14]
        t = weight_dict["lm_head.decoder.weight"].contiguous().cuda() # don't transpose, keep [vocab, hidden]
        if pad_length > 0:
            t = torch.cat((t, torch.zeros(pad_length, t.shape[1]).to(t.device)))
        self.w.append(t)
        # [15]
        t = torch.stack([weight_dict["decoder.layers.{}.self_attn.relative_attention_bias.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous()
        self.w.append(t)

        if self.agm_with_bias:
            # [16]
            t = torch.stack([weight_dict["decoder.layers.{}.self_attn_layer_norm.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [17]
            # t = torch.stack([weight_dict["decoder.layers.{}.self_attn.qkv.bias".format(i)]
            #                 for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            # t = t.reshape([t.shape[0], 3, t.shape[-1] // 3])
            # t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            # self.w.append(t)
            self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())
            # [18]
            # t = torch.stack([weight_dict["decoder.layers.{}.self_attn.o.bias".format(i)]
            #                 for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            # self.w.append(t)
            self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())
            # [19]
            t = torch.stack([weight_dict["decoder.layers.{}.final_layer_norm.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [20]
            # t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn.q_proj.bias".format(i)]
            #                 for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            # t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            # self.w.append(t)
            # [21]
            # t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn.k_proj.bias".format(i)]
            #                 for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            # t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            # self.w.append(t)
            # [22]
            # t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn.v_proj.bias".format(i)]
            #                 for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            # t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            # self.w.append(t)
            # [23]
            # t = torch.stack([weight_dict["decoder.layers.{}.encoder_attn.out_proj.bias".format(i)]
            #                 for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            # self.w.append(t)
            # [20] to [24]
            self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

            # [25]
            t = torch.stack([weight_dict["decoder.layers.{}.fc1.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            # [26] add empty bias for gated activation for now (BART/mBART model by default don't use gated activation)
            t = torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda()
            self.w.append(t)
            # [27]
            t = torch.stack([weight_dict["decoder.layers.{}.fc2.bias".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            # [28]
            t = weight_dict["decoder.layernorm_output.bias"].contiguous().cuda()
            self.w.append(t)
            # [29] embedding bias aka final_logits_bias (may not exist, keys to ignore)
            if self.pad_embedding:
                t = weight_dict.get("final_logits_bias", torch.zeros((1, new_vocab_size), dtype=torch_weight_dtype)).contiguous().cuda()
            else:
                t = weight_dict.get("final_logits_bias", torch.zeros((1, self.config.vocab_size), dtype=torch_weight_dtype)).contiguous().cuda()
            self.w.append(t)
        else:
            #TODO: pass None Type to Torch Op
            for i in range(14):
                self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
        # adapter weights
        for i in range(9):
                self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

    def load_from_bin(self, ckpt_path, model_type):
        start_layer = self.pipeline_para_rank * self.num_layer //  self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer //  self.pipeline_para_size

        np_weight_dtype = self.weight_data_type
        torch_weight_dtype = {np.float32: torch.float32, np.float16: torch.float16}[np_weight_dtype]
        
        # load by binary files
        if model_type == "Megatron":
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer_norm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.self_attn.qkv.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.self_attn.o.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer_norm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.EncDecAttention.q.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.EncDecAttention.k.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.EncDecAttention.v.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.EncDecAttention.o.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer.2.layer_norm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer.2.DenseReluDense.wi.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            if self.use_gated_activation:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer.2.DenseReluDense.wi2.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
            else:
                self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer.2.DenseReluDense.wo.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.final_layer_norm.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.weight_T.bin", dtype=np_weight_dtype).reshape([self.config.d_model, self.config.vocab_size])).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/lm_head.weight.bin", dtype=np_weight_dtype).reshape(
                [self.config.d_model, self.config.vocab_size])).contiguous().cuda()
            self.w.append(t)
            t = None
            if (self.position_embedding_type == 0):
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.0.self_attn.relative_attention_bias.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)).contiguous().cuda()
            else:
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.ape.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            
            # add 14 additional bias if it is agm megatron structure
            if self.agm_with_bias:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer_norm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.self_attn.qkv.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.self_attn.o.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer_norm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.EncDecAttention.q.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.EncDecAttention.k.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.EncDecAttention.v.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.EncDecAttention.o.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer.2.layer_norm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer.2.DenseReluDense.wi.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                if self.use_gated_activation:
                    t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer.2.DenseReluDense.wi2.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                    self.w.append(t)
                else:
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer.2.DenseReluDense.wo.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.final_layer_norm.bias.bin", dtype=np_weight_dtype)).contiguous().cuda()
                self.w.append(t)
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.bias.bin", dtype=np_weight_dtype)).contiguous().cuda()
                self.w.append(t)
            else:
                #TODO: pass None Type to Torch Op
                for i in range(14):
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            # add empty moe gate weight
            self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

            if self.adapter_inter_size > 0:
                ckpt_path_layers = f"{ckpt_path}/decoder.layers"
                for adapter in ["after_attention_adapter", "after_ffn_adapter"]:
                    for in_out in ["wi", "wo"]:
                        t = torch.stack([torch.from_numpy(np.fromfile(
                            f"{ckpt_path_layers}.{i}.{adapter}.DenseSiluDense.{in_out}.weight.bin",
                            dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                        self.w.append(t)
                    for weight_bias in ["weight", "bias"]:
                        t = torch.stack([torch.from_numpy(np.fromfile(
                            f"{ckpt_path_layers}.{i}.{adapter}.layer_norm.{weight_bias}.bin",
                            dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                        self.w.append(t)
            else:
                for i in range(8):
                    self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())

        else:
            # Megatron-DeepSpeed, no tensor parallelism currently
            #TODO: add tensor parallelism in the conversion script
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.input_layernorm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.attention.query_key_value.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.attention.dense.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.post_attention_layernorm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.query.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.key.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.value.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.dense.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.post_inter_attention_layernorm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            
            # =========== process normal and moe dense layer =================
            t_list = []
            for i in range(start_layer, end_layer):
                if (os.path.isfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_h_to_4h.weight.{self.tensor_para_rank}.bin")):
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_h_to_4h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                else:
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_h_to_4h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
            self.w.append(torch.cat(t_list, 0).contiguous().cuda())
            # ================================================================

            # We don't have use_gated_activation in Megatron-DeepSpeed currently, so here weight placeholder is always empty
            # If we have it in the future, the binary file name should be modified according to the actual name.
            if self.use_gated_activation:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer.2.DenseReluDense.wi2.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
            else:
                self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

            # =========== process normal and moe dense layer =================
            t_list = []
            for i in range(start_layer, end_layer):
                if (os.path.isfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_4h_to_h.weight.{self.tensor_para_rank}.bin")):
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_4h_to_h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                else:
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_4h_to_h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
            self.w.append(torch.cat(t_list, 0).contiguous().cuda())
            # ================================================================

            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.final_layernorm.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/word_embeddings.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            # lm_head weight
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/word_embeddings.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            # assume absolute position
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/position_embeddings.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)

            if self.agm_with_bias:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.input_layernorm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.attention.query_key_value.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.attention.dense.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.post_attention_layernorm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.query.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.key.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.value.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.dense.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.post_inter_attention_layernorm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                
                # =========== process normal and moe dense layer =================
                t_list = []
                for i in range(start_layer, end_layer):
                    if (os.path.isfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_h_to_4h.bias.{self.tensor_para_rank}.bin")):
                        t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_h_to_4h.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                    else:
                        t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_h_to_4h.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                self.w.append(torch.cat(t_list, 0).contiguous().cuda())
                # ================================================================

                # We don't have use_gated_activation in Megatron-DeepSpeed currently, so here weight placeholder is always empty
                # If we have it in the future, the binary file name should be modified according to the actual name.
                if self.use_gated_activation:
                    t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.layer.2.DenseReluDense.wi2.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                    self.w.append(t)
                else:
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

                # =========== process normal and moe dense layer =================
                t_list = []
                for i in range(start_layer, end_layer):
                    if (os.path.isfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_4h_to_h.bias.bin")):
                        t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_4h_to_h.bias.bin", dtype=np_weight_dtype)))
                    else:
                        t_list.append(torch.zeros_like(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_4h_to_h.bias.bin", dtype=np_weight_dtype))))
                self.w.append(torch.cat(t_list, 0).contiguous().cuda())
                # ================================================================

                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.final_layernorm.bias.bin", dtype=np_weight_dtype)).contiguous().cuda()
                self.w.append(t)
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.bias.bin", dtype=np_weight_dtype)).contiguous().cuda()
                self.w.append(t)
            else:
                for i in range(14):
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            
            if self.agm_with_moe:
                gate_list = []
                for i in range(start_layer, end_layer):
                    if (os.path.isfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.gate.wg.weight.bin")):
                        gate_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.gate.wg.weight.bin", dtype=np_weight_dtype)))
                self.w.append(torch.stack(gate_list, 0).contiguous().cuda())
            else:
                self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

            # adapters are not supported in Megatron-DeepSpeed currently, so here weight placeholder is always empty
            for i in range(8):
                self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())

    def to_cuda(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].cuda()

    def to_float(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].float()

    def to_half(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].half()

    def to_single(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].float()

    def to_bfloat16(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].bfloat16()


class FTAgmDecoding(nn.Module):
    def __init__(self, decoding_weight_list, lib_path, head_num, head_size, inter_size,
                 mem_d_model, d_model, num_layer, start_id, end_id, vocab_size, q_scaling=1.0, num_bucket=32,
                 num_expert=0, moe_layer_index=[],
                 max_distance=128, tensor_para_size=1, pipeline_para_size=1, agm_with_bias=False,
                 position_embedding_type=0, moe_k=0,
                 activation_type="relu", tie_word_embeddings=False, adapter_inter_size=0, adapter_norm_position="pre", use_bf16=True):
        super().__init__()

        self.use_mpi = dist.is_mpi_available()
        self.d_model = d_model 
        self.data_type = torch.bfloat16 if use_bf16 else torch.float32

        if self.use_mpi:
            try:
                dist.init_process_group(backend='mpi')
            except:
                print("[INFO] WARNING: Exception occurred in dist.init_process_group(backend = 'mpi'). Maybe the process group has been initialized somewhere else.")
        else:
            print("[INFO] MPI is not available in this PyTorch build.")
            assert tensor_para_size == 1, "[FATAL] MPI is required for tensor_para_size > 1."
            assert pipeline_para_size == 1, "[FATAL] MPI is required for pipeline_para_size > 1."

        torch.classes.load_library(lib_path)
        try:
            self.decoding = torch.classes.FasterTransformer.AgmDecoding(head_num, head_size, inter_size, mem_d_model,
                                                                       d_model, num_layer,
                                                                       vocab_size, num_bucket, num_expert, max_distance,
                                                                       q_scaling, start_id, end_id,
                                                                       tensor_para_size, pipeline_para_size,
                                                                       agm_with_bias,
                                                                       position_embedding_type, moe_k, activation_type,
                                                                       tie_word_embeddings, adapter_inter_size,
                                                                       adapter_norm_position,
                                                                       moe_layer_index, *decoding_weight_list)
        except:
            self.decoding = torch.classes.FasterTransformerAgmDecoding(head_num, head_size, inter_size, mem_d_model,
                                                                      d_model, num_layer,
                                                                      vocab_size, num_bucket, num_expert, max_distance,
                                                                      q_scaling, start_id, end_id,
                                                                      tensor_para_size, pipeline_para_size,
                                                                      agm_with_bias,
                                                                      position_embedding_type, moe_k, activation_type,
                                                                      tie_word_embeddings, adapter_inter_size,
                                                                      adapter_norm_position,
                                                                      moe_layer_index, *decoding_weight_list)

    def forward(self, beam_width, max_seq_len, top_k, top_p,
                beam_search_diversity_rate, temperature,
                len_penalty, repetition_penalty, presence_penalty, min_length, random_seed,
                mem_hidden_states, mem_seq_len,
                is_return_output_log_probs, is_return_cum_log_probs, is_return_cross_attentions=False,
                bad_words_list=None, stop_words_list=None, no_repeat_ngram_size=None, forced_decoder_ids=None):
        # TODO (bhsueh) Not found an method to put a None Type into op forward function
        # So, the top_k and top_p must be some values now.
        results = self.decoding.forward(beam_width, max_seq_len,
                                        top_k, top_p, beam_search_diversity_rate,
                                        temperature, len_penalty, repetition_penalty, presence_penalty, min_length,
                                        random_seed, mem_hidden_states, mem_seq_len,
                                        is_return_output_log_probs, is_return_cum_log_probs, is_return_cross_attentions,
                                        bad_words_list, stop_words_list, no_repeat_ngram_size, forced_decoder_ids)
        return results


class FTAgm(nn.Module):
    def __init__(self, decoding):
        super().__init__()
        self.decoding = decoding

    def forward(self, input_token, attention_mask, inputs_embeds, beam_size, max_seq_len,
                top_k, top_p, beam_search_diversity_rate = 0.0,
                temperature=1.0, len_penalty=0.0, repetition_penalty=None, presence_penalty=None, min_length=0, random_seed=0,
                is_return_output_log_probs=False, is_return_cum_log_probs=False, is_return_cross_attentions=False,
                bad_words_list=None, stop_words_list=None, no_repeat_ngram_size=None, forced_decoder_ids=None):
        input_ids = input_token.to("cuda").type(torch.int32)
        # mem_seq_len = torch.sum(attention_mask, dim=1).type(torch.int32).to("cuda").contiguous()
        mem_seq_len = torch.full([input_ids.shape[0]], input_ids.shape[1]).type(torch.int32).to("cuda").contiguous()

        if type(no_repeat_ngram_size) is int:
            if no_repeat_ngram_size == 0:
                no_repeat_ngram_size = None # passing zero means no ngram control
            else:
                assert no_repeat_ngram_size >= 0, "[FATAL] No repeat ngram size must be >= 0."
                # print("[INFO] Scalar value is given for no repeat ngram size. Auto expanded across batch dimension to apply the same to all.")
                no_repeat_ngram_size = torch.full([input_ids.shape[0]], no_repeat_ngram_size).to(torch.int32).to("cuda").contiguous() 
        elif type(no_repeat_ngram_size) is list:
            assert min(no_repeat_ngram_size) >= 0, "[FATAL] All no repeat ngram size must be >= 0."
            no_repeat_ngram_size = torch.Tensor(no_repeat_ngram_size).to(torch.int32).to("cuda").contiguous() 

        ft_encoder_outputs = torch.zeros((input_ids.shape[0], input_ids.shape[1], self.decoding.d_model), dtype=self.decoding.data_type).to("cuda").contiguous()  ##
        results = self.decoding.forward(beam_size,  # optional, can be None
                                        max_seq_len,
                                        top_k,  # optional, can be None
                                        top_p,  # optional, can be None
                                        beam_search_diversity_rate,  # optional, can be None
                                        temperature,  # optional, can be None
                                        len_penalty,  # optional, can be None
                                        repetition_penalty,  # optional, can be None
                                        presence_penalty,  # optional, can be None
                                        min_length,  # optional, can be None
                                        random_seed,  # optional, can be None
                                        ft_encoder_outputs,
                                        mem_seq_len,
                                        is_return_output_log_probs,  # optional, can be None
                                        is_return_cum_log_probs,  # optional, can be None
                                        is_return_cross_attentions,  # optional, can be None
                                        bad_words_list, # optional, can be None
                                        stop_words_list, # optional, can be None
                                        no_repeat_ngram_size, # optional, can be None
                                        forced_decoder_ids)
        return_dict = {}
        return_dict['output_ids'] = results.pop(0).reshape([-1, beam_size, max_seq_len]).cpu().numpy()
        return_dict['sequence_lengths'] = results.pop(0).reshape([-1, beam_size]).cpu().numpy()
        if is_return_output_log_probs:
            return_dict['output_log_probs'] = results.pop(0).cpu().numpy()
        if is_return_cum_log_probs:
            return_dict['cum_log_probs'] = results.pop(0).cpu().numpy()
        if is_return_cross_attentions:
            return_dict['cross_attentions'] = results.pop(0).cpu().numpy()
            
        return return_dict
