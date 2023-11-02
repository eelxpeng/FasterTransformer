# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

cmake_minimum_required(VERSION 3.8)

add_library(AgmCommon STATIC AgmAdapterWeight.cc)
set_property(TARGET AgmCommon PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET AgmCommon PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(AgmCommon PUBLIC -lcudart LinearAdapterLayer)

add_library(AgmDecoder ${MODEL_LIB_TYPE} AgmDecoder.cc AgmDecoderLayerWeight.cc)
set_property(TARGET AgmDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET AgmDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(AgmDecoder PUBLIC -lcudart cublasMMWrapper AgmCommon TensorParallelDecoderSelfAttentionLayer
        TensorParallelDecoderCrossAttentionLayer TensorParallelReluFfnLayer
        TensorParallelGeluFfnLayer TensorParallelSiluFfnLayer
        layernorm_kernels add_residual_kernels nccl_utils memory_utils tensor cuda_utils logger)

add_library(AgmContextDecoder STATIC AgmContextDecoder.cc)
set_property(TARGET AgmContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET AgmContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(AgmContextDecoder PUBLIC -lcudart cublasMMWrapper
                        TensorParallelGptContextAttentionLayer
                        TensorParallelReluFfnLayer
                        TensorParallelGeluFfnLayer
                        TensorParallelSiluFfnLayer
                        layernorm_kernels
                        add_residual_kernels
                        gpt_kernels
                        tensor
                        nccl_utils
                        cuda_utils
                        logger)

add_library(AgmDecoding ${MODEL_LIB_TYPE} AgmDecoding.cc AgmDecodingWeight.cc)
set_property(TARGET AgmDecoding PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET AgmDecoding PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(AgmDecoding PUBLIC -lcudart cublasMMWrapper AgmDecoder AgmContextDecoder bert_preprocess_kernels
                                        decoding_kernels DynamicDecodeLayer BaseBeamSearchLayer 
                                        beam_search_topk_kernels gpt_kernels tensor cuda_utils logger)
