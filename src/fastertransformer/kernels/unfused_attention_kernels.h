/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "src/fastertransformer/utils/Tensor.h"

namespace fastertransformer {

template<typename T>
void invokeAddQKVBiasIA3Transpose(T*           q_buf,
                                  T*           k_buf,
                                  T*           v_buf,
                                  T*           Q,
                                  const T*     bias_Q,
                                  T*           K,
                                  const T*     bias_K,
                                  T*           V,
                                  const T*     bias_V,
                                  const int    batch_size,
                                  const int    seq_len,
                                  const int    head_num,
                                  const int    size_per_head,
                                  const int*   ia3_tasks,
                                  const T*     ia3_key_weights,
                                  const T*     ia3_value_weights,
                                  cudaStream_t stream);

template<typename T, typename T_IN>
struct MaskedSoftmaxParam {
    // Common parameters.
    T*          attention_score = nullptr;  // (batch_size, head_num, q_length, k_length)
    const T_IN* qk              = nullptr;  // (batch_size, head_num, q_length, k_length)
    const T*    attention_mask  = nullptr;  // (batch_size, q_length, k_length)
    int         batch_size      = 0;
    int         q_length        = 0;
    int         k_length        = 0;
    int         num_heads       = 0;
    T           qk_scale        = T(0.0f);

    // Optional parameters that depend on the type of attention.
    // The slopes of the linear position bias of ALiBi.
    const T* linear_bias_slopes         = nullptr;  // (head_num,), optional

    // The relative attn bias information for Agm models. This will be added to qk_buf
    const T* relative_attn_bias_table   = nullptr;  // (head_num, num_buckets) optional
    int num_buckets                     = 0;        // optional
    int max_distance                    = 0;        // optional
};

template<typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN>& param, cudaStream_t stream);

template<typename T>
void invokeTransposeQKV(T*           dst,
                        T*           src,
                        const int    batch_size,
                        const int    seq_len,
                        const int    head_num,
                        const int    size_per_head,
                        const float* scale,
                        const int    int8_mode,
                        cudaStream_t stream);

template<typename T>
void invokeAddQKVBiasIA3RebuildPadding(T*           Q,
                                       const T*     bias_Q,
                                       T*           K,
                                       const T*     bias_K,
                                       T*           V,
                                       const T*     bias_V,
                                       T*           q_buf,
                                       T*           k_buf,
                                       T*           v_buf,
                                       const int    batch_size,
                                       const int    seq_len,
                                       const int    head_num,
                                       const int    size_per_head,
                                       const int    valid_word_num,
                                       const int*   mask_offset,
                                       const int*   ia3_tasks,
                                       const T*     ia3_key_weights,
                                       const T*     ia3_value_weights,
                                       cudaStream_t stream);

template<typename T>
void invokeTransposeAttentionOutRemovePadding(T*           src,
                                              T*           dst,
                                              const int    valid_word_num,
                                              const int    batch_size,
                                              const int    seq_len,
                                              const int    head_num,
                                              const int    size_per_head,
                                              const int*   mask_offset,
                                              const float* scale,
                                              const int    int8_mode,
                                              cudaStream_t stream);

// Prefix Prompt Parameters
template<typename T>
struct PrefixPromptBatchWeightsParam {
    const T**  d_prefix_prompt_batch    = nullptr;
    const int* d_prefix_prompt_lengths  = nullptr;
    const int  max_prefix_prompt_length = 0;
    // l * 2 * hidden_units_ / tensor_para_.world_size_
    const size_t prefix_prompt_layer_offset_per_seq = 0;
};

template<typename T>
void invokeAddFusedQKVBiasTranspose(T*           q_buf,
                                    T*           k_buf,
                                    T*           v_buf,
                                    T*           QKV,
                                    const T*     qkv_bias,
                                    const int*   padding_offset,
                                    const int    batch_size,
                                    const int    seq_len,
                                    const int    token_num,
                                    const int    head_num,
                                    const int    size_per_head,
                                    cudaStream_t stream)
{
    invokeAddFusedQKVBiasTranspose(q_buf,
                                   k_buf,
                                   v_buf,
                                   PrefixPromptBatchWeightsParam<T>{},
                                   QKV,
                                   qkv_bias,
                                   padding_offset,
                                   batch_size,
                                   seq_len,
                                   token_num,
                                   head_num,
                                   size_per_head,
                                   0,
                                   false,
                                   (float*)nullptr,
                                   0,
                                   stream);
}

template<typename T>
void invokeAddFusedQKVBiasTranspose(T*                               q_buf,
                                    T*                               k_buf,
                                    T*                               v_buf,
                                    PrefixPromptBatchWeightsParam<T> param,
                                    T*                               QKV,
                                    const T*                         qkv_bias,
                                    const int*                       padding_offset,
                                    const int                        batch_size,
                                    const int                        seq_len,
                                    const int                        token_num,
                                    const int                        head_num,
                                    const int                        size_per_head,
                                    const int                        rotary_embedding_dim,
                                    const float                      rotary_position_interpolation_factor,
                                    const float                      rotary_position_freq_base,
                                    const int                        neox_rotary_style,
                                    const float*                     scale,
                                    const int                        int8_mode,
                                    cudaStream_t                     stream);

template<typename T>
void invokeTranspose4d(T*           dst,
                       T*           src,
                       const int    local_batch_size,
                       const int    seq_len,
                       const int    size_per_head,
                       const int    local_hidden_units,
                       const int    local_head_num,
                       const int    batch_size,
                       const int    ite,
                       cudaStream_t stream);

template<typename T>
void invokeTranspose4dBatchMajor(T*           k_dst,
                                 T*           v_dst,
                                 const T*     k_src,
                                 const T*     v_src,
                                 const int    local_batch_size,
                                 const int    seq_len,
                                 const int    max_seq_len,
                                 const int    size_per_head,
                                 const int    local_head_num,
                                 cudaStream_t stream);

template<typename T, typename BT>
void invokeAddRelativeAttentionBiasUnaligned(T*           qk_buf,
                                             const BT*    relative_attention_bias,
                                             const int    batch_size,
                                             const int    head_num,
                                             const int    seq_len,
                                             const int    max_seq_len,
                                             cudaStream_t stream);

template<typename T>
void invokeAddRelativeAttentionBias(T*           qk_buf,
                                    const T*     relative_attention_bias,
                                    const int    batch_size,
                                    const int    head_num,
                                    const int    seq_len,
                                    cudaStream_t stream);

template<typename T>
void invokeAddHead3SizeQKVBias(const T*     mm_qkv,
                               const T*     bias_qkv,
                               T*           q_buf_,
                               T*           k_buf_,
                               T*           v_buf_,
                               const int    batch,
                               const int    window_num,
                               const int    window_len,
                               const int    head_num,
                               const int    size_per_head,
                               cudaStream_t stream);

template<typename T>
void invokeMaskedSoftMaxWithRelPosBias(T*           qk_buf,
                                       const T*     attn_mask,
                                       const T*     relative_pos_bias,
                                       const int    batch_size,
                                       const int    num_head,
                                       const int    window_num,
                                       const int    window_len,
                                       const float  qk_scale,
                                       cudaStream_t stream);

template<typename T>
void invokeTransposeAttentions(Tensor& attentions_out, const Tensor& attentions_in, cudaStream_t stream = 0);

}  // namespace fastertransformer
