/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/agm/AgmContextDecoder.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"

#include "src/fastertransformer/layers/TensorParallelGeluFfnLayer.h"
#include "src/fastertransformer/layers/TensorParallelReluFfnLayer.h"
#include "src/fastertransformer/layers/TensorParallelSiluFfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelGptContextAttentionLayer.h"
// #include "src/fastertransformer/layers/attention_layers/TensorParallelDecoderContextCrossAttentionLayer.h"

namespace fastertransformer {

template<typename T>
void AgmContextDecoder<T>::initialize()
{
    self_attention_layer_ = new TensorParallelGptContextAttentionLayer<T>(0,  // max_batch_size
                                                                          0,  // max_seq_len
                                                                          head_num_,
                                                                          size_per_head_,
                                                                          0,      // rotary_embedding_dim_
                                                                          false,  // neox_rotary_style_
                                                                          tensor_para_,
                                                                          stream_,
                                                                          cublas_wrapper_,
                                                                          allocator_,
                                                                          true,  // !use_gptj_residual_ <-- misleading comment...this is do_all_reduce! must be true
                                                                          is_free_buffer_after_forward_,
                                                                          false,  // is_qk_buf_float_
                                                                          false,  // sparse
                                                                          0,      // int8_mode
                                                                          custom_all_reduce_comm_,
                                                                          enable_custom_all_reduce_,
                                                                          q_scaling_);

    // cross attention layer should be based on a context attention cross attention layer as well (like the previous
    // self-attn) cross_attention_layer_ =
    //     new TensorParallelDecoderContextCrossAttentionLayer<T>(0,  // max_batch_size
    //                                                         0,  // max_seq_len
    //                                                         head_num_,
    //                                                         size_per_head_,
    //                                                         0, // rotary_embedding_dim_
    //                                                         false, // neox_rotary_style_
    //                                                         q_scaling_,
    //                                                         tensor_para_,
    //                                                         stream_,
    //                                                         cublas_wrapper_,
    //                                                         allocator_,
    //                                                         false, // !use_gptj_residual_
    //                                                         is_free_buffer_after_forward_,
    //                                                         false, // is_qk_buf_float_
    //                                                         custom_all_reduce_comm_,
    //                                                         enable_custom_all_reduce_);

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU
                                || activation_type_ == ActivationType::SiGLU;
    if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::GeGLU) {
        ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
                                                       1,
                                                       1,
                                                       hidden_units_,
                                                       0,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       false,
                                                       0,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Relu || activation_type_ == ActivationType::ReGLU) {
        ffn_layer_ = new TensorParallelReluFfnLayer<T>(max_batch_size_,
                                                       1,
                                                       1,
                                                       hidden_units_,
                                                       0,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       false,
                                                       0,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Silu || activation_type_ == ActivationType::SiGLU) {
        ffn_layer_ = new TensorParallelSiluFfnLayer<T>(max_batch_size_,
                                                       1,
                                                       1,
                                                       hidden_units_,
                                                       0,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       false,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }

    if (has_adapters()) {
        adapter_layer_ = new LinearAdapterLayer<T>(adapter_config_,
                                                   max_batch_size_,
                                                   1,
                                                   hidden_units_,
                                                   tensor_para_,
                                                   stream_,
                                                   cublas_wrapper_,
                                                   allocator_,
                                                   is_free_buffer_after_forward_,
                                                   sparse_,
                                                   custom_all_reduce_comm_,
                                                   enable_custom_all_reduce_,
                                                   layernorm_eps_);
    }
}

template<typename T>
void AgmContextDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void AgmContextDecoder<T>::allocateBuffer(size_t batch_size, size_t seq_len, size_t max_seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    decoder_normed_input_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    self_attn_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(self_attn_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    normed_self_attn_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(normed_self_attn_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    cross_attn_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(cross_attn_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    normed_cross_attn_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(normed_cross_attn_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    decoder_layer_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    // ContextDecoder additional ptrs
    h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);
    padding_offset_ =
        reinterpret_cast<int*>(allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false));
    cu_seqlens_ = reinterpret_cast<int*>(allocator_->reMalloc(cu_seqlens_, sizeof(int) * (batch_size + 1), false));
    // for moe
    expert_scales_ = reinterpret_cast<T*>(
        allocator_->reMalloc(expert_scales_, sizeof(T) * pad_to_multiple_of_16(moe_k_ * batch_size), false));
    expanded_source_row_to_expanded_dest_row_ = reinterpret_cast<int*>(allocator_->reMalloc(
        expanded_source_row_to_expanded_dest_row_, sizeof(int) * pad_to_multiple_of_16(moe_k_ * batch_size), false));
    expert_for_source_row_                    = reinterpret_cast<int*>(
        allocator_->reMalloc(expert_for_source_row_, sizeof(int) * pad_to_multiple_of_16(moe_k_ * batch_size), false));
    fc2_result_ = reinterpret_cast<T*>(
        allocator_->reMalloc(fc2_result_,
                             sizeof(T) * pad_to_multiple_of_16(moe_k_ * batch_size * hidden_units_),
                             false));  // batch_size * seq_len * hidden_units_ ???

    is_allocate_buffer_ = true;
}

template<typename T>
void AgmContextDecoder<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_ == true) {
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&normed_self_attn_output_));
        allocator_->free((void**)(&cross_attn_output_));
        allocator_->free((void**)(&normed_cross_attn_output_));
        allocator_->free((void**)(&decoder_layer_output_));

        // ContextDecoder additional ptrs
        allocator_->free((void**)(&h_pinned_token_num_ptr_), true);
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&cu_seqlens_));

        allocator_->free((void**)(&expert_scales_));
        allocator_->free((void**)(&expanded_source_row_to_expanded_dest_row_));
        allocator_->free((void**)(&expert_for_source_row_));
        allocator_->free((void**)(&fc2_result_));

        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool AgmContextDecoder<T>::isValidBatchSize(size_t batch_size)
{
    if (batch_size <= max_batch_size_) {
        return true;
    }
    else {
        freeBuffer();
        max_batch_size_ = batch_size * 1.2;
        return true;
    }
}

template<typename T>
bool AgmContextDecoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool AgmContextDecoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool AgmContextDecoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int AgmContextDecoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
AgmContextDecoder<T>::AgmContextDecoder(size_t                              max_batch_size,
                                        size_t                              head_num,
                                        size_t                              size_per_head,
                                        size_t                              inter_size,
                                        size_t                              num_layer,
                                        size_t                              expert_num,
                                        size_t                              moe_k,
                                        float                               layernorm_eps,
                                        std::vector<int64_t>                moe_layer_index,
                                        cudaStream_t                        stream,
                                        cublasMMWrapper*                    cublas_wrapper,
                                        IAllocator*                         allocator,
                                        bool                                is_free_buffer_after_forward,
                                        NcclParam                           tensor_para,
                                        NcclParam                           pipeline_para,
                                        ActivationType                      activation_type,
                                        bool                                is_qk_buf_float,
                                        AttentionType                       attention_type,
                                        float                               q_scaling,
                                        std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                                        int                                 enable_custom_all_reduce,
                                        LinearAdapterConfig const&          adapter_config):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    expert_num_(expert_num),
    moe_k_(moe_k),
    moe_layer_index_(moe_layer_index),
    layernorm_eps_(layernorm_eps),
    hidden_units_(head_num_ * size_per_head),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    is_qk_buf_float_(is_qk_buf_float),
    attention_type_(attention_type),
    activation_type_(activation_type),
    q_scaling_(q_scaling),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    adapter_config_{adapter_config}
{
    initialize();
}

template<typename T>
AgmContextDecoder<T>::AgmContextDecoder(AgmContextDecoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    expert_num_(decoder.expert_num_),
    moe_layer_index_(decoder.moe_layer_index_),
    moe_k_(decoder.moe_k_),
    layernorm_eps_(decoder.layernorm_eps_),
    hidden_units_(decoder.hidden_units_),
    tensor_para_(decoder.tensor_para_),
    pipeline_para_(decoder.pipeline_para_),
    is_qk_buf_float_(decoder.is_qk_buf_float_),
    attention_type_(decoder.attention_type_),
    activation_type_(decoder.activation_type_),
    q_scaling_(decoder.q_scaling_),
    custom_all_reduce_comm_(decoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(decoder.enable_custom_all_reduce_),
    adapter_config_(decoder.adapter_config_)
{
    initialize();
}

template<typename T>
AgmContextDecoder<T>::~AgmContextDecoder()
{
    delete self_attention_layer_;
    // delete cross_attention_layer_;
    delete ffn_layer_;
    if (adapter_layer_ != nullptr) {
        delete adapter_layer_;
        adapter_layer_ = nullptr;
    }
    freeBuffer();
}

template<typename T>
void AgmContextDecoder<T>::forward(TensorMap*                                    output_tensors,
                                   const TensorMap*                              input_tensors,
                                   const std::vector<AgmDecoderLayerWeight<T>*>* decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [batch_size, seq_len, hidden_units_],
    //      encoder_output [batch_size, mem_max_seq_len, mem_hidden_units_],
    //      encoder_sequence_length [batch_size],
    //      finished [batch_size],
    //      step [1] on cpu
    //      sequence_lengths [batch_size]
    //      ite [1] on cpu
    //      cache_indirection [batch_size / beam_width, beam_width, max_seq_len]
    //              Here, batch_size contains the beam_width, so batch_size / beam_width
    //              is real batch_size.
    //      attention_mask [batch_size, 1, seq_len, seq_len]
    //      input_lengths [batch_size]
    //      ia3_tasks [batch_size], optional
    //      with_bias [1] on cpu

    // output tensors:
    //      decoder_output [batch_size, seq_len, hidden_units_],
    //      key_cache [num_layer / pipeline_para_.world_size_, batch, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer / pipeline_para_.world_size_, batch, head_num, max_seq_len, size_per_head]
    //      key_mem_cache [num_layer / pipeline_para_.world_size_, batch_size, mem_max_seq_len, hidden_dimension],
    //      value_mem_cache [num_layer / pipeline_para_.world_size_, batch_size, mem_max_seq_len, hidden_dimension]
    //      attention_output: shape = [num_layer / pipeline_para_.world_size_, batch_size, beam,
    //          head_num / tensor_para_.world_size_, max_seq_len, mem_max_seq_len]
    //          offset = [batch_offset, layer_offset_base] optional, float*

    isValidBatchSize(input_tensors->at("decoder_input").shape[0]);
    const size_t batch_size = input_tensors->at("decoder_input").shape[0];
    const size_t seq_len    = input_tensors->at("decoder_input").shape[1];
    const uint max_seq_len = output_tensors->at("key_cache").shape[4];
    allocateBuffer(batch_size, seq_len, max_seq_len);

    const size_t   mem_max_seq_len = input_tensors->at("encoder_output").shape[1];
    const uint     ite             = input_tensors->at("ite").getVal<uint>();
    const DataType data_type       = getTensorType<T>();
    const bool     has_ia3         = input_tensors->isExist("ia3_tasks");

    const int local_batch_size = getLocalBatchSize(batch_size, seq_len, pipeline_para_.world_size_);
    FT_CHECK(batch_size % local_batch_size == 0);
    const int iteration_num = batch_size / local_batch_size;

    // T*         decoder_input           = input_tensors->at("decoder_input").getPtr<T>();
    T*         decoder_output_final          = output_tensors->at("decoder_output").getPtr<T>();
    const T* attention_mask = input_tensors->at("attention_mask").getPtr<const T>();

    std::vector<size_t> self_k_cache_shape;
    self_k_cache_shape.push_back(local_batch_size);
    for (auto t = output_tensors->at("key_cache").shape.begin() + 2; t != output_tensors->at("key_cache").shape.end();
         ++t) {
        self_k_cache_shape.push_back(*t);
    }
    std::vector<size_t> self_v_cache_shape;
    self_v_cache_shape.push_back(local_batch_size);
    for (auto t = output_tensors->at("value_cache").shape.begin() + 2;
         t != output_tensors->at("value_cache").shape.end();
         ++t) {
        self_v_cache_shape.push_back(*t);
    }

    const std::vector<size_t> mem_cache_shape = {
        local_batch_size, output_tensors->at("key_mem_cache").shape[2], output_tensors->at("key_mem_cache").shape[3]};

    AttentionType attention_type  = attention_type_;
    const bool    is_unpadded_mha = isUnPaddedMHA(attention_type);

    for (int ite = 0; ite < iteration_num; ite++) {
        size_t h_token_num = local_batch_size * seq_len;
        if (is_unpadded_mha) {
            const int* base_input_lengths = input_tensors->at("input_lengths").getPtr<int>();
            // printf("invokeGetPaddingOffsetAndCuSeqLens with local_batch_size%d\n", local_batch_size);
            invokeGetPaddingOffsetAndCuSeqLens(h_pinned_token_num_ptr_,
                                               &h_token_num,
                                               padding_offset_,
                                               cu_seqlens_,
                                               base_input_lengths + ite * local_batch_size,
                                               local_batch_size,
                                               seq_len,
                                               stream_);
            sync_check_cuda_error();
        }

        for (uint l = 0; l < num_layer_; l++) {
            if (isValidLayerParallelId(l) == false) {
                continue;
            }

            if (l == 0 && is_unpadded_mha) {
                invokeRemovePadding(decoder_layer_output_,
                                    input_tensors->at("decoder_input").getPtr<T>()
                                        + ite * local_batch_size * seq_len * hidden_units_,
                                    padding_offset_,
                                    h_token_num,
                                    hidden_units_,
                                    stream_);
            }

            const bool is_final = false;  // TODO(bhsueh) remove this flag

            // Code will accumulate all the Attention + FFN layer outputs to the decoder_layer_output_ buffer which does not have padding
            // Once the loop has completed, padding will be rebuilt and final output tensor will be updated
            // It is critical that accumulations do not happen in the final output tensor or it will mess up the rebuildPadding kernel

            T* decoder_input = decoder_layer_output_;
            T* decoder_output = decoder_layer_output_;


            if (isFirstLayerParallelId(l) == true && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {

                ftNcclRecv(decoder_input + h_token_num * hidden_units_ / tensor_para_.world_size_ * tensor_para_.rank_,
                           h_token_num * hidden_units_ / tensor_para_.world_size_,
                           pipeline_para_.rank_ - 1,
                           pipeline_para_,
                           stream_);
                if (tensor_para_.world_size_ > 1) {
                    ftNcclAllGather(decoder_input,
                                    decoder_input,
                                    (int)h_token_num * hidden_units_ / tensor_para_.world_size_,
                                    tensor_para_.rank_,
                                    tensor_para_,
                                    stream_);
                }
            }

            size_t cache_offset = l - getFirstLayerParallelId();
            for (auto t = output_tensors->at("key_cache").shape.begin() + 1;
                 t != output_tensors->at("key_cache").shape.end();
                 ++t) {
                cache_offset *= *t;
            };
            size_t ite_cache_offset = ite * local_batch_size;
            for (auto t = output_tensors->at("key_cache").shape.begin() + 2;
                 t != output_tensors->at("key_cache").shape.end();
                 ++t) {
                ite_cache_offset *= *t;
            }
            cache_offset += ite_cache_offset;

            size_t mem_cache_offset = l - getFirstLayerParallelId();
            for (auto t = output_tensors->at("key_mem_cache").shape.begin() + 1;
                 t != output_tensors->at("key_mem_cache").shape.end();
                 ++t) {
                mem_cache_offset *= *t;
            };
            ite_cache_offset = ite * local_batch_size;
            for (auto t = output_tensors->at("key_mem_cache").shape.begin() + 2;
                 t != output_tensors->at("key_mem_cache").shape.end();
                 ++t) {
                ite_cache_offset *= *t;
            }
            mem_cache_offset += ite_cache_offset;

            auto const& layer_weight = decoder_layer_weight->at(l);
            invokeGeneralT5LayerNorm(decoder_normed_input_,
                                     decoder_input,
                                     layer_weight->pre_layernorm_weights.gamma,
                                     layer_weight->pre_layernorm_weights.beta,
                                     layernorm_eps_,
                                     h_token_num,
                                     hidden_units_,
                                     stream_);
            sync_check_cuda_error();

            // if (l == 0){
            //         printf("layer 0 context pre LN output=========================================\n");
            //         print_to_screen(decoder_normed_input_, 10);
            // }
            TensorMap self_attention_input_tensors{
                {"input_query", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, decoder_normed_input_}},
                {"finished", input_tensors->at("finished")},
                {"sequence_lengths", input_tensors->at("sequence_lengths")},
                {"step", input_tensors->at("step")},
                {"attention_mask",
                 Tensor{MEMORY_GPU,
                        data_type,
                        {(size_t)local_batch_size, (size_t)1, (size_t)seq_len, (size_t)(seq_len)},
                        attention_mask + local_batch_size * ite * seq_len * (seq_len)}},
                {"attention_type", Tensor{MEMORY_CPU, TYPE_VOID, {1}, &attention_type}},
                {"is_final_layer", Tensor{MEMORY_CPU, TYPE_BOOL, {(size_t)1}, &is_final}},
                {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}}};
            sync_check_cuda_error();

            // Input Tensors for on the fly Relative Attention Bias computation. Bias table shape is [local_head_num, num_buckets]
            self_attention_input_tensors.insert(
                    "relative_attn_bias_table", Tensor{MEMORY_GPU, data_type,
                                {(head_num_ / tensor_para_.world_size_), input_tensors->at("num_buckets").getVal<int>()},
                                layer_weight->absolute_or_relative_position_embedding});
            self_attention_input_tensors.insert("num_buckets", input_tensors->at("num_buckets") );
            self_attention_input_tensors.insert("max_distance", input_tensors->at("max_distance") );

            if (has_ia3) {
                self_attention_input_tensors.insert("ia3_tasks", input_tensors->at("ia3_tasks"));
            }
            if (is_unpadded_mha) {
                self_attention_input_tensors.insert("padding_offset",
                                                    Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num}, padding_offset_});
                self_attention_input_tensors.insert(
                    "cu_seqlens", Tensor{MEMORY_GPU, TYPE_INT32, {size_t(local_batch_size + 1)}, cu_seqlens_});
            }

            TensorMap self_attention_output_tensors{
                {"hidden_features", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, self_attn_output_}},
                {"key_cache",
                 Tensor{MEMORY_GPU,
                        data_type,
                        self_k_cache_shape,
                        output_tensors->at("key_cache").getPtrWithOffset(cache_offset)}},
                {"value_cache",
                 Tensor{MEMORY_GPU,
                        data_type,
                        self_v_cache_shape,
                        output_tensors->at("value_cache").getPtrWithOffset(cache_offset)}}};

            self_attention_layer_->forward(
                &self_attention_output_tensors, &self_attention_input_tensors, &layer_weight->self_attention_weights);

            // if (l == 0){
            //     printf("layer 0 context self attn output=========================================\n");
            //     print_to_screen(self_attn_output_, 10);
            // }
            const T* attention_bias = layer_weight->self_attention_weights.attention_output_weight.bias;

            if (has_adapters() && layer_weight->has_adapters()) {
                if (attention_bias != nullptr) {
                    invokeAddBias(self_attn_output_, attention_bias, h_token_num, hidden_units_, stream_);
                    attention_bias = nullptr;
                }
                Tensor input_tensor{
                    MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, self_attn_output_};
                Tensor output_tensor{
                    MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, self_attn_output_};
                adapter_layer_->forward(
                    &input_tensor, &output_tensor, &layer_weight->adapter_weights_.after_attention_adapter_weights_);
            }

            invokeGeneralAddBiasResidualT5PreLayerNorm(self_attn_output_,
                                                       normed_self_attn_output_,
                                                       decoder_input,
                                                       layer_weight->self_attn_layernorm_weights.gamma,
                                                       layer_weight->self_attn_layernorm_weights.beta,
                                                       attention_bias,
                                                       layernorm_eps_,
                                                       h_token_num,
                                                       hidden_units_,
                                                       stream_);
            sync_check_cuda_error();

            // if (l == 0){
            //     printf("layer 0 context ffn after LN output=========================================\n");
            //     print_to_screen(normed_self_attn_output_, 10);
            // }
            // TensorMap cross_attention_input_tensors{
            //     {"input_query", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_},
            //     normed_self_attn_output_}},
            //     {"encoder_output", input_tensors->at("encoder_output")},
            //     {"encoder_sequence_length", input_tensors->at("encoder_sequence_length")},
            //     {"finished", input_tensors->at("finished")},
            //     {"step", input_tensors->at("step")},
            //     {"attention_mask",
            //         Tensor{MEMORY_GPU,
            //                 data_type,
            //                 {(size_t)local_batch_size, (size_t)1, (size_t)seq_len, (size_t)(seq_len)},
            //                 attention_mask + local_batch_size * ite * seq_len * (seq_len)}},
            //     {"attention_type", Tensor{MEMORY_CPU, TYPE_VOID, {1}, &attention_type}},
            //     {"is_final_layer", Tensor{MEMORY_CPU, TYPE_BOOL, {(size_t)1}, &is_final}},
            //     {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}},
            //     {"with_bias", input_tensors->at("with_bias")}
            //     };
            // if (has_ia3) {
            //     cross_attention_input_tensors.insert("ia3_tasks", input_tensors->at("ia3_tasks"));
            // }
            // TensorMap cross_attention_output_tensors{
            //     {"hidden_features", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, cross_attn_output_}},
            //     {"key_cache",
            //     Tensor{MEMORY_GPU, data_type, mem_cache_shape,
            //     output_tensors->at("key_mem_cache").getPtrWithOffset(mem_cache_offset)}},
            //     {"value_cache",
            //     Tensor{MEMORY_GPU, data_type, mem_cache_shape,
            //     output_tensors->at("value_mem_cache").getPtrWithOffset(mem_cache_offset)}}};
            // if (output_cross_attention) {
            //     int          local_layer_id          = l - getFirstLayerParallelId();
            //     const size_t cross_attentions_offset = local_layer_id *
            //     output_tensors->at("attention_output").offsets[1]
            //                                         + output_tensors->at("attention_output").offsets[0] * head_num_
            //                                                 / tensor_para_.world_size_ * max_seq_len *
            //                                                 mem_max_seq_len;
            //     cross_attention_output_tensors.insert(
            //         "cross_attentions",
            //         Tensor{MEMORY_GPU,
            //             TYPE_FP32,
            //             {local_batch_size, head_num_ / tensor_para_.world_size_, max_seq_len, mem_max_seq_len},
            //             output_tensors->at("attention_output").getPtrWithOffset<float>(cross_attentions_offset)});
            // }

            // cross_attention_layer_->forward(
            //     &cross_attention_output_tensors, &cross_attention_input_tensors,
            //     &layer_weight->cross_attention_weights);

            // if (is_final == false) {
            // invokeGeneralAddBiasResidualT5PreLayerNorm(cross_attn_output_,
            //                                         normed_cross_attn_output_,
            //                                         self_attn_output_,
            //                                         layer_weight->cross_attn_layernorm_weights.gamma,
            //                                         layer_weight->cross_attn_layernorm_weights.beta,
            //                                         layer_weight->cross_attention_weights.attention_output_weight.bias,
            //                                         layernorm_eps_,
            //                                         h_token_num,
            //                                         hidden_units_,
            //                                         stream_);
            // sync_check_cuda_error();

            // TensorMap ffn_input_tensors(
            //     {{"ffn_input", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_},
            //     normed_cross_attn_output_}}});
            TensorMap ffn_input_tensors(
                {{"ffn_input", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, normed_self_attn_output_}}});
            if (has_ia3) {
                ffn_input_tensors.insert("ia3_tasks", input_tensors->at("ia3_tasks"));
            }

            TensorMap ffn_output_tensors;

            bool use_moe = std::find(moe_layer_index_.begin(), moe_layer_index_.end(), l) != moe_layer_index_.end();
            if (use_moe) {
                ffn_input_tensors.insert("moe_k", Tensor{MEMORY_CPU, TYPE_UINT64, {1}, &moe_k_});

                ffn_output_tensors.insert(
                    "ffn_output", Tensor{MEMORY_GPU, data_type, {moe_k_ * h_token_num, hidden_units_}, fc2_result_});
                ffn_output_tensors.insert("expert_scales",
                                          Tensor{MEMORY_GPU, data_type, {h_token_num, moe_k_}, expert_scales_});
                ffn_output_tensors.insert(
                    "expanded_source_row_to_expanded_dest_row",
                    Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num, moe_k_}, expanded_source_row_to_expanded_dest_row_});
                ffn_output_tensors.insert(
                    "expert_for_source_row",
                    Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num, moe_k_}, expert_for_source_row_});
            }
            else {
                ffn_output_tensors.insert("ffn_output",
                                          Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, decoder_output});
            }

            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->ffn_weights);

            if (use_moe) {
                // residual addition for moe, we should pass the unnormed attention output if using pre_layernorm
                // and pass the normed attention output if using post_layernorm. They all point to the attn_out_buf_.
                finalize_moe_routing_kernelLauncher(fc2_result_,
                                                    decoder_output,
                                                    self_attn_output_,
                                                    layer_weight->ffn_weights.output_weight.bias,
                                                    expert_scales_,
                                                    expanded_source_row_to_expanded_dest_row_,
                                                    expert_for_source_row_,
                                                    h_token_num,
                                                    hidden_units_,
                                                    moe_k_,
                                                    stream_);
            }
            else {
                auto* ffn_bias = layer_weight->ffn_weights.output_weight.bias;
                if (has_adapters() && layer_weight->has_adapters()) {
                    if (ffn_bias != nullptr) {
                        invokeAddBias(decoder_output, ffn_bias, h_token_num, hidden_units_, stream_);
                        ffn_bias = nullptr;
                    }
                    Tensor input_tensor{
                        MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, decoder_output};
                    Tensor output_tensor{
                        MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, decoder_output};
                    adapter_layer_->forward(
                        &input_tensor, &output_tensor, &layer_weight->adapter_weights_.after_ffn_adapter_weights_);
                }
                invokeT5AddBiasResidual(
                    decoder_output, self_attn_output_, ffn_bias, h_token_num, hidden_units_, stream_);
            }
            sync_check_cuda_error();

            // if (l == 0){
            //     printf("layer 0 context ffn output=========================================\n");
            //     print_to_screen(decoder_output, 10);
            // }

            if (isLastLayerParallelId(l) == true && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
                && pipeline_para_.world_size_ > 1) {
                // ftNcclSend(decoder_output, local_batch_size * hidden_units_, pipeline_para_.rank_ + 1,
                // pipeline_para_, stream_);
                int data_size = h_token_num * hidden_units_ / tensor_para_.world_size_;

                ftNcclSend(decoder_output + data_size * tensor_para_.rank_,
                           data_size * hidden_units_ / tensor_para_.world_size_,
                           pipeline_para_.rank_ + 1,
                           pipeline_para_,
                           stream_);
            }

            if ((l == num_layer_ - 1) && is_unpadded_mha) {
                invokeRebuildPadding(decoder_output_final + ite * local_batch_size * seq_len * hidden_units_,
                                     decoder_layer_output_,
                                     padding_offset_,
                                     h_token_num,
                                     head_num_ * size_per_head_,
                                     stream_);
            }

            // } // if_final check
        }
    }

    // TODO(bhsueh) We could optimize this point by only computing the last token for the last layer
    invokeLookupHiddenStateOfLastToken(output_tensors->at("last_token_hidden_units").getPtr<T>(),
                                       output_tensors->at("decoder_output").getPtr<T>(),
                                       input_tensors->at("input_lengths").getPtr<int>(),
                                       seq_len,
                                       batch_size,
                                       hidden_units_,
                                       stream_);
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class AgmContextDecoder<float>;
template class AgmContextDecoder<half>;
#ifdef ENABLE_BF16
template class AgmContextDecoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
