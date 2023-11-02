/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/agm/AgmDecoding.h"
#include "src/fastertransformer/kernels/beam_search_topk_kernels.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"

namespace fastertransformer {

template<typename T>
void AgmDecoding<T>::initialize()
{
    context_decoder_ = new AgmContextDecoder<T>(0,  // max_batch_size_ * beam_width_,
                                                head_num_,
                                                size_per_head_,
                                                inter_size_,
                                                num_layer_,
                                                expert_num_,
                                                moe_k_,
                                                layernorm_eps_,
                                                moe_layer_index_,
                                                stream_,
                                                cublas_wrapper_,
                                                allocator_,
                                                is_free_buffer_after_forward_,
                                                tensor_para_,
                                                pipeline_para_,
                                                activation_type_,
                                                false,                        // is_qk_buf_float
                                                AttentionType::UNFUSED_MHA,  // attention_type
                                                q_scaling_,
                                                custom_all_reduce_comm_,
                                                enable_custom_all_reduce_,
                                                adapter_config_);

    decoder_ = new AgmDecoder<T>(0,  // max_batch_size_ * beam_width_,
                                 head_num_,
                                 size_per_head_,
                                 inter_size_,
                                 d_model_,
                                 num_layer_,
                                 expert_num_,
                                 moe_k_,
                                 layernorm_eps_,
                                 moe_layer_index_,
                                 stream_,
                                 cublas_wrapper_,
                                 allocator_,
                                 is_free_buffer_after_forward_,
                                 tensor_para_,
                                 pipeline_para_,
                                 activation_type_,
                                 q_scaling_,
                                 custom_all_reduce_comm_,
                                 enable_custom_all_reduce_,
                                 adapter_config_);

    dynamic_decode_layer_ = new DynamicDecodeLayer<DynamicDecodeType>(vocab_size_,
                                                                      vocab_size_padded_,
                                                                      0,  // end_id, deprecated
                                                                      stream_,
                                                                      cublas_wrapper_,
                                                                      allocator_,
                                                                      is_free_buffer_after_forward_,
                                                                      cuda_device_prop_);
}

template<typename T>
void AgmDecoding<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void AgmDecoding<T>::allocateBuffer(size_t batch_size,
                                    size_t beam_width,
                                    size_t max_seq_len,
                                    size_t max_mem_seq_len,
                                    size_t max_input_len,
                                    size_t encoder_d_model)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // Note: To put the start_ids, we use max_seq_len + 1 for ouptut_ids_buf_
    // And to consistent to the output_ids_buf_, some related buffers are also
    // use max_seq_len + 1, but not max_seq_len.
    // This only affects the buffer size, not affect the performance.

    const size_t batchxbeam      = batch_size * beam_width;
    const size_t self_cache_size = (num_layer_ / pipeline_para_.world_size_) * batchxbeam * (max_seq_len + 1)
                                   * (hidden_units_ / tensor_para_.world_size_);
    const size_t mem_cache_size = (num_layer_ / pipeline_para_.world_size_) * batchxbeam * max_mem_seq_len
                                  * (hidden_units_ / tensor_para_.world_size_);

    if (vocab_size_ != vocab_size_padded_) {
        padded_embedding_kernel_ =
            (T*)(allocator_->reMalloc(padded_embedding_kernel_, sizeof(T) * d_model_ * vocab_size_padded_, true));
        padded_embedding_kernel_ptr_ = padded_embedding_kernel_;

        padded_post_decoder_embedding_bias_ =
            (T*)(allocator_->reMalloc(padded_post_decoder_embedding_bias_, sizeof(T) * vocab_size_padded_, true));
        padded_post_decoder_embedding_bias_ptr_ = padded_post_decoder_embedding_bias_;
    }

    tiled_total_padding_count_ =
        (int*)allocator_->reMalloc(tiled_total_padding_count_, batchxbeam * sizeof(int), false);

    decoder_input_buf_  = (T*)(allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * d_model_, false));
    decoder_output_buf_ = (T*)(allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * d_model_, false));
    normed_decoder_output_buf_ =
        (T*)(allocator_->reMalloc(normed_decoder_output_buf_, sizeof(T) * batchxbeam * d_model_, false));
    logits_buf_      = (DynamicDecodeType*)(allocator_->reMalloc(
        logits_buf_, sizeof(DynamicDecodeType) * batchxbeam * vocab_size_padded_, false));
    nccl_logits_buf_ = (DynamicDecodeType*)(allocator_->reMalloc(
        nccl_logits_buf_, sizeof(DynamicDecodeType) * batchxbeam * vocab_size_padded_, false));
    cum_log_probs_   = (float*)(allocator_->reMalloc(cum_log_probs_, sizeof(float) * batchxbeam, false));
    finished_buf_    = (bool*)(allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false));
    h_finished_buf_  = (bool*)realloc(h_finished_buf_, sizeof(bool) * batchxbeam);

    key_cache_ = (T*)(allocator_->reMalloc(key_cache_, sizeof(T) * (2 * self_cache_size + 2 * mem_cache_size), false));
    value_cache_     = key_cache_ + self_cache_size;
    key_mem_cache_   = value_cache_ + self_cache_size;
    value_mem_cache_ = key_mem_cache_ + mem_cache_size;
    if (beam_width > 1) {
        cache_indirections_[0] = (int*)(allocator_->reMalloc(
            cache_indirections_[0], sizeof(int) * batchxbeam * (max_seq_len + 1) * 2, true));
        cache_indirections_[1] = cache_indirections_[0] + batchxbeam * (max_seq_len + 1);
    }
    tiled_encoder_output_ = (T*)(allocator_->reMalloc(
        tiled_encoder_output_, sizeof(T) * batchxbeam * max_mem_seq_len * encoder_d_model, false));
    tiled_encoder_sequence_length_ =
        (int*)(allocator_->reMalloc(tiled_encoder_sequence_length_, sizeof(int) * batchxbeam, false));

    start_ids_buf_ = (int*)(allocator_->reMalloc(start_ids_buf_, sizeof(int) * batch_size, false));
    end_ids_buf_   = (int*)(allocator_->reMalloc(end_ids_buf_, sizeof(int) * batch_size, false));

    output_ids_buf_ =
        (int*)(allocator_->reMalloc(output_ids_buf_, sizeof(int) * batchxbeam * (max_seq_len + 1), false));
    parent_ids_buf_ =
        (int*)(allocator_->reMalloc(parent_ids_buf_, sizeof(int) * batchxbeam * (max_seq_len + 1), false));
    output_ids_transpose_buf_ =
        (int*)(allocator_->reMalloc(output_ids_transpose_buf_, sizeof(int) * batchxbeam * (max_seq_len + 1), false));
    output_log_probs_buf_ =
        (float*)(allocator_->reMalloc(output_log_probs_buf_, sizeof(float) * batchxbeam * (max_seq_len + 1), false));

    tiled_input_lengths_buf_ = (int*)(allocator_->reMalloc(tiled_input_lengths_buf_, sizeof(int) * batchxbeam, false));

    if (max_input_len > 1) {

        tiled_input_ids_buf_ =
            (int*)(allocator_->reMalloc(tiled_input_ids_buf_, sizeof(int) * batchxbeam * max_input_len, false));
        context_decoder_input_buf_  = (T*)(allocator_->reMalloc(
            context_decoder_input_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
        context_decoder_output_buf_ = (T*)(allocator_->reMalloc(
            context_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));

        input_attention_mask_ = (T*)(allocator_->reMalloc(
            input_attention_mask_, sizeof(T) * batchxbeam * max_input_len * max_input_len, false));

        masked_tokens_ = (bool*)(allocator_->reMalloc(masked_tokens_, sizeof(bool) * batchxbeam * (max_seq_len+1), true));
        // cross_attention_mask_ = (T*)(allocator_->reMalloc(
        //     cross_attention_mask_, sizeof(T) * batchxbeam * max_input_len * max_mem_seq_len, false));

        is_allocate_buffer_forced_input_ = true;
    }

    if (using_beam_hyps) {
        // Let beam_hyps_ can record at most 2*beam_width because we
        // may find beam_width finished candidates during generation,
        // and may compare them with unfinifhsed another beam_width candidates
        // during finalization.
        beam_hyps_.output_ids_tgt = (int*)allocator_->reMalloc(
            beam_hyps_.output_ids_tgt, sizeof(int) * batch_size * beam_width * 2 * (max_seq_len + 1), true);
        beam_hyps_.sequence_lengths_tgt = (int*)allocator_->reMalloc(
            beam_hyps_.sequence_lengths_tgt, sizeof(int) * batch_size * beam_width * 2, true);
        beam_hyps_.cum_log_probs =
            (float*)allocator_->reMalloc(beam_hyps_.cum_log_probs, sizeof(float) * batch_size * beam_width * 2, true);
        beam_hyps_.normed_scores =
            (float*)allocator_->reMalloc(beam_hyps_.normed_scores, sizeof(float) * batch_size * beam_width * 2, true);
        beam_hyps_.log_probs = (float*)allocator_->reMalloc(
            beam_hyps_.log_probs, sizeof(float) * batch_size * beam_width * 2 * (max_seq_len + 1), true);
        beam_hyps_.min_normed_scores =
            (float*)allocator_->reMalloc(beam_hyps_.min_normed_scores, sizeof(float) * batch_size, true);
        beam_hyps_.num_beams = (int*)allocator_->reMalloc(beam_hyps_.num_beams, sizeof(int) * batch_size, true);
        beam_hyps_.is_done   = (bool*)allocator_->reMalloc(beam_hyps_.is_done, sizeof(bool) * batch_size, true);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void AgmDecoding<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = nullptr;
            allocator_->free((void**)(&padded_embedding_kernel_));

            padded_post_decoder_embedding_bias_ptr_ = nullptr;
            allocator_->free((void**)(&padded_post_decoder_embedding_bias_));
        }

        allocator_->free((void**)(&tiled_input_lengths_buf_));

        if (is_allocate_buffer_forced_input_) {
            allocator_->free((void**)(&input_attention_mask_));
            // allocator_->free((void**)(&cross_attention_mask_));
            allocator_->free((void**)(&tiled_input_ids_buf_));
            allocator_->free((void**)(&context_decoder_input_buf_));
            allocator_->free((void**)(&context_decoder_output_buf_));

            is_allocate_buffer_forced_input_ = false;
        }

        allocator_->free((void**)(&decoder_input_buf_));
        allocator_->free((void**)(&decoder_output_buf_));
        allocator_->free((void**)(&normed_decoder_output_buf_));
        allocator_->free((void**)(&logits_buf_));
        allocator_->free((void**)(&nccl_logits_buf_));
        allocator_->free((void**)(&cum_log_probs_));
        allocator_->free((void**)(&finished_buf_));
        free(h_finished_buf_);

        allocator_->free((void**)(&key_cache_));
        if (cache_indirections_[0] != nullptr) {
            allocator_->free((void**)(&cache_indirections_)[0]);
        }

        allocator_->free((void**)(&tiled_encoder_output_));
        allocator_->free((void**)(&tiled_encoder_sequence_length_));

        allocator_->free((void**)(&start_ids_buf_));
        allocator_->free((void**)(&end_ids_buf_));

        allocator_->free((void**)(&output_ids_buf_));
        allocator_->free((void**)(&parent_ids_buf_));
        allocator_->free((void**)(&output_ids_transpose_buf_));
        allocator_->free((void**)(&output_log_probs_buf_));
        allocator_->free((void**)(&tiled_total_padding_count_));
        allocator_->free((void**)(&masked_tokens_));

        if (using_beam_hyps) {
            allocator_->free((void**)(&beam_hyps_.output_ids_tgt));
            allocator_->free((void**)(&beam_hyps_.sequence_lengths_tgt));
            allocator_->free((void**)(&beam_hyps_.cum_log_probs));
            allocator_->free((void**)(&beam_hyps_.normed_scores));
            allocator_->free((void**)(&beam_hyps_.log_probs));
            allocator_->free((void**)(&beam_hyps_.min_normed_scores));
            allocator_->free((void**)(&beam_hyps_.num_beams));
            allocator_->free((void**)(&beam_hyps_.is_done));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void AgmDecoding<T>::setStream(cudaStream_t stream)
{
    decoder_->setStream(stream);
    dynamic_decode_layer_->setStream(stream);
    BaseLayer::setStream(stream);
}

template<typename T>
AgmDecoding<T>::AgmDecoding(size_t                              max_batch_size,
                            size_t                              max_seq_len,
                            size_t                              mem_max_seq_len,
                            size_t                              beam_width,
                            size_t                              head_num,
                            size_t                              size_per_head,
                            size_t                              inter_size,
                            size_t                              d_model,
                            size_t                              num_layer,
                            size_t                              vocab_size,
                            size_t                              num_bucket,
                            size_t                              expert_num,
                            size_t                              max_distance,
                            size_t                              moe_k,
                            float                               q_scaling,
                            int                                 start_id,
                            int                                 end_id,
                            float                               beam_search_diversity_rate,
                            size_t                              top_k,
                            float                               top_p,
                            float                               temperature,
                            float                               len_penalty,
                            float                               repetition_penalty,
                            std::vector<int64_t>                moe_layer_index,
                            cudaStream_t                        stream,
                            cublasMMWrapper*                    cublas_wrapper,
                            IAllocator*                         allocator,
                            bool                                is_free_buffer_after_forward,
                            cudaDeviceProp*                     cuda_device_prop,
                            NcclParam                           tensor_para,
                            NcclParam                           pipeline_para,
                            ActivationType                      activation_type,
                            bool                                tie_word_embeddings,
                            std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                            int                                 enable_custom_all_reduce,
                            LinearAdapterConfig const&          adapter_config):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    d_model_(d_model),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    num_bucket_(num_bucket),
    expert_num_(expert_num),
    max_distance_(max_distance),
    moe_k_(moe_k),
    q_scaling_(q_scaling),
    start_id_(start_id),
    end_id_(end_id),
    beam_search_diversity_rate_(beam_search_diversity_rate),
    hidden_units_(head_num_ * size_per_head_),
    top_k_(top_k),
    top_p_(top_p),
    temperature_(temperature),
    len_penalty_(len_penalty),
    repetition_penalty_(repetition_penalty),
    moe_layer_index_(moe_layer_index),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    activation_type_(activation_type),
    tie_word_embeddings_(tie_word_embeddings),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    adapter_config_(adapter_config)
{
    int local_vacab_size = ceil(vocab_size_ / 1.f / tensor_para_.world_size_);
    if (std::is_same<half, T>::value
#ifdef ENABLE_BF16
        || std::is_same<__nv_bfloat16, T>::value
#endif
    ) {
        local_vacab_size = ceil(local_vacab_size / 8.f) * 8;
    }
    vocab_size_padded_ = (size_t)local_vacab_size * tensor_para_.world_size_;
    initialize();
}

template<typename T>
AgmDecoding<T>::AgmDecoding(AgmDecoding<T> const& decoding):
    BaseLayer(decoding),
    head_num_(decoding.head_num_),
    size_per_head_(decoding.size_per_head_),
    inter_size_(decoding.inter_size_),
    d_model_(decoding.d_model_),
    num_layer_(decoding.num_layer_),
    vocab_size_(decoding.vocab_size_),
    num_bucket_(decoding.num_bucket_),
    expert_num_(decoding.expert_num_),
    max_distance_(decoding.max_distance_),
    moe_k_(decoding.moe_k_),
    q_scaling_(decoding.q_scaling_),
    start_id_(decoding.start_id_),
    end_id_(decoding.end_id_),
    beam_search_diversity_rate_(decoding.beam_search_diversity_rate_),
    hidden_units_(decoding.hidden_units_),
    top_k_(decoding.top_k_),
    top_p_(decoding.top_p_),
    temperature_(decoding.temperature_),
    len_penalty_(decoding.len_penalty_),
    repetition_penalty_(decoding.repetition_penalty_),
    moe_layer_index_(decoding.moe_layer_index_),
    vocab_size_padded_(decoding.vocab_size_padded_),
    tensor_para_(decoding.tensor_para_),
    pipeline_para_(decoding.pipeline_para_),
    activation_type_(decoding.activation_type_),
    tie_word_embeddings_(decoding.tie_word_embeddings_),
    custom_all_reduce_comm_(decoding.custom_all_reduce_comm_),
    enable_custom_all_reduce_(decoding.enable_custom_all_reduce_),
    adapter_config_(decoding.adapter_config_)
{
    initialize();
}

template<typename T>
AgmDecoding<T>::~AgmDecoding()
{
    delete decoder_;
    delete dynamic_decode_layer_;
    freeBuffer();
}

template<typename T>
void AgmDecoding<T>::registerCallback(callback_sig* fn, void* ctx)
{
    token_generated_cb_  = fn;
    token_generated_ctx_ = ctx;
}

template<typename T>
void AgmDecoding<T>::unRegisterCallback()
{
    token_generated_cb_  = nullptr;
    token_generated_ctx_ = nullptr;
}

template<typename T>
void AgmDecoding<T>::forward(std::vector<Tensor>*        output_tensors,
                             const std::vector<Tensor>*  input_tensors,
                             const AgmDecodingWeight<T>* decoding_weights)
{
    // input_tensors:
    //      encoder_output [batch_size, mem_max_seq_len, memory_hidden_dimension]
    //      encoder_sequence_length [batch_size]

    // output_tensors:
    //      output_ids [batch_size, beam, max_seq_len]
    //      sequence_length [batch_size, beam], record the number of generated token, except the start token

    // Step is from 1 ~ max_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.

    std::unordered_map<std::string, Tensor> input_tensors_map{{"encoder_output", input_tensors->at(0)},
                                                              {"encoder_sequence_length", input_tensors->at(1)}};

    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"sequence_length", output_tensors->at(1)}};
    forward(&output_tensors_map, &input_tensors_map, decoding_weights);
}

template<typename T>
void AgmDecoding<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                             const std::unordered_map<std::string, Tensor>* input_tensors,
                             const AgmDecodingWeight<T>*                    decoding_weights)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_map(*input_tensors);
    TensorMap output_map(*output_tensors);
    forward(&output_map, &input_map, decoding_weights);
}

template<typename T>
void AgmDecoding<T>::setOutputTensors(TensorMap* output_tensors, TensorMap const* input_tensors)
{
    if (pipeline_para_.rank_ != pipeline_para_.world_size_ - 1) {
        return;
    }

    auto const batch_size       = output_tensors->at("output_ids").shape[0];
    auto const beam_width       = output_tensors->at("output_ids").shape[1];
    auto const sequence_lengths = output_tensors->at("sequence_length").getPtr<int>();
    auto const max_seq_len      = output_tensors->at("output_ids").shape[2];

    if (beam_width > 1) {
        if (using_beam_hyps) {
            beam_hyps_.sequence_lengths_src = sequence_lengths;
            beam_hyps_.parent_ids_src       = parent_ids_buf_;
            beam_hyps_.output_ids_src       = output_ids_buf_;
            beam_hyps_.log_probs_src        = output_log_probs_buf_;
            beam_hyps_.max_seq_len          = max_seq_len;
            beam_hyps_.length_penalty =
                input_tensors->isExist("len_penalty") ? input_tensors->at("len_penalty").getVal<float>() : 0.0f;

            invokeInsertUnfinishedPath(beam_hyps_, finished_buf_, cum_log_probs_, batch_size, beam_width, stream_);
            sync_check_cuda_error();

            invokeFinalize(output_tensors->getPtr<int>("output_ids"),
                           output_tensors->getPtr<int>("sequence_length"),
                           output_tensors->getPtr<float>("cum_log_probs", nullptr),
                           output_tensors->getPtr<float>("output_log_probs", nullptr),
                           beam_hyps_.output_ids_tgt,
                           beam_hyps_.sequence_lengths_tgt,
                           beam_hyps_.normed_scores,
                           beam_hyps_.cum_log_probs,
                           beam_hyps_.log_probs,
                           beam_hyps_.num_beams,
                           beam_width,
                           max_seq_len,
                           batch_size,
                           stream_);
            sync_check_cuda_error();
        }
        else {
            // For beam search, do gather_tree
            invokeGatherTree(output_ids_transpose_buf_,
                             output_tensors->at("sequence_length").getPtr<int>(),
                             max_seq_len,
                             batch_size,
                             beam_width,
                             output_ids_buf_ + batch_size * beam_width,
                             parent_ids_buf_ + batch_size * beam_width,
                             end_ids_buf_,
                             stream_);

            // transpose and take output_parent_ids as inter buffer
            invokeTransposeAxis01(output_tensors->at("output_ids").getPtr<int>(),
                                  output_ids_transpose_buf_,
                                  max_seq_len,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }
    }
    else {
        // For sampling, only transpose the results to output_tensor
        invokeTransposeAxis01(output_tensors->at("output_ids").getPtr<int>(),
                              output_ids_buf_,
                              max_seq_len,
                              batch_size * beam_width,
                              1,
                              stream_);
    }

    // Return the cumulative log probability and log probability if requested.
    if (beam_width == 1 || !using_beam_hyps) {
        if (output_tensors->isExist("output_log_probs")) {
            invokeTransposeAxis01(output_tensors->at("output_log_probs").getPtr<float>(),
                                  output_log_probs_buf_,
                                  max_seq_len,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }
        if (output_tensors->isExist("cum_log_probs")) {
            Tensor cum_log_probs = output_tensors->at("cum_log_probs");
            FT_CHECK_WITH_INFO(cum_log_probs.size() == batch_size * beam_width,
                               "The shape of cum_log_probs does not match with batch_size x beam_width.");
            cudaD2Dcpy(cum_log_probs.getPtr<float>(), cum_log_probs_, batch_size * beam_width);
        }
    }

    if (output_tensors->isExist("is_finished")) {
        cudaD2Dcpy(
            output_tensors->at("is_finished").getPtr<bool>(), finished_buf_, output_tensors->at("is_finished").size());
    }

    // Add input prompt length for prompt truncation
    if (output_tensors->isExist("response_input_lengths")) {
        cudaD2Dcpy(
                output_tensors->at("response_input_lengths").getPtr<int>(),
                input_tensors->at("encoder_sequence_length").getPtr<int>(),
                input_tensors->at("encoder_sequence_length").size()
                );
    }
}

template<typename T>
void AgmDecoding<T>::sendTensorsToFirstPipelineNode(TensorMap* output_tensors, TensorMap const* input_tensors)
{
    if (pipeline_para_.world_size_ == 1) {
        // throw errors when detected
        ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
        return;
    }

    auto const batch_size  = output_tensors->at("output_ids").shape[0];
    auto const beam_width  = output_tensors->at("output_ids").shape[1];
    auto const max_seq_len = output_tensors->at("output_ids").shape[2];

    ftNcclGroupStart();
    if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
        ftNcclSend(output_tensors->at("output_ids").getPtr<int>(),
                   batch_size * beam_width * max_seq_len,
                   0,
                   pipeline_para_,
                   stream_);

        ftNcclSend(
            output_tensors->at("sequence_length").getPtr<int>(), batch_size * beam_width, 0, pipeline_para_, stream_);

        if (output_tensors->isExist("cum_log_probs") && output_tensors->at("cum_log_probs").data != nullptr) {
            ftNcclSend(output_tensors->at("cum_log_probs").getPtr<float>(),
                       batch_size * beam_width,
                       0,
                       pipeline_para_,
                       stream_);
        }

        if (output_tensors->isExist("output_log_probs") && output_tensors->at("output_log_probs").data != nullptr) {
            ftNcclSend(output_tensors->at("output_log_probs").getPtr<float>(),
                       batch_size * beam_width * max_seq_len,
                       0,
                       pipeline_para_,
                       stream_);
        }
    }
    else if (pipeline_para_.rank_ == 0) {
        ftNcclRecv(output_tensors->at("output_ids").getPtr<int>(),
                   batch_size * beam_width * max_seq_len,
                   pipeline_para_.world_size_ - 1,
                   pipeline_para_,
                   stream_);

        ftNcclRecv(output_tensors->at("sequence_length").getPtr<int>(),
                   batch_size * beam_width,
                   pipeline_para_.world_size_ - 1,
                   pipeline_para_,
                   stream_);

        if (output_tensors->isExist("cum_log_probs") && output_tensors->at("cum_log_probs").data != nullptr) {
            ftNcclRecv(output_tensors->at("cum_log_probs").getPtr<float>(),
                       batch_size * beam_width,
                       pipeline_para_.world_size_ - 1,
                       pipeline_para_,
                       stream_);
        }

        if (output_tensors->isExist("output_log_probs") && output_tensors->at("output_log_probs").data != nullptr) {
            ftNcclRecv(output_tensors->at("output_log_probs").getPtr<float>(),
                       batch_size * beam_width * max_seq_len,
                       pipeline_para_.world_size_ - 1,
                       pipeline_para_,
                       stream_);
        }
    }
    ftNcclGroupEnd();

    // throw errors when detected
    ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
}

template<typename T>
void AgmDecoding<T>::forward(TensorMap*                  output_tensors,
                             TensorMap*                  input_tensors,
                             const AgmDecodingWeight<T>* decoding_weights)
{
    // input_tensors:
    //      encoder_output [batch_size, mem_max_seq_len, memory_hidden_dimension]
    //      encoder_sequence_length [batch_size]
    //      stop_words_list [batch_size, 2, stop_words_length], optional
    //      bad_words_list [batch_size, 2, stop_words_length], optional
    //      forced_decoder_ids [batch_size * beam_width, forced_decoder_ids_length], optional
    //      start_id [batch_size] on cpu, optional
    //      end_id [batch_size] on cpu, optional
    //      runtime_top_k [1] or [batch_size] on cpu, optional, uint.
    //      runtime_top_p [1] or [batch_size] on cpu, optional, float.
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional, float.
    //      temperature [1] or [batch_size] on cpu, optional, float.
    //      len_penalty [1] or [batch_size] on cpu, optional, float.
    //      repetition_penalty [1] or [batch_size] on cpu, optional, float.
    //      presence_penalty [1] or [batch_size] on cpu, optional, float.
    //          Only one of repetition and presence penalties is allowed.
    //      min_length [1] or [batch_size] on cpu, optional, int
    //      no_repeat_ngram_size [1] or [batch_size] on cpu, optional, uint.
    //      random_seed [1] or [batch_size] on cpu, optional, unsigned long long int.
    //      top_p_decay [batch_size] on gpu, float, optional
    //      top_p_min [batch_size] on gpu, float, optional
    //      top_p_reset_ids [batch_size] on gpu, uint32, optional
    //      ia3_tasks [batch_size], optional

    // output_tensors:
    //      output_ids [batch_size, beam, max_seq_len]
    //      is_finised [batch_size, beam_width], optional
    //      sequence_length [batch_size, beam], record the number of generated token, except the start token
    //      output_log_probs [batch_size, beam, max_seq_len], optional, must be float*.
    //      cum_log_probs [batch_size, beam], optional, must be float*.
    //      cross_attentions [num_layer / pipeline_para_size, batch_size, beam,
    //         head_num / tensor_para_size, max_seq_len, mem_max_seq_len], optional, must be float*.

    // Step is from 1 ~ max_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() >= 2);
    FT_CHECK(output_tensors->size() >= 2);
    FT_CHECK(input_tensors->at("encoder_output").shape.size() == 3);
    const size_t batch_size      = output_tensors->at("output_ids").shape[0];
    const size_t beam_width      = output_tensors->at("output_ids").shape[1];
    const size_t max_seq_len     = output_tensors->at("output_ids").shape[2];
    const size_t mem_max_seq_len = input_tensors->at("encoder_output").shape[1];
    const bool   has_ia3_tasks   = input_tensors->isExist("ia3_tasks");

    // allocateBuffer(batch_size, beam_width, max_seq_len, mem_max_seq_len,
    // input_tensors->at("encoder_output").shape[2]);
    const bool has_forced_decoder_ids = input_tensors->isExist("forced_decoder_ids");
    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data
    int max_input_length = has_forced_decoder_ids ? input_tensors->at("forced_decoder_ids").shape[1] : 0;

    allocateBuffer(batch_size,
                   beam_width,
                   max_seq_len,
                   mem_max_seq_len,
                   max_input_length,
                   input_tensors->at("encoder_output").shape[2]);

    {
        TensorMap input_map(*input_tensors);
        dynamic_decode_layer_->setup(batch_size, beam_width, &input_map);
        handleOptArg(&input_map, "start_id", start_ids_buf_, start_id_, batch_size);
        handleOptArg(&input_map, "end_id", end_ids_buf_, end_id_, batch_size);
    }

    FT_CHECK_WITH_INFO(input_tensors->at("encoder_output").shape[2] == d_model_,
                       fmtstr("expect input_tensors->at(\"encoder_output\").shape[2] == d_model_, "
                              "but get input_tensors->at(\"encoder_output\").shape[2] = %d, d_model_ = %d",
                              input_tensors->at("encoder_output").shape[2],
                              d_model_));
    if (has_ia3_tasks) {
        FT_CHECK_WITH_INFO(batch_size == input_tensors->at("ia3_tasks").shape[0],
                           fmtstr("\"ia3_tasks\" tensor has shape [%d], expected [%d]\n",
                                  input_tensors->at("ia3_tasks").shape[0],
                                  batch_size));
    }

    // const int      max_input_length = 1;
    const DataType data_type        = getTensorType<T>();
    int*           sequence_lengths = output_tensors->at("sequence_length").getPtr<int>();

    cudaMemsetAsync(
        output_tensors->at("output_ids").getPtr<int>(), 0, output_tensors->at("output_ids").sizeBytes(), stream_);
    cudaMemsetAsync(output_ids_buf_, 0, sizeof(int) * batch_size * beam_width * (max_seq_len + 1), stream_);
    cudaMemsetAsync(parent_ids_buf_, 0, sizeof(int) * batch_size * beam_width * (max_seq_len + 1), stream_);
    cudaMemsetAsync(masked_tokens_, false, sizeof(bool) * batch_size * beam_width * (max_seq_len+1), stream_);
    cudaMemsetAsync(tiled_total_padding_count_, 0, sizeof(int) * batch_size * beam_width, stream_);
    if (beam_width > 1) {
        cudaMemsetAsync(
            cache_indirections_[0], 0, 2 * sizeof(int) * batch_size * beam_width * (max_seq_len + 1), stream_);
    }

    if (beam_width > 1) {
        invokeTileEncoderResults(tiled_encoder_output_,
                                 tiled_encoder_sequence_length_,
                                 input_tensors->at("encoder_output").getPtr<T>(),
                                 input_tensors->at("encoder_sequence_length").getPtr<const int>(),
                                 batch_size,
                                 beam_width,
                                 mem_max_seq_len,
                                 d_model_,
                                 stream_);
        sync_check_cuda_error();
        encoder_output_ptr_          = tiled_encoder_output_;
        encoder_sequence_length_ptr_ = tiled_encoder_sequence_length_;
    }
    else {
        encoder_output_ptr_          = input_tensors->at("encoder_output").getPtr<const T>();
        encoder_sequence_length_ptr_ = input_tensors->at("encoder_sequence_length").getPtr<const int>();
    }

    const std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    head_num_ / tensor_para_.world_size_,
                                                    size_per_head_ / (16 / sizeof(T)),
                                                    max_seq_len + 1,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    head_num_ / tensor_para_.world_size_,
                                                    (size_t)(max_seq_len + 1),
                                                    size_per_head_};
    const std::vector<size_t> mem_cache_shape    = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    mem_max_seq_len,
                                                    head_num_ / tensor_para_.world_size_ * size_per_head_};

    if (max_input_length > 1) {

        // printf("has_forced_decoder_ids! with max_input_length %d \n", max_input_length);

        std::vector<int> h_input_lengths(batch_size * beam_width, max_input_length);
        int*             input_lengths;
        input_lengths = (int*)allocator_->reMalloc(input_lengths, sizeof(int) * batch_size * beam_width, false);
        cudaMemcpyAsync(input_lengths,
                        h_input_lengths.data(),
                        sizeof(int) * batch_size * beam_width,
                        cudaMemcpyHostToDevice,
                        stream_);
        sync_check_cuda_error();

        invokeTileGptInputs(tiled_input_ids_buf_,
                            tiled_input_lengths_buf_,
                            input_tensors->at("forced_decoder_ids").getPtr<int>(),
                            encoder_sequence_length_ptr_,
                            batch_size,
                            beam_width,
                            max_input_length,
                            stream_);
        sync_check_cuda_error();

        // printf("T5Decoding tiled_input_ids_buf_ debugging: with batch_size %d ===================\n", batch_size);
        // print_to_screen(sequence_lengths, beam_width * batch_size);
        // print_to_screen(tiled_input_ids_buf_, beam_width * batch_size * max_input_length);
        // printf("=========================================\n");

        invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf_,
                                                 output_ids_buf_,
                                                 decoding_weights->pre_decoder_embedding_table,
                                                 decoding_weights->position_embedding_type
                                                         == PositionEmbeddingType::relative ?
                                                     (T*)nullptr :
                                                     decoding_weights->absolute_or_relative_position_embedding,
                                                 pPromptTuningParam<T>{},  // p/prompt tuning
                                                 tiled_input_ids_buf_,
                                                 1,
                                                 max_input_length,
                                                 max_input_length,
                                                 batch_size * beam_width,
                                                 hidden_units_,
                                                 stream_);
        sync_check_cuda_error();

        // printf("T5Decoding finished invokeInputIdsEmbeddingLookupPosEncoding\n");
        // printf("T5Decoding invokeInputIdsEmbeddingLookupPosEncoding debugging: with batch_size %d
        // ===================\n", batch_size); print_to_screen(output_ids_buf_,  (max_seq_len + 1) * batch_size *
        // beam_width); printf("context input embedding (match)=========================================\n");
        // print_to_screen(context_decoder_input_buf_, 10);
        // printf("=========================================\n");
        // print_to_screen(sequence_lengths,  batch_size * beam_width);
        // printf("=========================================\n");
        // print_to_screen(finished_buf_,  batch_size * beam_width);
        // printf("=========================================\n");

        invokeBuildDecoderAttentionMask(input_attention_mask_,
                                        tiled_input_lengths_buf_,
                                        nullptr,  // tiled_prompt_lengths_buf_
                                        batch_size * beam_width,
                                        max_input_length,
                                        0,  // max_prefix_prompt_length
                                        stream_);
        sync_check_cuda_error();

        // No attention masks are needed in cross attention context decoder.
        // invokeBuildDecoderAttentionMask(cross_attention_mask_,
        //                                 tiled_input_lengths_buf_,
        //                                 nullptr, // tiled_prompt_lengths_buf_
        //                                 batch_size * beam_width,
        //                                 max_input_length,
        //                                 0, //max_prefix_prompt_length
        //                                 stream_);
        // sync_check_cuda_error();

        const int context_id_offset     = 0;  // ite * batch_size * beam_width;
        const int context_src_indir_idx = beam_width > 1 ? (max_input_length - 1) & 0x1 : 0;  // step = max_input_length
        uint      context_ite           = 0;

        TensorMap context_decoder_input_tensors({
            {"decoder_input",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                    context_decoder_input_buf_}},  // has_pre_decoder_layernorm: context_decoder_normed_input_buf_
            {"encoder_output",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width,
                     input_tensors->at("encoder_output").shape[1],
                     input_tensors->at("encoder_output").shape[2]},
                    encoder_output_ptr_
                        + context_id_offset * input_tensors->at("encoder_output").shape[1]
                              * input_tensors->at("encoder_output").shape[2]}},
            {"encoder_sequence_length",
             Tensor{
                 MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, encoder_sequence_length_ptr_ + context_id_offset}},
            {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, finished_buf_ + context_id_offset}},
            {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},  // step
            {"sequence_lengths",
             Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, sequence_lengths + context_id_offset}},
            {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &context_ite}},
            {"cache_indirection",
             Tensor{MEMORY_GPU,
                    TYPE_INT32,
                    {batch_size, beam_width, max_seq_len + 1},
                    beam_width > 1 ?
                        cache_indirections_[context_src_indir_idx] + context_id_offset * (max_seq_len + 1) :
                        nullptr}},
            {"attention_mask",
             Tensor(MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width, 1, (size_t)max_input_length, (size_t)max_input_length},
                    input_attention_mask_)},
            {"input_lengths", Tensor{MEMORY_CPU, TYPE_UINT32, {batch_size}, tiled_input_lengths_buf_}},
        });
        if (has_ia3_tasks) {
            context_decoder_input_tensors.insert("ia3_tasks",
                                                 input_tensors->at("ia3_tasks").slice({batch_size}, context_id_offset));
        }
        if (decoding_weights->decoder_layer_weights.at(0)->absolute_or_relative_position_embedding != nullptr){
            context_decoder_input_tensors.insert("num_buckets", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &num_bucket_});
            context_decoder_input_tensors.insert("max_distance",Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_distance_});
        }

        TensorMap context_decoder_output_tensors(
            {{"decoder_output",
              Tensor{MEMORY_GPU,
                     data_type,
                     {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                     context_decoder_output_buf_}},
             {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
             {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}},
             {"key_mem_cache", Tensor{MEMORY_GPU, data_type, mem_cache_shape, key_mem_cache_}},
             {"value_mem_cache", Tensor{MEMORY_GPU, data_type, mem_cache_shape, value_mem_cache_}},
             {"last_token_hidden_units",
              Tensor{MEMORY_GPU, data_type, {batch_size * beam_width, hidden_units_}, decoder_output_buf_}}});

        if (output_tensors->isExist("cross_attentions")) {
            context_decoder_output_tensors.insert("attention_output",
                                                  Tensor{MEMORY_GPU,
                                                         TYPE_FP32,
                                                         output_tensors->at("cross_attentions").shape,
                                                         output_tensors->at("cross_attentions").data,
                                                         {(size_t)context_id_offset,
                                                          batch_size * beam_width * head_num_ / tensor_para_.world_size_
                                                              * max_seq_len * mem_max_seq_len}});
        }

        context_decoder_->forward(
            &context_decoder_output_tensors, &context_decoder_input_tensors, &decoding_weights->decoder_layer_weights);

        // printf("context output =========================================\n");
        // print_to_screen(context_decoder_output_buf_, 10);
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths,
                                 nullptr,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 max_input_length - 1,
                                 stream_);
        sync_check_cuda_error();

        // printf("context_decoder_output_buf_ debugging:===================\n");
        // print_to_screen(context_decoder_output_buf_, hidden_units_ * batch_size * beam_width * max_input_length);
        // printf("=========================================\n");

    }  // single forced input id -> correspond to using force input id to control the start_id ???
    else if (max_input_length == 1) {
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths,
                                 nullptr,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 max_input_length - 1,
                                 stream_);
        sync_check_cuda_error();
        invokeTileGptInputs(tiled_input_ids_buf_,
                            tiled_input_lengths_buf_,
                            input_tensors->at("input_ids").getPtr<int>(),
                            input_tensors->at("input_lengths").getPtr<int>(),
                            batch_size,
                            beam_width,
                            max_input_length,
                            stream_);
        sync_check_cuda_error();

        cudaAutoCpy(output_ids_buf_, tiled_input_ids_buf_, batch_size * beam_width, stream_);
    }
    else if (max_input_length == 0) {  // no forced input ids
        max_input_length++;
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths,
                                 output_ids_buf_,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 max_input_length - 1,
                                 stream_);
        std::vector<int> h_input_lengths_1(batch_size * beam_width, 1);
        cudaMemcpyAsync(tiled_input_lengths_buf_,
                        h_input_lengths_1.data(),
                        sizeof(int) * batch_size * beam_width,
                        cudaMemcpyHostToDevice,
                        stream_);
        for (auto t : h_input_lengths_1) {
            printf("h_input_lengths_1 %d\n", t);
        }
        sync_check_cuda_error();
    }

    // invokeDecodingInitialize(finished_buf_,
    //                          sequence_lengths,
    //                          output_ids_buf_,
    //                          cum_log_probs_,
    //                          start_ids_buf_,
    //                          batch_size,
    //                          beam_width,
    //                          max_input_length - 1,
    //                          stream_);
    // sync_check_cuda_error();

    if (vocab_size_ == vocab_size_padded_) {
        padded_embedding_kernel_ptr_            = decoding_weights->post_decoder_embedding.kernel;
        padded_post_decoder_embedding_bias_ptr_ = decoding_weights->post_decoder_embedding.bias;
    }
    else {
        invokePaddingEmbeddingKernel(padded_embedding_kernel_,
                                     decoding_weights->post_decoder_embedding.kernel,
                                     d_model_,
                                     vocab_size_,
                                     vocab_size_padded_,
                                     stream_);
        sync_check_cuda_error();
        if (decoding_weights->post_decoder_embedding.bias != nullptr) {
            cudaAutoCpy(padded_post_decoder_embedding_bias_,
                        decoding_weights->post_decoder_embedding.bias,
                        vocab_size_,
                        stream_);
        }
    }

    // const std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
    //                                                 batch_size * beam_width,
    //                                                 head_num_ / tensor_para_.world_size_,
    //                                                 size_per_head_ / (16 / sizeof(T)),
    //                                                 max_seq_len + 1,
    //                                                 16 / sizeof(T)};
    // const std::vector<size_t> self_v_cache_shape = {num_layer_ / pipeline_para_.world_size_,
    //                                                 batch_size * beam_width,
    //                                                 head_num_ / tensor_para_.world_size_,
    //                                                 (size_t)(max_seq_len + 1),
    //                                                 size_per_head_};
    // const std::vector<size_t> mem_cache_shape    = {num_layer_ / pipeline_para_.world_size_,
    //                                                 batch_size * beam_width,
    //                                                 mem_max_seq_len,
    //                                                 head_num_ / tensor_para_.world_size_ * size_per_head_};

    const size_t local_batch_size = getLocalBatchSize(batch_size, 1, pipeline_para_.world_size_);
    FT_CHECK(batch_size % local_batch_size == 0);
    const size_t iteration_num = batch_size / local_batch_size;
    invokeMaskPaddingTokens(masked_tokens_,
                        input_tensors->at("encoder_sequence_length").getPtr<const int>(),  // not_tiled
                        nullptr,
                        max_seq_len+1,
                        max_input_length + 0,
                        0,
                        batch_size,
                        beam_width,
                        stream_);

    for (int step = max_input_length; step <= (int)max_seq_len; step++) {
        // printf("step %d\n", step);
        FT_LOG_DEBUG("%s::step: %d", __PRETTY_FUNCTION__, step);
        const int src_indir_idx = beam_width > 1 ? (step - 1) & 0x1 : 0;
        const int tgt_indir_idx = 1 - src_indir_idx;

        for (uint ite = 0; ite < iteration_num; ++ite) {
            const int id_offset               = ite * local_batch_size * beam_width;
            const int d_model_offset          = id_offset * d_model_;
            const int vocab_size_units_offset = id_offset * vocab_size_padded_;


            if (step == max_input_length) {
                /* We have just finished processing input: update the padding count:
                 * total_padding_count += (max_input_length - input_lengths)
                 * if has prefix prompts, += (max_prefix_prompt_length - prompt_length)
                 */
                invokeUpdatePaddingCount(tiled_total_padding_count_,
                                         input_tensors->at("encoder_sequence_length").getPtr<const int>(),  // not_tiled
                                         (const int*)nullptr,
                                         max_input_length,
                                         0,
                                         batch_size,
                                         beam_width,
                                         stream_);
            }

            // Rank 0~N-1 needs to update the buffer by the results of last rank when the pipeline parallelism is
            // enabled (pipeline_para_.world_size_ > 1). And if step == max_input_length, then this is the first step
            // and these buffers are initialized by context directly.
            if (step != max_input_length && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
                && pipeline_para_.world_size_ > 1) {
                ftNcclGroupStart();
                // receive updated sequence_length_ from last rank
                ftNcclRecv(sequence_lengths + id_offset,
                           local_batch_size * beam_width,
                           pipeline_para_.world_size_ - 1,
                           pipeline_para_,
                           stream_);

                // receive updated cache_indirections from last rank
                if (beam_width > 1) {
                    ftNcclRecv(cache_indirections_[tgt_indir_idx] + id_offset * (max_seq_len + 1),
                               local_batch_size * beam_width * (max_seq_len + 1),
                               pipeline_para_.world_size_ - 1,
                               pipeline_para_,
                               stream_);
                }

                // for ids of next step, only first rank needs to receive updated ids
                if (pipeline_para_.rank_ == 0) {
                    ftNcclRecv(output_ids_buf_ + (step - 1) * batch_size * beam_width + id_offset,
                               local_batch_size * beam_width,
                               pipeline_para_.world_size_ - 1,
                               pipeline_para_,
                               stream_);
                }

                ftNcclGroupEnd();
                // throw errors when detected
                ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
                sync_check_cuda_error();
            }

            // only continue to run decoder if keep generating, otherwise step=max_input_length is already completed by context decoder
            if (step > max_input_length) {
                if (pipeline_para_.rank_ == 0) {
                    invokeEmbeddingLookupPosEncodingPadCount(
                        decoder_input_buf_ + d_model_offset,
                        decoding_weights->pre_decoder_embedding_table,
                        decoding_weights->position_embedding_type == PositionEmbeddingType::relative ?
                            (T*)nullptr :
                            decoding_weights->absolute_or_relative_position_embedding,
                        output_ids_buf_ + id_offset,
                        tiled_total_padding_count_ + id_offset,
                        local_batch_size * beam_width,
                        d_model_,
                        (T)1.0f,
                        step - 1,
                        batch_size * beam_width,
                        0,
                        stream_);
                    sync_check_cuda_error();
                }

                if (pipeline_para_.rank_ == 0) {
                    invokeEmbeddingLookupPosEncodingPadCount(
                        decoder_input_buf_ + d_model_offset,
                        decoding_weights->pre_decoder_embedding_table,
                        decoding_weights->position_embedding_type == PositionEmbeddingType::relative ?
                            (T*)nullptr :
                            decoding_weights->absolute_or_relative_position_embedding,
                        output_ids_buf_ + id_offset,
                        nullptr,
                        local_batch_size * beam_width,
                        d_model_,
                        (T)1.0f,
                        step - 1,
                        batch_size * beam_width,
                        0,
                        stream_);
                    sync_check_cuda_error();
                }

                // print_abs_mean(decoder_input_buf_, local_batch_size * beam_width * d_model_, stream_, "decoder
                // input"); print_to_screen(output_ids_buf_, 5); print_to_screen(decoder_input_buf_, 5);

                TensorMap decoder_input_tensors({
                    {"decoder_input",
                        Tensor{MEMORY_GPU,
                        data_type,
                        {local_batch_size * beam_width, d_model_},
                        decoder_input_buf_ + d_model_offset}},
                    {"encoder_output",
                        Tensor{MEMORY_GPU,
                        data_type,
                        {local_batch_size * beam_width, input_tensors->at("encoder_output").shape[1],input_tensors->at("encoder_output").shape[2]},
                        encoder_output_ptr_ + id_offset * input_tensors->at("encoder_output").shape[1] * input_tensors->at("encoder_output").shape[2]}},
                    {"finished",
                        Tensor{MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width}, finished_buf_ + id_offset}},
                    {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
                    {"sequence_lengths",
                        Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width}, sequence_lengths + id_offset}},
                    {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},
                    {"pipeline_iteration", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
                    {"cache_indirection",
                      Tensor{MEMORY_GPU,
                             TYPE_INT32,
                             {local_batch_size, beam_width, max_seq_len + 1},
                             beam_width > 1 ? cache_indirections_[src_indir_idx] + id_offset * (max_seq_len + 1) : nullptr}},
                    {"masked_tokens",
                     Tensor{MEMORY_GPU,
                            TYPE_BOOL,
                            {local_batch_size * beam_width, max_input_length},
                            masked_tokens_ + id_offset * max_input_length}},
                    {"total_padding_tokens",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size * beam_width},
                            tiled_total_padding_count_ + id_offset}}
                });
                if (has_ia3_tasks) {
                    decoder_input_tensors.insert(
                        "ia3_tasks",
                        input_tensors->at("ia3_tasks").slice({local_batch_size}, id_offset));
                }

                if (decoding_weights->decoder_layer_weights.at(0)->absolute_or_relative_position_embedding != nullptr){
                    decoder_input_tensors.insert("num_bucket", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &num_bucket_});
                    decoder_input_tensors.insert("max_distance",Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_distance_});
                }

                TensorMap decoder_output_tensors({
                    {"decoder_output",
                     Tensor{MEMORY_GPU,
                            data_type,
                            {local_batch_size * beam_width, d_model_},
                            decoder_output_buf_ + d_model_offset}},
                    {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
                    {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}},
                    {"key_mem_cache", Tensor{MEMORY_GPU, data_type, mem_cache_shape, key_mem_cache_}},
                    {"value_mem_cache", Tensor{MEMORY_GPU, data_type, mem_cache_shape, value_mem_cache_}}
                });

                decoder_->forward(
                    &decoder_output_tensors, &decoder_input_tensors, &decoding_weights->decoder_layer_weights);
                // print_abs_mean(decoder_output_buf_, local_batch_size * beam_width * d_model_, stream_, "decoder
                // output"); print_to_screen(output_ids_buf_, 5);
            }

            bool agm_with_bias = decoding_weights->agm_with_bias;

            const cudaDataType_t gemm_data_type = getCudaDataType<T>();

            if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                invokeGeneralT5LayerNorm(normed_decoder_output_buf_ + d_model_offset,
                                         decoder_output_buf_ + d_model_offset,
                                         decoding_weights->post_decoder_layernorm.gamma,
                                         decoding_weights->post_decoder_layernorm.beta,
                                         layernorm_eps_,
                                         local_batch_size * beam_width,
                                         d_model_,
                                         stream_);
                sync_check_cuda_error();

                DataType logits_data_type = data_type;

                // bf16 logits computation fallback to fp32
                if (tensor_para_.world_size_ == 1) {
                    float alpha = (!agm_with_bias && tie_word_embeddings_) ? 1.0f / sqrt(d_model_) : 1.0f;
                    float beta  = 0.0f;
#ifdef ENABLE_BF16
                    if (std::is_same<T, __nv_bfloat16>::value) {
                        logits_data_type = TYPE_FP32;
#else
                    if (false) {
#endif
                        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                              CUBLAS_OP_N,
                                              vocab_size_padded_,  // n
                                              local_batch_size * beam_width,
                                              d_model_,  // k
                                              &alpha,
                                              padded_embedding_kernel_ptr_,
                                              gemm_data_type,
                                              d_model_,  // k
                                              normed_decoder_output_buf_ + d_model_offset,
                                              gemm_data_type,
                                              d_model_,  // k
                                              &beta,
                                              logits_buf_ + vocab_size_units_offset,
                                              CUDA_R_32F,
                                              vocab_size_padded_, /* n */
                                              CUDA_R_32F,
                                              cublasGemmAlgo_t(-1));
                    }
                    else {
                        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                              CUBLAS_OP_N,
                                              vocab_size_padded_,  // n
                                              local_batch_size * beam_width,
                                              d_model_,  // k
                                              padded_embedding_kernel_ptr_,
                                              d_model_,  // k
                                              normed_decoder_output_buf_ + d_model_offset,
                                              d_model_,  // k
                                              logits_buf_ + vocab_size_units_offset,
                                              vocab_size_padded_ /* n */,
                                              alpha,
                                              beta);
                    }
                }
                else {
                    const int local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
                    float     alpha = (!agm_with_bias && tie_word_embeddings_) ? 1.0f / sqrt(d_model_) : 1.0f;
                    float     beta  = 0.0f;
#ifdef ENABLE_BF16
                    if (std::is_same<T, __nv_bfloat16>::value) {
                        logits_data_type = TYPE_FP32;
#else
                    if (false) {
#endif
                        cublas_wrapper_->Gemm(
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            local_vocab_size,  // n
                            local_batch_size * beam_width,
                            d_model_,  // k
                            &alpha,
                            padded_embedding_kernel_ptr_ + tensor_para_.rank_ * local_vocab_size * d_model_,
                            gemm_data_type,
                            d_model_,  // k
                            normed_decoder_output_buf_ + d_model_offset,
                            gemm_data_type,
                            d_model_,  // k
                            &beta,
                            nccl_logits_buf_ + vocab_size_units_offset
                                + tensor_para_.rank_ * local_batch_size * beam_width * local_vocab_size,
                            CUDA_R_32F,
                            local_vocab_size, /* n */
                            CUDA_R_32F,
                            cublasGemmAlgo_t(-1));
                    }
                    else {
                        cublas_wrapper_->Gemm(
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            local_vocab_size,  // n
                            local_batch_size * beam_width,
                            d_model_,  // k
                            padded_embedding_kernel_ptr_ + tensor_para_.rank_ * local_vocab_size * d_model_,
                            d_model_,  // k
                            normed_decoder_output_buf_ + d_model_offset,
                            d_model_,  // k
                            nccl_logits_buf_ + vocab_size_units_offset
                                + tensor_para_.rank_ * local_batch_size * beam_width * local_vocab_size,
                            local_vocab_size /* n */,
                            alpha,
                            beta);
                    }
                    ftNcclAllGather(nccl_logits_buf_ + vocab_size_units_offset,
                                    nccl_logits_buf_ + vocab_size_units_offset,
                                    local_batch_size * beam_width * local_vocab_size,
                                    tensor_para_.rank_,
                                    tensor_para_,
                                    stream_);
                    invokeTransposeAxis01(logits_buf_ + vocab_size_units_offset,
                                          nccl_logits_buf_ + vocab_size_units_offset,
                                          tensor_para_.world_size_,
                                          local_batch_size * beam_width,
                                          local_vocab_size,
                                          stream_);
                }

                if (agm_with_bias) {
                    invokeGenericActivation<IdentityActivation, DynamicDecodeType, T>(
                        logits_buf_ + vocab_size_units_offset,
                        padded_post_decoder_embedding_bias_ptr_,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        local_batch_size * beam_width,
                        vocab_size_padded_,
                        0,
                        nullptr,
                        nullptr,
                        stream_);
                }

                int       tmp_local_batch_size       = local_batch_size;
                bool      is_initialize_random_table = step == 1;
                TensorMap dynamic_decode_input_tensors(
                    {{"logits",
                      Tensor{MEMORY_GPU, logits_data_type, {batch_size, beam_width, vocab_size_padded_}, logits_buf_}},
                     {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
                     {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},
                     {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
                     {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, end_ids_buf_}},
                     {"input_lengths",
                      Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, tiled_input_lengths_buf_}},
                     {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_local_batch_size}},
                     {"is_initialize_random_table", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_initialize_random_table}}});

                if (cache_indirections_[src_indir_idx] != nullptr) {
                    dynamic_decode_input_tensors.insert(
                        "src_cache_indirection",
                        Tensor{MEMORY_GPU,
                               TYPE_INT32,
                               {local_batch_size, beam_width, (max_seq_len + 1)},
                               cache_indirections_[src_indir_idx] + id_offset * (max_seq_len + 1)});
                }

                for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
                    if (!dynamic_decode_input_tensors.isExist(t->first)) {
                        dynamic_decode_input_tensors.insert(*t);
                    }
                }

                // common outputs
                TensorMap dynamic_decode_output_tensors(
                    {{"output_ids",
                      Tensor{MEMORY_GPU, TYPE_INT32, {(max_seq_len + 1), batch_size, beam_width}, output_ids_buf_}},
                     {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, finished_buf_}},
                     {"parent_ids",
                      Tensor{MEMORY_GPU, TYPE_INT32, {(max_seq_len + 1), batch_size, beam_width}, parent_ids_buf_}},
                     {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, sequence_lengths}}});

                if (using_beam_hyps) {
                    dynamic_decode_output_tensors.insert("beam_hyps", Tensor{MEMORY_GPU, TYPE_VOID, {1}, &beam_hyps_});
                }

                // cum_log_probs is necessary for beam search, while it is optional for sampling.
                if (beam_width > 1 || output_tensors->isExist("cum_log_probs")) {
                    dynamic_decode_output_tensors.insert(
                        "cum_log_probs", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size * beam_width}, cum_log_probs_});
                }

                if (output_tensors->getPtr<float>("output_log_probs", nullptr) != nullptr) {
                    dynamic_decode_output_tensors.insert(
                        "output_log_probs",
                        Tensor{
                            MEMORY_GPU, TYPE_FP32, {(max_seq_len + 1), batch_size, beam_width}, output_log_probs_buf_});
                }

                if (cache_indirections_[tgt_indir_idx] != nullptr) {
                    dynamic_decode_output_tensors.insert(
                        "tgt_cache_indirection",
                        Tensor{MEMORY_GPU,
                               TYPE_INT32,
                               {local_batch_size, beam_width, (max_seq_len + 1)},
                               cache_indirections_[tgt_indir_idx] + id_offset * (max_seq_len + 1)});
                }

                for (auto t = output_tensors->begin(); t != output_tensors->end(); ++t) {
                    // Handle exceptions.
                    if (t->first == "cum_log_probs" || t->first == "output_log_probs") {
                        continue;
                    }
                    dynamic_decode_output_tensors.insert(*t);
                }

                dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);

                // printf("==========decoding output\n");
                // print_to_screen(output_ids_buf_, 5);
            }
        }

        if (pipeline_para_.world_size_ > 1) {
            ftNcclGroupStart();
            ftNcclBroadCast(output_ids_buf_ + step * batch_size * beam_width,
                            batch_size * beam_width,
                            pipeline_para_.world_size_ - 1,
                            pipeline_para_,
                            stream_);

            ftNcclBroadCast(
                sequence_lengths, batch_size * beam_width, pipeline_para_.world_size_ - 1, pipeline_para_, stream_);

            ftNcclBroadCast(
                finished_buf_, batch_size * beam_width, pipeline_para_.world_size_ - 1, pipeline_para_, stream_);

            if (beam_width > 1) {
                ftNcclBroadCast(cache_indirections_[tgt_indir_idx],
                                batch_size * beam_width * (max_seq_len + 1),
                                pipeline_para_.world_size_ - 1,
                                pipeline_para_,
                                stream_);
            }
            ftNcclGroupEnd();
            // throw errors when detected
            ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
            sync_check_cuda_error();
        }

        cudaD2Hcpy(h_finished_buf_, finished_buf_, batch_size * beam_width);
        uint sum = 0;

        for (uint i = 0; i < batch_size * beam_width; i++) {
            sum += (int)h_finished_buf_[i];
        }
        if (sum == batch_size * beam_width) {
            break;
        }
        else if (step < (int)max_seq_len && token_generated_cb_) {
            setOutputTensors(output_tensors, input_tensors);
            sendTensorsToFirstPipelineNode(output_tensors, input_tensors);
            if (pipeline_para_.rank_ == 0 && tensor_para_.rank_ == 0) {
                token_generated_cb_(output_tensors, token_generated_ctx_);
            }
        }
    }

    setOutputTensors(output_tensors, input_tensors);
    sendTensorsToFirstPipelineNode(output_tensors, input_tensors);

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class AgmDecoding<float>;
template class AgmDecoding<half>;
#ifdef ENABLE_BF16
template class AgmDecoding<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
