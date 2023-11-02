/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/triton_backend/agm/AgmTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/triton_backend/triton_utils.hpp"
#include "src/fastertransformer/utils/Tensor.h"
#include <vector>

namespace ft = fastertransformer;

template<typename T>
void triton_stream_callback(ft::TensorMap* output_tensors, void* ctx)
{
    auto* const model  = reinterpret_cast<AgmTritonModelInstance<T>*>(ctx);
    auto const  result = AgmTritonModelInstance<T>::convert_outputs(*output_tensors);

    model->stream_cb_(result, model->stream_ctx_);
}

template<typename T>
AgmTritonModelInstance<T>::AgmTritonModelInstance(std::unique_ptr<ft::AgmDecoding<T>>       agm_decoding,
                                                  std::shared_ptr<ft::AgmDecodingWeight<T>> agm_decoding_weight,
                                                  std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                                                  std::unique_ptr<ft::cublasAlgoMap>   cublas_algo_map,
                                                  std::unique_ptr<std::mutex>          cublas_wrapper_mutex,
                                                  std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper,
                                                  std::unique_ptr<cudaDeviceProp>      cuda_device_prop_ptr):
    agm_decoding_(std::move(agm_decoding)),
    agm_decoding_weight_(agm_decoding_weight),
    allocator_(std::move(allocator)),
    cublas_algo_map_(std::move(cublas_algo_map)),
    cublas_wrapper_mutex_(std::move(cublas_wrapper_mutex)),
    cublas_wrapper_(std::move(cublas_wrapper)),
    cuda_device_prop_ptr_(std::move(cuda_device_prop_ptr))
{
}

template<typename T>
ft::TensorMap AgmTritonModelInstance<T>::convert_inputs(
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    move_tensor_H2D(input_tensors->at("input_ids"), d_input_ids_, &allocator_);
    move_tensor_H2D(input_tensors->at("sequence_length"), d_input_lengths_, &allocator_);

    ft::TensorMap ft_input_tensors(
        {{"input_ids", as_GPU_tensor(input_tensors->at("input_ids"), d_input_ids_)},
         {"sequence_length", as_GPU_tensor(input_tensors->at("sequence_length"), d_input_lengths_)}});

    if (input_tensors->count("prompt_learning_task_name_ids")) {
        ft_input_tensors.insert({"prompt_learning_task_name_ids",
                                 input_tensors->at("prompt_learning_task_name_ids").convertTritonTensorToFt()});
    }
    if (input_tensors->count("request_prompt_lengths")) {
        move_tensor_H2D(input_tensors->at("request_prompt_lengths"), d_request_prompt_lengths_, &allocator_);
        ft_input_tensors.insert(
            {"request_prompt_lengths",
             as_GPU_tensor(input_tensors->at("request_prompt_lengths"), d_request_prompt_lengths_)});
    }
    if (input_tensors->count("request_prompt_embedding")) {
        move_tensor_H2D(input_tensors->at("request_prompt_embedding"), d_request_prompt_embedding_, &allocator_);
        ft_input_tensors.insert(
            {"request_prompt_embedding",
             as_GPU_tensor(input_tensors->at("request_prompt_embedding"), d_request_prompt_embedding_)});
    }
    if (input_tensors->count("ia3_tasks")) {
        ft_input_tensors.insert({"ia3_tasks", as_GPU_tensor(input_tensors->at("ia3_tasks"), d_input_ia3_tasks_)});
    }
    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
AgmTritonModelInstance<T>::convert_outputs(ft::TensorMap& output_tensors)
{
    std::unordered_map<std::string, triton::Tensor>* outputs_mapping =
        new std::unordered_map<std::string, triton::Tensor>();

    for (auto it = output_tensors.begin(); it != output_tensors.end(); it++) {
        outputs_mapping->insert({it->first, triton::Tensor::convertFtTensorToTriton(it->second)});
    }

    return std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(outputs_mapping);
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
AgmTritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    const size_t request_batch_size = input_tensors->at("input_ids").shape[0];
    const size_t mem_max_seq_len    = input_tensors->at("input_ids").shape[1];
    const size_t max_output_len     = *((uint*)input_tensors->at("max_output_len").data);
    const size_t beam_width =
        input_tensors->count("beam_width") ? (size_t)(*(uint*)input_tensors->at("beam_width").data) : 1;
    const bool has_ia3_tasks = input_tensors->count("ia3_tasks");

    allocateBuffer(request_batch_size, beam_width, max_output_len, mem_max_seq_len);

    if (has_ia3_tasks) {
        move_tensor_H2D(input_tensors->at("ia3_tasks"), d_input_ia3_tasks_, &allocator_);
    }

    ft::TensorMap input_tensors_gpu = convert_inputs(input_tensors);

    ft::TensorMap decoding_input_tensors(
        {{"encoder_output",
          ft::Tensor{ft::MEMORY_GPU,
                     ft::getTensorType<T>(),
                     std::vector<size_t>{request_batch_size, mem_max_seq_len, agm_decoding_->getDModel()},
                     d_encoder_outputs_}},
         {"encoder_sequence_length", input_tensors_gpu.at("sequence_length")},
         {"forced_decoder_ids", input_tensors_gpu.at("input_ids")}});

    if (input_tensors->find("top_p_decay") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_decay"), d_top_p_decay_, &allocator_);
        decoding_input_tensors.insert({"top_p_decay", as_GPU_tensor(input_tensors->at("top_p_decay"), d_top_p_decay_)});
    }
    if (input_tensors->find("top_p_min") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_min"), d_top_p_min_, &allocator_);
        decoding_input_tensors.insert({"top_p_min", as_GPU_tensor(input_tensors->at("top_p_min"), d_top_p_min_)});
    }
    if (input_tensors->find("top_p_reset_ids") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_reset_ids"), d_top_p_reset_ids_, &allocator_);
        decoding_input_tensors.insert(
            {"top_p_reset_ids", as_GPU_tensor(input_tensors->at("top_p_reset_ids"), d_top_p_reset_ids_)});
    }
    if (input_tensors->find("no_repeat_ngram_size") != input_tensors->end()) {
        if (input_tensors->at("no_repeat_ngram_size").shape.size() == 1) {
            // expand a single value to [batch size] tensor
            int ngram = ((int*)input_tensors->at("no_repeat_ngram_size").data)[0];
            std::vector<int> ngram_batch(request_batch_size, ngram);
            if (ngram > 0) {
                ft::check_cuda_error(cudaMemcpy(
                    d_ngram_size_, (int*)ngram_batch.data(), sizeof(int) * request_batch_size, cudaMemcpyDefault));
                decoding_input_tensors.insert({"no_repeat_ngram_size",
                                               ft::Tensor{ft::MEMORY_GPU,
                                                          ft::TYPE_INT32,
                                                          std::vector<size_t>{request_batch_size},
                                                          d_ngram_size_}});
            }
        }
        else if (input_tensors->at("no_repeat_ngram_size").shape.size() == request_batch_size) {
            move_tensor_H2D(input_tensors->at("no_repeat_ngram_size"), d_ngram_size_, &allocator_);
            decoding_input_tensors.insert(
                {"no_repeat_ngram_size", as_GPU_tensor(input_tensors->at("no_repeat_ngram_size"), d_ngram_size_)});
        }
        else {
            FT_CHECK_WITH_INFO(false, "No-repeat ngram size must be [1] or [batch size] tensor");
        }
    }

    std::set<std::string> keys_on_gpu = {"input_ids",
                                         "sequence_length",
                                         "bad_words_list",
                                         "stop_words_list",
                                         "ia3_tasks",
                                         "top_p_decay",
                                         "top_p_min",
                                         "top_p_reset_ids",
                                         "no_repeat_ngram_size"};
    for (auto& t : *input_tensors) {
        if (keys_on_gpu.count(t.first) == 0) {
            decoding_input_tensors.insert({t.first, t.second.convertTritonTensorToFt()});
        }
    }

    if (input_tensors->find("bad_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("bad_words_list"), d_input_bad_words_, &allocator_);
        decoding_input_tensors.insert(
            {"bad_words_list", as_GPU_tensor(input_tensors->at("bad_words_list"), d_input_bad_words_)});
    }

    if (input_tensors->find("stop_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("stop_words_list"), d_input_stop_words_, &allocator_);
        decoding_input_tensors.insert(
            {"stop_words_list", as_GPU_tensor(input_tensors->at("stop_words_list"), d_input_stop_words_)});
    }

    ft::TensorMap decoding_output_tensors(
        {{"output_ids",
          ft::Tensor{ft::MEMORY_GPU,
                     ft::TYPE_INT32,
                     std::vector<size_t>{request_batch_size, beam_width, max_output_len},
                     d_output_ids_}},
         {"sequence_length",
          ft::Tensor{ft::MEMORY_GPU,
                     ft::TYPE_INT32,
                     std::vector<size_t>{request_batch_size, beam_width},
                     d_sequence_lengths_}},
         {"is_finished",
          ft::Tensor{
              ft::MEMORY_GPU, ft::TYPE_BOOL, std::vector<size_t>{request_batch_size, beam_width}, d_is_finished_}},
         {"response_input_lengths",
          ft::Tensor{
              ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size, 1}, d_input_prompt_lengths_}}});
    if (input_tensors->count("is_return_log_probs") > 0
        && input_tensors->at("is_return_log_probs").convertTritonTensorToFt().getVal<bool>()) {
        decoding_output_tensors.insert({"output_log_probs",
                                        ft::Tensor{ft::MEMORY_GPU,
                                                   ft::TYPE_FP32,
                                                   std::vector<size_t>{request_batch_size, beam_width, max_output_len},
                                                   d_output_log_probs_}});
        decoding_output_tensors.insert({"cum_log_probs",
                                        ft::Tensor{ft::MEMORY_GPU,
                                                   ft::TYPE_FP32,
                                                   std::vector<size_t>{request_batch_size, beam_width},
                                                   d_cum_log_probs_}});
    }

    // if (has_ia3_tasks) {
    //     const auto num_ia3_tasks = agm_encoder_weight_->getNumIA3Tasks();
    //     FT_CHECK_WITH_INFO(num_ia3_tasks > 0, "Cannot request ia3_tasks, model has no IA3 adapters");
    //     const bool is_within_range = ft::invokeCheckRange<int>(
    //         d_input_ia3_tasks_, request_batch_size, 0, num_ia3_tasks - 1, d_within_range_,
    //         agm_encoder_->getStream());
    //     FT_CHECK_WITH_INFO(is_within_range,
    //                        ft::fmtstr("Requested IA3 tasks aren't in the range [0, %d).", num_ia3_tasks));

    //     decoding_input_tensors.insert({"ia3_tasks", as_GPU_tensor(input_tensors->at("ia3_tasks"),
    //     d_input_ia3_tasks_)});
    // }

    try {
        if (stream_cb_ != nullptr) {
            agm_decoding_->registerCallback(triton_stream_callback<T>, this);
        }

        agm_decoding_->forward(&decoding_output_tensors, &decoding_input_tensors, agm_decoding_weight_.get());

        if (stream_cb_ != nullptr) {
            agm_decoding_->unRegisterCallback();
        }
    }
    catch (...) {
        h_exception_ = std::current_exception();
        decoding_output_tensors.insert(
            {"error_message", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BYTES, {1}, &h_exception_}});
    }

    return convert_outputs(decoding_output_tensors);
}

template<typename T>
AgmTritonModelInstance<T>::~AgmTritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void AgmTritonModelInstance<T>::allocateBuffer(const size_t request_batch_size,
                                               const size_t beam_width,
                                               const size_t max_output_len,
                                               const size_t mem_max_seq_len)
{
    d_output_ids_      = (int*)(allocator_->reMalloc(
        d_output_ids_, sizeof(int) * request_batch_size * beam_width * max_output_len, false));
    d_encoder_outputs_ = (T*)(allocator_->reMalloc(
        d_encoder_outputs_, sizeof(T) * request_batch_size * mem_max_seq_len * agm_decoding_->getDModel(), false));
    d_sequence_lengths_ =
        (int*)(allocator_->reMalloc(d_sequence_lengths_, sizeof(int) * request_batch_size * beam_width, false));
    d_output_log_probs_ = (float*)(allocator_->reMalloc(
        d_output_log_probs_, sizeof(float) * request_batch_size * beam_width * max_output_len, false));
    d_cum_log_probs_    = (float*)(allocator_->reMalloc(
        d_cum_log_probs_, sizeof(float) * request_batch_size * beam_width * max_output_len, false));
    d_within_range_     = (bool*)(allocator_->reMalloc(d_within_range_, sizeof(bool)));
    d_is_finished_ =
        (bool*)(allocator_->reMalloc(d_is_finished_, sizeof(bool) * request_batch_size * beam_width, true));
    d_ngram_size_ = (int*)(allocator_->reMalloc(d_ngram_size_, sizeof(int) * request_batch_size, false));
    d_input_prompt_lengths_ = (int*)(allocator_->reMalloc(d_input_prompt_lengths_, sizeof(int) * request_batch_size, false));
}

template<typename T>
void AgmTritonModelInstance<T>::freeBuffer()
{
    allocator_->free((void**)(&d_encoder_outputs_));
    allocator_->free((void**)(&d_output_ids_));
    allocator_->free((void**)(&d_sequence_lengths_));
    allocator_->free((void**)(&d_output_log_probs_));
    allocator_->free((void**)(&d_cum_log_probs_));
    allocator_->free((void**)(&d_within_range_));
    allocator_->free((void**)(&d_is_finished_));
    allocator_->free((void**)(&d_ngram_size_));
    allocator_->free((void**)(&d_input_prompt_lengths_));
}

template struct AgmTritonModelInstance<float>;
template struct AgmTritonModelInstance<half>;
#ifdef ENABLE_BF16
template struct AgmTritonModelInstance<__nv_bfloat16>;
#endif
