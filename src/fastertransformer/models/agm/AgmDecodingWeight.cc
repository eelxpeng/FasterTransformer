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

#include "src/fastertransformer/models/agm/AgmDecodingWeight.h"
#include "src/fastertransformer/utils/IA3.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
AgmDecodingWeight<T>::AgmDecodingWeight(const size_t                head_num,
                                        const size_t                size_per_head,
                                        const size_t                d_model,
                                        const size_t                inter_size,
                                        const size_t                vocab_size,
                                        const size_t                num_layer,
                                        const size_t                mem_d_model,
                                        const size_t                num_bucket_or_max_seq_len,
                                        const size_t                tensor_para_size,
                                        const size_t                tensor_para_rank,
                                        const size_t                pipeline_para_size,
                                        const size_t                pipeline_para_rank,
                                        const bool                  agm_with_bias_para,
                                        const bool                  use_gated_activation_para,
                                        const PositionEmbeddingType pe_type,
                                        const size_t                ia3_num_tasks,
                                        const size_t                adapter_inter_size):
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    mem_d_model_(mem_d_model),
    num_bucket_or_max_seq_len_(num_bucket_or_max_seq_len),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    pipeline_para_size_(pipeline_para_size),
    pipeline_para_rank_(pipeline_para_rank),
    agm_with_bias(agm_with_bias_para),
    use_gated_activation(use_gated_activation_para),
    position_embedding_type(pe_type),
    real_weights_num_(agm_with_bias ? 6 : 4),
    ia3_num_tasks_(ia3_num_tasks),
    adapter_inter_size_(adapter_inter_size)
{
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " start");
    FT_CHECK(num_layer_ % pipeline_para_size_ == 0);
    initialize();
    mallocWeights();
    setWeightPtr();

    decoder_layer_weights.clear();
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            decoder_layer_weights.push_back(new AgmDecoderLayerWeight<T>(head_num_,
                                                                         size_per_head_,
                                                                         d_model_,
                                                                         inter_size_,
                                                                         mem_d_model_,
                                                                         num_bucket_or_max_seq_len_,
                                                                         tensor_para_size_,
                                                                         tensor_para_rank_,
                                                                         agm_with_bias,
                                                                         use_gated_activation,
                                                                         position_embedding_type,
                                                                         ia3_num_tasks_,
                                                                         adapter_inter_size_));
        }
        else {
            decoder_layer_weights.push_back(new AgmDecoderLayerWeight<T>());
        }
    }
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " end");
}

template<typename T>
void AgmDecodingWeight<T>::initialize()
{
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " start");
    weights_size[0] = d_model_ * vocab_size_;
    if (position_embedding_type == PositionEmbeddingType::absolute) {
        weights_size[1] = num_bucket_or_max_seq_len_ * d_model_;
    }
    else {
        weights_size[1] = (head_num_ / tensor_para_size_) * num_bucket_or_max_seq_len_; // not used. it's per decoder
    }
    weights_size[2] = d_model_;
    weights_size[3] = d_model_ * vocab_size_;
    weights_size[4] = d_model_;
    weights_size[5] = vocab_size_;
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " end");
}

template<typename T>
AgmDecodingWeight<T>::~AgmDecodingWeight()
{
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " start");
    if (is_maintain_buffer == true) {
        decoder_layer_weights.clear();
        for (int i = 0; i < real_weights_num_; i++) {
            if (shared_embed_ && i == 3 && weights_ptr[i] == nullptr) {
                continue;
            }
            deviceFree(weights_ptr[i]);
        }

        pre_decoder_embedding_table             = nullptr;
        absolute_or_relative_position_embedding = nullptr;
        post_decoder_layernorm.gamma            = nullptr;
        post_decoder_embedding.kernel           = nullptr;
        post_decoder_embedding.bias             = nullptr;
        post_decoder_layernorm.beta             = nullptr;
        is_maintain_buffer                      = false;
    }
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " end");
}

template<typename T>
AgmDecodingWeight<T>::AgmDecodingWeight(const AgmDecodingWeight& other):
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    d_model_(other.d_model_),
    inter_size_(other.inter_size_),
    vocab_size_(other.vocab_size_),
    num_layer_(other.num_layer_),
    mem_d_model_(other.mem_d_model_),
    num_bucket_or_max_seq_len_(other.num_bucket_or_max_seq_len_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    pipeline_para_size_(other.pipeline_para_size_),
    pipeline_para_rank_(other.pipeline_para_rank_),
    agm_with_bias(other.agm_with_bias),
    use_gated_activation(other.use_gated_activation),
    position_embedding_type(other.position_embedding_type),
    real_weights_num_(other.real_weights_num_),
    ia3_num_tasks_(other.ia3_num_tasks_),
    adapter_inter_size_(other.adapter_inter_size_),
    shared_embed_(other.shared_embed_)
{
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    decoder_layer_weights.clear();
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(new AgmDecoderLayerWeight<T>(*other.decoder_layer_weights[l]));
    }
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " end");
}

template<typename T>
AgmDecodingWeight<T>& AgmDecodingWeight<T>::operator=(const AgmDecodingWeight& other)
{
    head_num_                  = other.head_num_;
    size_per_head_             = other.size_per_head_;
    d_model_                   = other.d_model_;
    inter_size_                = other.inter_size_;
    vocab_size_                = other.vocab_size_;
    num_layer_                 = other.num_layer_;
    mem_d_model_               = other.mem_d_model_;
    num_bucket_or_max_seq_len_ = other.num_bucket_or_max_seq_len_;
    tensor_para_size_          = other.tensor_para_size_;
    tensor_para_rank_          = other.tensor_para_rank_;
    pipeline_para_size_        = other.pipeline_para_size_;
    pipeline_para_rank_        = other.pipeline_para_rank_;
    agm_with_bias              = other.agm_with_bias;
    use_gated_activation       = other.use_gated_activation;
    position_embedding_type    = other.position_embedding_type;
    real_weights_num_          = other.real_weights_num_;
    ia3_num_tasks_             = other.ia3_num_tasks_;
    adapter_inter_size_        = other.adapter_inter_size_;
    shared_embed_              = other.shared_embed_;

    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    decoder_layer_weights.clear();
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(new AgmDecoderLayerWeight<T>(*other.decoder_layer_weights[l]));
    }
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " end");
    return *this;
}

template<typename T>
void AgmDecodingWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " end");
}

template<typename T>
void AgmDecodingWeight<T>::setWeightPtr()
{
    pre_decoder_embedding_table             = weights_ptr[0];
    absolute_or_relative_position_embedding = weights_ptr[1];
    post_decoder_layernorm.gamma            = weights_ptr[2];
    if (shared_embed_ && weights_ptr[1] != weights_ptr[3]) {
        deviceFree(weights_ptr[3]);
        weights_ptr[3]                = nullptr;
        post_decoder_embedding.kernel = weights_ptr[1];
    }
    else {
        post_decoder_embedding.kernel = weights_ptr[3];
    }
    if (agm_with_bias) {
        post_decoder_layernorm.beta = weights_ptr[4];
        post_decoder_embedding.bias = weights_ptr[5];
    }
}

template<typename T>
void AgmDecodingWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " start");
    FT_CHECK(is_maintain_buffer == true);
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "decoder");

    loadWeightFromBin<T>(weights_ptr[0], {(size_t)weights_size[0]}, dir_path + "/shared.weight.bin", model_file_type);
    // if (position_embedding_type == PositionEmbeddingType::absolute) {
    //     loadWeightFromBin<T>(weights_ptr[1], {(size_t)weights_size[1]}, dir_path + "/shared.ape.bin",
    //     model_file_type);
    // }
    // else {
    //     loadWeightFromBin<T>(weights_ptr[1],
    //                          {(size_t)weights_size[1]},
    //                          dir_path + "/decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight."
    //                              + std::to_string(tensor_para_rank_) + ".bin",
    //                          model_file_type);
    // }
    loadWeightFromBin<T>(
        weights_ptr[2], {(size_t)weights_size[2]}, dir_path + "/decoder.layernorm_output.weight.bin", model_file_type);
    if (checkIfFileExist(dir_path + "/lm_head.decoder.weight.bin")) {
        shared_embed_ = false;
        loadWeightFromBin<T>(
            weights_ptr[3], {(size_t)weights_size[3]}, dir_path + "/lm_head.decoder.weight.bin", model_file_type);
    }
    else {
        shared_embed_ = true;
        loadWeightFromBin<T>(
            weights_ptr[3], {(size_t)weights_size[3]}, dir_path + "/shared.weight.bin", model_file_type);
    }

    if (agm_with_bias) {
        loadWeightFromBin<T>(
            weights_ptr[4], {(size_t)weights_size[4]}, dir_path + "/decoder.layernorm_output.bias.bin", model_file_type);
        std::ifstream f(dir_path + "/final_logits_bias.bin");
        if (f.is_open()) {
            f.close();
            loadWeightFromBin<T>(
                weights_ptr[5], {(size_t)weights_size[5]}, dir_path + "/final_logits_bias.bin", model_file_type);  // may not exist
        }
        else {
            cudaMemset(weights_ptr[5], 0, sizeof(T) * (size_t)weights_size[5]);// set to zero, otherwise random value may exist
        }

    }

    for (int l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            decoder_layer_weights[l]->loadModel(dir_path + "/decoder.layers." + std::to_string(l) + ".",
                                                model_file_type);
        }
    }
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " end");
}

template<typename T>
bool AgmDecodingWeight<T>::isValidLayerParallelId(int l)
{
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " start");
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_rank_)
           && (l < local_num_layer * (pipeline_para_rank_ + 1));
}

template<typename T>
void AgmDecodingWeight<T>::resizeLayer(const int num_layer)
{
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " start");
    decoder_layer_weights.clear();
    num_layer_ = num_layer;
    decoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(new AgmDecoderLayerWeight<T>());
    }
    FT_LOG_DEBUG("AgmDecodingWeight " + std::string(__func__) + " end");
}

template<typename T>
void AgmDecodingWeight<T>::setAgmStructureDiff(bool                  agm_with_bias_para,
                                               bool                  use_gated_activation_para,
                                               PositionEmbeddingType position_embedding_type_para)
{
    agm_with_bias           = agm_with_bias_para;
    use_gated_activation    = use_gated_activation_para;
    position_embedding_type = position_embedding_type_para;
    for (int i = 0; i < num_layer_; i++) {
        decoder_layer_weights[i]->setAgmWithBias(agm_with_bias_para, use_gated_activation_para);
    }
}

template struct AgmDecodingWeight<float>;
template struct AgmDecodingWeight<half>;
#ifdef ENABLE_BF16
template struct AgmDecodingWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
