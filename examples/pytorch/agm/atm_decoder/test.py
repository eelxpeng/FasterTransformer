import torch

from .configuration_atm_decoder import ATMT5Config
from .modeling_atm_decoder import ATMT5CE4ForConditionalGeneration

# python -m atm_decoder.test

def get_model_config(model_name):
    common_params = {'gradient_checkpointing': True,
                     'use_cache': False,
                     'tie_word_embeddings': True,
                     'use_megatron_softmax': False,
                     'use_activation_offloading': False,
                     'position_bias_per_layer': True,
                     'scale_attention': True,
                     'mup_scale': True,
                     'scale_init_for_embeddings': True,
                     'is_encoder_decoder': False,
                     'add_lm_head_bias': False}

    if model_name.lower() == "100b":
        params = {
            'relative_attention_num_buckets': 256,
            'relative_attention_max_distance': 450,
            'd_model': 10240,
            'decoder_ffn_dim': 40960,
            'decoder_layers': 76,
            'decoder_attention_heads': 80
        }
    elif model_name.lower() == "30b":
        params = {
            'relative_attention_num_buckets': 32,
            'relative_attention_max_distance': 128,
            'd_model': 7168,
            'decoder_ffn_dim': 28672,
            'decoder_layers': 48,
            'decoder_attention_heads': 56
        }
        params.update(common_params)
    elif model_name.lower() == "test":
        params = {
            'relative_attention_num_buckets': 32,
            'relative_attention_max_distance': 128,
            'd_model': 256,
            'decoder_ffn_dim': 1024,
            'decoder_layers': 12,
            'decoder_attention_heads': 4
        }
    else:
        raise ValueError(f"unknown model {model_name}")

    params.update(common_params)
    return params


def main():
    model_name = "test"
    cfg = ATMT5Config(**get_model_config(model_name))
    # model = ATMT5CE4ForConditionalGeneration(cfg)
    model = ATMT5CE4ForConditionalGeneration.from_pretrained('/tmp/alexa')

    max_output_len = 6
    # print(cfg)
    # "use_activation_offloading": false,
    # "use_alibi_position_embedding": false,
    # "use_cache": false,
    # "use_flash_attention": false,
    # "use_megatron_softmax": false,
    # "use_relative_attention_position_embedding": true,
    # 'position_bias_per_layer': True,
    # "scale_attention": true,
    # "scale_embedding": false,
    # "mup_scale": true,
    # "embed_dim": null
    # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         print(name, param.shape)
    #         if "relative_attention_bias" in name:
    #             print(param)

    print(model.generate(input_ids=torch.tensor([[2,0,12942]]), max_length=max_output_len, use_cache=True))


if __name__ == "__main__":
    main()
