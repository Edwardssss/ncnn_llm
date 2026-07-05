#pragma once

#include <vector>
#include <mat.h>
#include <layer.h>

class MlaAttention : public ncnn::Layer
{
public:
    MlaAttention();

    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int load_model(const ncnn::ModelBin& mb);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const;

public:
    int hidden_size_;
    int num_heads_;
    int qk_nope_head_dim_;
    int qk_rope_head_dim_;
    int v_head_dim_;
    int kv_lora_rank_;
    int q_lora_rank_;
    int qk_head_dim_;
    int intermediate_size_;
    float scaling_;
    int layer_idx_;

    ncnn::Mat q_a_proj_weight_;
    ncnn::Mat q_a_layernorm_weight_;
    ncnn::Mat q_b_proj_weight_;
    ncnn::Mat kv_a_proj_weight_;
    ncnn::Mat kv_a_layernorm_weight_;
    ncnn::Mat kv_b_proj_weight_;
    ncnn::Mat o_proj_weight_;
    ncnn::Mat input_layernorm_weight_;
    ncnn::Mat post_attention_layernorm_weight_;
    ncnn::Mat gate_proj_weight_;
    ncnn::Mat up_proj_weight_;
    ncnn::Mat down_proj_weight_;
};

ncnn::Layer* MlaAttention_creator(void*);
void MlaAttention_destroyer(ncnn::Layer* layer, void*);
