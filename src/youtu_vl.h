#pragma once
// Youtu-VL model — inherits ncnn_llm_gpt, overrides prefill/generate for MLA KV cache

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

#include "ncnn_llm_gpt.h"
#include "utils/mla_attention.h"

// MLA KV cache: one kv_latent blob per layer
using YoutuKVCache = std::vector<ncnn::Mat>;

class YoutuMlaDecoder
{
public:
    bool load(FILE* fp, int num_layers, int hidden_size, int num_heads,
              int qk_nope_head_dim, int qk_rope_head_dim, int v_head_dim,
              int kv_lora_rank, int q_lora_rank, int intermediate_size);

    ncnn::Mat forward(const ncnn::Mat& hidden, const ncnn::Mat& mask,
                      const ncnn::Mat& cos_cache, const ncnn::Mat& sin_cache,
                      YoutuKVCache& kv_cache, bool is_prefill) const;

    int num_layers() const { return num_layers_; }

private:
    std::vector<MlaAttention> layers_;
    ncnn::Mat final_norm_weight_;
    int num_layers_ = 0;
    int hidden_size_ = 0;
};

class ncnn_llm_youtu : public ncnn_llm_gpt
{
public:
    ncnn_llm_youtu(const std::string& model_path, bool use_vulkan = false,
                   int num_threads = 0, int vulkan_device = 0);

    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text) const;
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(
        const std::string& input_text, const ncnn::Mat& bgr,
        const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx) const;
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(
        const std::string& input_text,
        const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx) const;

    std::shared_ptr<ncnn_llm_gpt_ctx> generate(
        const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in,
        const GenerateConfig& cfg,
        std::function<void(const std::string&)> callback) const;

    std::string run(const std::string& prompt, const std::string& image_path,
                    const GenerateConfig& cfg = GenerateConfig{});

    bool ok() const { return ok_; }

private:
    void load_youtu_config(const nlohmann::json& config);

    bool ok_ = false;

    YoutuMlaDecoder mla_decoder_;

    int image_token_id_        = 128264;
    int vision_start_token_id_ = 128262;
    int vision_end_token_id_   = 128263;
    int coord_x0_id_           = 278267;
    int coord_max_             = 2047;
    int ref_begin_id_          = 283371;
    int ref_end_id_            = 283372;
    int mask_begin_id_         = 27;
    int mask_end_id_           = 713;
    int depth_begin_id_        = 440;
    int comma_id_              = 11;
    int digit_start_id_        = 15;
    int mask_rle_id_           = 7;
    int mask_rle_end_id_       = 8;
    int others_token_id_       = 283375;
};
