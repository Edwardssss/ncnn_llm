#pragma once

#include <array>
#include <cassert>
#include <cstdio>
#include <exception>
#include <functional>
#include <locale>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <unordered_set>
#include <algorithm>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

#include "ncnn_llm_base.h"
#include "ncnn_llm_gpt.h"
#include "ncnn_text_runtime.h"
#include "utils/tokenizer/bpe_tokenizer.h"
#include "utils/rope_embed.h"
#include "utils/image_utils.h"

using nlohmann::json;

class ncnn_llm_ocr : public ncnn_llm_base {
public:
    ncnn_llm_ocr(const std::string& model_path, bool use_vulkan = false, int num_threads = 0);

    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& prompt_text, const ncnn::Mat& bgr_image);
    std::shared_ptr<ncnn_llm_gpt_ctx> generate(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx,
                                               const GenerateConfig& cfg,
                                               std::function<void(const std::string&)> callback);

    bool ok() const { return ok_; }
    const std::string& model_type() const { return model_type_; }

private:
    std::shared_ptr<ncnn::Net> vision_net_;
    std::shared_ptr<ncnn::Net> text_embed_net_;
    std::shared_ptr<ncnn::Net> text_decoder_net_;
    std::shared_ptr<ncnn::Net> lm_head_net_;
    std::shared_ptr<BpeTokenizer> bpe_;
    std::unordered_set<int> additional_special_id_set_;

    std::string model_type_ = "glm_ocr";

    // --- HunyuanOCR-specific settings (model_type_ == "hunyuan_ocr") ---
    bool use_kv_cache_ = true;
    float rope_alpha_ = 1000.0f;
    std::vector<int> xdrope_section_;
    int bos_id_ = -1;
    int system_end_id_ = -1;
    int user_end_id_ = -1;
    int image_start_id_ = -1;
    int image_end_id_ = -1;
    int special_id_begin_ = 0x7fffffff;
    std::unordered_set<int> eos_ids_;

    int attn_cnt_ = 16;
    int hidden_size_ = 1536;
    int head_dim_ = 128;
    float rope_theta_ = 10000.0f;
    std::vector<int> mrope_section_;
    int image_token_id_ = 59280;
    int patch_size_ = 14;
    int spatial_merge_size_ = 2;
    int vision_hidden_size_ = 1024;
    int vision_head_dim_ = 64;
    int vision_num_heads_ = 16;
    float vision_rope_theta_ = 0.0f;
    int vision_rope_dim_ = 0;
    std::vector<int> vision_mrope_section_;
    int max_num_patches_ = 3432;
    long long min_pixels_ = 12544;
    long long max_pixels_ = 9633792;
    float image_mean_[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std_[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    int eos_ = -1;
    int eop_ = -1;
    int vocab_size_ = 59392;

    void get_image_size_for_patches(int img_h, int img_w, int& target_h, int& target_w) const;
    ncnn::Mat bgr_to_image_strip(const ncnn::Mat& bgr, int& num_patches_h, int& num_patches_w) const;

    ncnn::Mat run_vision(const ncnn::Mat& image_strip, const ncnn::Mat& cos_cache, const ncnn::Mat& sin_cache) const;

    void generate_text_rope_cache(int seq_len, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache) const;

    // --- HunyuanOCR path ---
    void smart_resize_hunyuan(int img_h, int img_w, int& target_h, int& target_w) const;
    ncnn::Mat bgr_to_chw_image_hunyuan(const ncnn::Mat& bgr, int& target_h, int& target_w) const;
    ncnn::Mat run_vision_hunyuan(const ncnn::Mat& image_chw) const;
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill_hunyuan(const std::string& prompt_text, const ncnn::Mat& bgr_image);
    std::shared_ptr<ncnn_llm_gpt_ctx> generate_hunyuan(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx,
                                                       const GenerateConfig& cfg,
                                                       std::function<void(const std::string&)> callback);
};
