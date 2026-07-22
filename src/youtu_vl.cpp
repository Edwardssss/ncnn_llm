#include "youtu_vl.h"
#include "ncnn_text_runtime.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

bool YoutuMlaDecoder::load(FILE* fp, int num_layers, int hidden_size, int num_heads,
                            int qk_nope_head_dim, int qk_rope_head_dim, int v_head_dim,
                            int kv_lora_rank, int q_lora_rank, int intermediate_size)
{
    num_layers_ = num_layers;
    hidden_size_ = hidden_size;

    int qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    int lf = 0;
    lf += hidden_size * q_lora_rank;
    lf += q_lora_rank;
    lf += q_lora_rank * num_heads * qk_head_dim;
    lf += hidden_size * (kv_lora_rank + qk_rope_head_dim);
    lf += kv_lora_rank;
    lf += kv_lora_rank * num_heads * (qk_nope_head_dim + v_head_dim);
    lf += (num_heads * v_head_dim) * hidden_size;
    lf += hidden_size;
    lf += hidden_size;
    lf += hidden_size * intermediate_size;
    lf += hidden_size * intermediate_size;
    lf += intermediate_size * hidden_size;

    layers_.resize(num_layers);

    for (int li = 0; li < num_layers; li++) {
        std::vector<float> buf(lf);
        if (fread(buf.data(), 4, lf, fp) != (size_t)lf) {
            fprintf(stderr, "[YoutuMlaDecoder] failed to read layer %d\n", li);
            return false;
        }

        MlaAttention& mla = layers_[li];
        ncnn::ParamDict pd;
        pd.set(0, hidden_size); pd.set(1, num_heads);
        pd.set(2, qk_nope_head_dim); pd.set(3, qk_rope_head_dim);
        pd.set(4, v_head_dim); pd.set(5, kv_lora_rank);
        pd.set(6, q_lora_rank); pd.set(7, intermediate_size);
        pd.set(8, li);
        mla.load_param(pd);

        size_t off = 0;
        auto mk = [&](int w, int h) {
            ncnn::Mat m(w, h);
            memcpy(m.data, &buf[off], (size_t)w * h * 4);
            off += (size_t)w * h;
            return m;
        };

        mla.q_a_proj_weight_                = mk(hidden_size, q_lora_rank);
        mla.q_a_layernorm_weight_           = mk(q_lora_rank, 1);
        mla.q_b_proj_weight_                = mk(q_lora_rank, num_heads * qk_head_dim);
        mla.kv_a_proj_weight_               = mk(hidden_size, kv_lora_rank + qk_rope_head_dim);
        mla.kv_a_layernorm_weight_          = mk(kv_lora_rank, 1);
        mla.kv_b_proj_weight_               = mk(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim));
        mla.o_proj_weight_                  = mk(num_heads * v_head_dim, hidden_size);
        mla.input_layernorm_weight_         = mk(hidden_size, 1);
        mla.post_attention_layernorm_weight_= mk(hidden_size, 1);
        mla.gate_proj_weight_               = mk(hidden_size, intermediate_size);
        mla.up_proj_weight_                 = mk(hidden_size, intermediate_size);
        mla.down_proj_weight_               = mk(intermediate_size, hidden_size);
    }

    std::vector<float> fn_buf(hidden_size);
    if (fread(fn_buf.data(), 4, hidden_size, fp) != (size_t)hidden_size) {
        fprintf(stderr, "[YoutuMlaDecoder] failed to read final_norm\n");
        return false;
    }
    final_norm_weight_ = ncnn::Mat(hidden_size, 1);
    memcpy(final_norm_weight_.data, fn_buf.data(), hidden_size * 4);

    fprintf(stderr, "[YoutuMlaDecoder] loaded %d layers (%.1f GB)\n",
            num_layers, (double)((int64_t)lf * num_layers + hidden_size) * 4 / 1e9);
    return true;
}

static void rms_norm(ncnn::Mat& x, const ncnn::Mat& weight, float eps = 1e-6f) {
    int n = x.h, dim = x.w;
    float* data = (float*)x.data;
    const float* w = (const float*)weight.data;
    for (int i = 0; i < n; i++) {
        float* row = data + i * dim;
        float ss = 1e-10f;
        for (int j = 0; j < dim; j++) ss += row[j] * row[j];
        float inv = 1.0f / sqrtf(ss / (float)dim);
        for (int j = 0; j < dim; j++) row[j] = row[j] * inv * w[j];
    }
}

ncnn::Mat YoutuMlaDecoder::forward(const ncnn::Mat& hidden, const ncnn::Mat& mask,
                                     const ncnn::Mat& cos_cache, const ncnn::Mat& sin_cache,
                                     YoutuKVCache& kv_cache, bool is_prefill) const
{
    ncnn::Mat h = hidden;
    ncnn::Option opt;
    opt.num_threads = 4;

    for (int i = 0; i < num_layers_; i++) {
        ncnn::Mat kv_in;
        if (!is_prefill && i < (int)kv_cache.size())
            kv_in = kv_cache[i];

        std::vector<ncnn::Mat> bottoms = {h, mask, cos_cache, sin_cache, kv_in};
        std::vector<ncnn::Mat> tops(2);
        layers_[i].forward(bottoms, tops, opt);

        h = tops[0];
        if (is_prefill)
            kv_cache.push_back(tops[1]);
        else
            kv_cache[i] = tops[1];
    }

    rms_norm(h, final_norm_weight_);
    return h;
}

ncnn_llm_youtu::ncnn_llm_youtu(const std::string& model_path, bool use_vulkan,
                                 int num_threads, int vulkan_device)
    : ncnn_llm_gpt(model_path, use_vulkan, num_threads, vulkan_device)
{
    if (use_vulkan)
        printf("[ncnn_llm_youtu] Vulkan enabled\n");
    else
        printf("[ncnn_llm_youtu] Vulkan disabled, using CPU only\n");

    nlohmann::json config;
    {
        std::ifstream ifs(model_path + "/model.json");
        if (!ifs.is_open())
            throw std::runtime_error("Cannot open model.json");
        ifs >> config;
    }
    load_youtu_config(config);

    std::string dec_bin = model_path + "/youtu_decoder.ncnn.bin";
    FILE* fp = fopen(dec_bin.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "[youtu-vl] ERROR: cannot open %s\n", dec_bin.c_str());
        throw std::runtime_error("Cannot open " + dec_bin);
    }
    if (!mla_decoder_.load(fp, attn_cnt, hidden_size, 32,
                            qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                            kv_lora_rank, q_lora_rank, mlp_intermediate_size)) {
        fclose(fp);
        throw std::runtime_error("YoutuMlaDecoder load failed");
    }
    fclose(fp);
    fprintf(stderr, "[youtu-vl] ready. layers=%d hidden=%d vocab=%d\n",
            attn_cnt, hidden_size, (int)bpe->vocab_size());

    ok_ = true;
}

void ncnn_llm_youtu::load_youtu_config(const nlohmann::json& config)
{
    auto& s = config["setting"];
    if (s.contains("image_token_id"))
        image_token_id_ = s["image_token_id"].get<int>();
    if (s.contains("vision_start_token_id"))
        vision_start_token_id_ = s["vision_start_token_id"].get<int>();
    if (s.contains("vision_end_token_id"))
        vision_end_token_id_ = s["vision_end_token_id"].get<int>();
    if (s.contains("custom_tokens")) {
        auto& ct = s["custom_tokens"];
        if (ct.contains("coord_begin_id"))
            coord_x0_id_ = ct["coord_begin_id"].get<int>();
        if (ct.contains("ref_begin_id"))
            ref_begin_id_ = ct["ref_begin_id"].get<int>();
        if (ct.contains("ref_end_id"))
            ref_end_id_ = ct["ref_end_id"].get<int>();
        if (ct.contains("depth_begin_id"))
            depth_begin_id_ = ct["depth_begin_id"].get<int>();
        if (ct.contains("others_id"))
            others_token_id_ = ct["others_id"].get<int>();
    }
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_youtu::prefill(
    const std::string& input_text) const
{
    auto token_ids = bpe->encode(input_text, false, false);

    if (token_ids.empty()) return std::make_shared<ncnn_llm_gpt_base_ctx>();

    int last_id = token_ids.back();
    token_ids.pop_back();

    int seq_len = (int)token_ids.size();

    ncnn::Mat token_embed = llm_run_text_embed(*embed_net, token_ids);

    ncnn::Mat cos_cache, sin_cache;
    generate_rope_embed_cache(seq_len, rope_head_dim, 0,
                               cos_cache, sin_cache, rope_theta);

    ncnn::Mat mask(seq_len, seq_len);
    mask.fill(0.0f);
    for (int i = 0; i < seq_len; i++) {
        float* row = mask.row(i);
        for (int j = i + 1; j < seq_len; j++) row[j] = -1e38f;
    }

    YoutuKVCache kv_cache;
    ncnn::Mat decode_out = mla_decoder_.forward(token_embed, mask,
                                                  cos_cache, sin_cache,
                                                  kv_cache, true);

    ncnn::Mat last_embed = llm_run_text_embed(*embed_net, last_id);

    int total_kv = seq_len + 1;
    ncnn::Mat last_cos, last_sin;
    generate_rope_embed_cache(total_kv, rope_head_dim, 0,
                               last_cos, last_sin, rope_theta);

    ncnn::Mat last_mask(total_kv, 1);
    last_mask.fill(0.0f);

    decode_out = mla_decoder_.forward(last_embed, last_mask,
                                       last_cos, last_sin,
                                       kv_cache, false);

    ncnn::Mat logits = llm_run_lm_head(*proj_out_net, decode_out, model_path_);

    int next_token = 0;
    {
        const float* p = (const float*)logits.data;
        float mv = p[0];
        for (int i = 1; i < logits.w; i++)
            if (p[i] > mv) { mv = p[i]; next_token = i; }
    }

    auto ctx = std::make_shared<ncnn_llm_gpt_base_ctx>();
    ctx->kv_cache.resize(kv_cache.size());
    for (size_t i = 0; i < kv_cache.size(); i++)
        ctx->kv_cache[i] = std::make_pair(kv_cache[i], ncnn::Mat());
    ctx->cur_token = next_token;
    ctx->position_id = total_kv;
    return ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_youtu::prefill(
    const std::string& input_text, const ncnn::Mat& bgr,
    const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx) const
{
    if (vision_type != Vision_Type::VISION_YOUTU_VL)
        return ncnn_llm_gpt::prefill(input_text, bgr, ctx);

    ncnn::Mat image_embeds;
    int num_patches_w = 0, num_patches_h = 0;
    if (get_visiual_features_youtu_vl(bgr, image_embeds, num_patches_w, num_patches_h) != 0) {
        fprintf(stderr, "[youtu-vl] vision failed\n");
        return ctx;
    }
    int num_visual = image_embeds.h;

    auto token_ids = bpe->encode(input_text, false, false);
    std::vector<int> ids;
    ids.push_back(vision_start_token_id_);
    for (int i = 0; i < num_visual; i++)
        ids.push_back(image_token_id_);
    ids.push_back(vision_end_token_id_);
    ids.insert(ids.end(), token_ids.begin(), token_ids.end());

    int last_id = ids.back();
    ids.pop_back();

    ncnn::Mat token_embed = llm_run_text_embed(*embed_net, ids);

    int pad_idx = -1;
    for (int i = 0; i < (int)ids.size(); i++) {
        if (ids[i] == image_token_id_) { pad_idx = i; break; }
    }
    if (pad_idx >= 0) {
        for (int i = 0; i < num_visual; i++) {
            float* dst = token_embed.row(pad_idx + i);
            const float* src = image_embeds.row(i);
            memcpy(dst, src, token_embed.w * sizeof(float));
        }
    }

    int seq_len = (int)ids.size();
    ncnn::Mat cos_cache, sin_cache;
    generate_rope_embed_cache(seq_len, rope_head_dim, 0,
                               cos_cache, sin_cache, rope_theta);

    ncnn::Mat mask(seq_len, seq_len);
    mask.fill(0.0f);
    for (int i = 0; i < seq_len; i++) {
        float* row = mask.row(i);
        for (int j = i + 1; j < seq_len; j++) row[j] = -1e38f;
    }

    YoutuKVCache kv_cache;
    ncnn::Mat decode_out = mla_decoder_.forward(token_embed, mask,
                                                  cos_cache, sin_cache,
                                                  kv_cache, true);

    ncnn::Mat last_embed = llm_run_text_embed(*embed_net, last_id);

    int total_kv = seq_len + 1;
    ncnn::Mat last_cos, last_sin;
    generate_rope_embed_cache(total_kv, rope_head_dim, 0,
                               last_cos, last_sin, rope_theta);

    ncnn::Mat last_mask(total_kv, 1);
    last_mask.fill(0.0f);

    decode_out = mla_decoder_.forward(last_embed, last_mask,
                                       last_cos, last_sin,
                                       kv_cache, false);

    ncnn::Mat logits = llm_run_lm_head(*proj_out_net, decode_out, model_path_);

    int next_token = 0;
    {
        const float* p = (const float*)logits.data;
        float mv = p[0];
        for (int i = 1; i < logits.w; i++) {
            if (p[i] > mv) { mv = p[i]; next_token = i; }
        }
    }

    auto out_ctx = std::make_shared<ncnn_llm_gpt_base_ctx>();
    out_ctx->kv_cache.resize(kv_cache.size());
    for (size_t i = 0; i < kv_cache.size(); i++)
        out_ctx->kv_cache[i] = std::make_pair(kv_cache[i], ncnn::Mat());
    out_ctx->cur_token = next_token;
    out_ctx->position_id = seq_len + 1;
    return out_ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_youtu::prefill(
    const std::string& input_text,
    const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx) const
{
    return ncnn_llm_gpt::prefill(input_text, ctx);
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_youtu::generate(
    const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in,
    const GenerateConfig& cfg,
    std::function<void(const std::string&)> callback) const
{
    std::vector<int> dummy_ids;
    return generate_with_ids(ctx_in, cfg, callback, dummy_ids);
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_youtu::generate_with_ids(
    const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in,
    const GenerateConfig& cfg,
    std::function<void(const std::string&)> callback,
    std::vector<int>& output_ids) const
{
    const int vocab_sz = (int)bpe->vocab_size();
    auto ctx = ctx_in->clone();
    std::unordered_set<int> history;
    history.insert(ctx->cur_token);

    YoutuKVCache kv_cache;
    kv_cache.resize(ctx->kv_cache.size());
    for (size_t i = 0; i < ctx->kv_cache.size(); i++)
        kv_cache[i] = ctx->kv_cache[i].first;

    for (int step = 0; step < cfg.max_new_tokens; step++) {
        if (ctx->cur_token == eos) break;

        output_ids.push_back(ctx->cur_token);

        std::vector<int> tok = {ctx->cur_token};
        callback(bpe->decode(tok, true));

        ncnn::Mat cur_embed = llm_run_text_embed(*embed_net, ctx->cur_token);

        int total_len = kv_cache.empty() ? 1 : kv_cache[0].h + 1;
        ncnn::Mat cos_cache, sin_cache;
        generate_rope_embed_cache(total_len, rope_head_dim, 0,
                                   cos_cache, sin_cache, rope_theta);
        ctx->position_id++;

        ncnn::Mat mask_v(total_len, 1);
        mask_v.fill(0.0f);

        ncnn::Mat decode_out = mla_decoder_.forward(cur_embed, mask_v,
                                                     cos_cache, sin_cache,
                                                     kv_cache, false);

        ncnn::Mat logits = llm_run_lm_head(*proj_out_net, decode_out, model_path_);

        LlmTokenSampleConfig scfg;
        scfg.vocab_size = vocab_sz;
        scfg.temperature = cfg.temperature;
        scfg.top_p = cfg.top_p;
        scfg.top_k = cfg.top_k;
        scfg.repetition_penalty = cfg.repetition_penalty;
        scfg.do_sample = cfg.do_sample;
        int next_id = llm_select_next_token(logits, history, scfg);

        ctx->cur_token = next_id;
        history.insert(next_id);
    }

    ctx->kv_cache.resize(kv_cache.size());
    for (size_t i = 0; i < kv_cache.size(); i++)
        ctx->kv_cache[i] = std::make_pair(kv_cache[i], ncnn::Mat());

    return ctx;
}

std::string ncnn_llm_youtu::run(const std::string& prompt,
                                  const std::string& image_path,
                                  const GenerateConfig& cfg)
{
    ncnn::Mat image = load_image_to_ncnn_mat(image_path);
    if (ncnn_mat_empty(image)) {
        fprintf(stderr, "[youtu-vl] cannot load image: %s\n", image_path.c_str());
        return "";
    }

    // --- 1. Run vision encoder ---
    ncnn::Mat image_embeds;
    int num_patches_w = 0, num_patches_h = 0;
    if (get_visiual_features_youtu_vl(image, image_embeds, num_patches_w, num_patches_h) != 0) {
        fprintf(stderr, "[youtu-vl] vision failed\n");
        return "";
    }
    int num_visual = image_embeds.h;

    // --- 2. Build full template matching HF chat_template ---
    // Format: <|begin_of_text|>system\n...<|end_of_text|>\n<|begin_of_text|>user\n<|vision_start|><|image_pad|>...<|vision_end|>prompt<|end_of_text|>\n<|begin_of_text|>assistant\n
    std::string system_part = "<|begin_of_text|>system\nYou are a helpful assistant.<|end_of_text|>\n<|begin_of_text|>user\n";

    // Encode system+user header as text
    auto sys_ids = bpe->encode(system_part, false, false);

    std::string user_part = prompt + "<|end_of_text|>\n<|begin_of_text|>assistant\n";
    auto user_ids = bpe->encode(user_part, false, false);

    // Combine all token IDs: system_ids + vision tokens + user_ids
    std::vector<int> ids;
    ids.insert(ids.end(), sys_ids.begin(), sys_ids.end());
    ids.push_back(vision_start_token_id_);
    for (int i = 0; i < num_visual; i++)
        ids.push_back(image_token_id_);
    ids.push_back(vision_end_token_id_);
    ids.insert(ids.end(), user_ids.begin(), user_ids.end());

    // --- 3. Prefill all tokens (split last token for next-token prediction) ---
    int last_id = ids.back();
    ids.pop_back();

    ncnn::Mat token_embed = llm_run_text_embed(*embed_net, ids);

    // Replace <|image_pad|> positions with vision embeddings
    int pad_idx = -1;
    for (int i = 0; i < (int)ids.size(); i++) {
        if (ids[i] == image_token_id_) { pad_idx = i; break; }
    }
    if (pad_idx >= 0) {
        for (int i = 0; i < num_visual; i++) {
            float* dst = token_embed.row(pad_idx + i);
            const float* src = image_embeds.row(i);
            memcpy(dst, src, token_embed.w * sizeof(float));
        }
    }

    int seq_len = (int)ids.size();
    ncnn::Mat cos_cache, sin_cache;
    generate_rope_embed_cache(seq_len, rope_head_dim, 0,
                               cos_cache, sin_cache, rope_theta);

    ncnn::Mat mask(seq_len, seq_len);
    mask.fill(0.0f);
    for (int i = 0; i < seq_len; i++) {
        float* row = mask.row(i);
        for (int j = i + 1; j < seq_len; j++) row[j] = -1e38f;
    }

    YoutuKVCache kv_cache;
    ncnn::Mat decode_out = mla_decoder_.forward(token_embed, mask,
                                                  cos_cache, sin_cache,
                                                  kv_cache, true);

    // Process the last token
    ncnn::Mat last_embed = llm_run_text_embed(*embed_net, last_id);

    int total_kv = seq_len + 1;
    ncnn::Mat last_cos, last_sin;
    generate_rope_embed_cache(total_kv, rope_head_dim, 0,
                               last_cos, last_sin, rope_theta);

    ncnn::Mat last_mask(total_kv, 1);
    last_mask.fill(0.0f);

    decode_out = mla_decoder_.forward(last_embed, last_mask,
                                       last_cos, last_sin,
                                       kv_cache, false);

    ncnn::Mat logits = llm_run_lm_head(*proj_out_net, decode_out, model_path_);

    int next_token = 0;
    {
        const float* p = (const float*)logits.data;
        float mv = p[0];
        for (int i = 1; i < logits.w; i++)
            if (p[i] > mv) { mv = p[i]; next_token = i; }
    }

    auto ctx = std::make_shared<ncnn_llm_gpt_base_ctx>();
    ctx->kv_cache.resize(kv_cache.size());
    for (size_t i = 0; i < kv_cache.size(); i++)
        ctx->kv_cache[i] = std::make_pair(kv_cache[i], ncnn::Mat());
    ctx->cur_token = next_token;
    ctx->position_id = total_kv;

    // --- 4. Generate with token ID collection for DensePrediction post-processing ---
    std::vector<int> generated_ids;
    // Greedy decoding — avoids MLA precision drift from top_p/temperature
    GenerateConfig greedy_cfg = cfg;
    greedy_cfg.do_sample = 0;

    std::string output;
    auto cb = [&output](const std::string& token) { output += token; };
    generate_with_ids(ctx, greedy_cfg, cb, generated_ids);

    // --- 5. Apply coordinate conversion (DensePrediction) ---
    std::string postprocessed = postprocess_output(ids, generated_ids,
                                                    num_patches_w, num_patches_h,
                                                    image.w, image.h);
    if (!output.empty()) { /* keep text */ }
    else if (!postprocessed.empty())
        output = postprocessed;

    return output;
}

std::string ncnn_llm_youtu::postprocess_output(
    const std::vector<int>& input_ids,
    const std::vector<int>& output_ids,
    int merger_w, int merger_h,
    int image_w, int image_h) const
{
    // Build scale factors matching HF DensePrediction
    // vision_input = merger_grid * spatial_merge_size * patch_size
    // spatial_merge_size = 2, patch_size = 16
    float vision_input_w = (float)(merger_w * 2 * 16);
    float vision_input_h = (float)(merger_h * 2 * 16);
    float scale_x = (float)image_w / vision_input_w;
    float scale_y = (float)image_h / vision_input_h;

    std::string result;
    for (int tid : output_ids) {
        if (tid >= coord_x0_id_ && tid <= coord_x0_id_ + coord_max_ * 2 + 1) {
            // Coordinate token: convert to scaled <x_N>/<y_N>
            int offset = tid - coord_x0_id_;
            bool is_y = (offset & 1) == 1;
            int idx = offset >> 1;
            if (idx < 0 || idx > coord_max_) {
                result += bpe->decode({tid}, true);
                continue;
            }
            int scaled = (int)std::round(idx * (is_y ? scale_y : scale_x));
            scaled = std::max(0, std::min(scaled, coord_max_));
            int new_tid = coord_x0_id_ + (scaled << 1) + (is_y ? 1 : 0);
            result += bpe->decode({new_tid}, true);
        } else {
            result += bpe->decode({tid}, true);
        }
    }
    return result;
}

std::string ncnn_llm_youtu::run_text(const std::string& prompt,
                                       const GenerateConfig& cfg)
{
    // Match HF chat_template: always prepend system prompt (same as HF behavior)
    std::string template_text = "<|begin_of_text|>system\nYou are a helpful assistant.<|end_of_text|>\n<|begin_of_text|>user\n" + prompt + "<|end_of_text|>\n<|begin_of_text|>assistant\n";

    auto ctx = prefill(template_text);

    // Use greedy for text-only (matching prefill behavior, avoiding
    // MLA precision drift through top_p/temperature sampling).
    GenerateConfig text_cfg = cfg;
    text_cfg.do_sample = 0;

    std::string output;
    auto cb = [&output](const std::string& token) { output += token; };
    generate(ctx, text_cfg, cb);
    return output;
}
