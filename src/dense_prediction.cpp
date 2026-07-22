#include "dense_prediction.h"

#include <algorithm>
#include <cmath>
#include <vector>

static std::vector<float> bilinear_interpolate(
    const float* src, int channels,
    int src_h, int src_w,
    int dst_h, int dst_w)
{
    std::vector<float> dst(channels * dst_h * dst_w);
    float scale_y = (float)src_h / dst_h;
    float scale_x = (float)src_w / dst_w;

    for (int c = 0; c < channels; c++) {
        const float* src_ch = src + c * src_h * src_w;
        float*       dst_ch = dst.data() + c * dst_h * dst_w;

        for (int dst_y = 0; dst_y < dst_h; dst_y++) {
            float src_y = (dst_y + 0.5f) * scale_y - 0.5f;
            if (src_y < 0.f) src_y = 0.f;
            int y0 = (int)src_y;
            int y1 = std::min(y0 + 1, src_h - 1);
            float wy = src_y - (float)y0;

            for (int dst_x = 0; dst_x < dst_w; dst_x++) {
                float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
                if (src_x < 0.f) src_x = 0.f;
                int x0 = (int)src_x;
                int x1 = std::min(x0 + 1, src_w - 1);
                float wx = src_x - (float)x0;

                float v00 = src_ch[y0 * src_w + x0];
                float v10 = src_ch[y0 * src_w + x1];
                float v01 = src_ch[y1 * src_w + x0];
                float v11 = src_ch[y1 * src_w + x1];

                dst_ch[dst_y * dst_w + dst_x] =
                    (1.f - wx) * (1.f - wy) * v00 +
                    wx * (1.f - wy) * v10 +
                    (1.f - wx) * wy * v01 +
                    wx * wy * v11;
            }
        }
    }
    return dst;
}

YoutuDensePrediction::YoutuDensePrediction(const DensePredConfig& cfg)
    : cfg_(cfg) {}

std::vector<int> YoutuDensePrediction::process(
    const std::vector<int>& input_ids,
    const std::vector<int>& output_ids,
    const ncnn::Mat& dense_logits,
    int merger_h, int merger_w,
    int image_w, int image_h) const
{
    // 1. 坐标检测 / 缩放
    if (has_coord_tokens(output_ids) && merger_h > 0 && merger_w > 0
        && image_w > 0 && image_h > 0)
    {
        // HF: scale = raw_image / vision_input
        // vision_input = merger_grid * spatial_merge_size * patch_size
        float vision_input_w = (float)(merger_w * cfg_.spatial_merge_size * cfg_.patch_size);
        float vision_input_h = (float)(merger_h * cfg_.spatial_merge_size * cfg_.patch_size);
        return convert_coords(output_ids,
                              (float)image_w / vision_input_w,
                              (float)image_h / vision_input_h);
    }

    // 2. 密集解码 (分割 / 深度)
    //    HF 条件: (<ref> 且无 <ins>) 或 含 <depth>
    bool has_ref   = false;
    bool has_depth = false;
    bool has_ins   = false;
    for (int id : output_ids) {
        if (id == cfg_.ref_begin)   has_ref   = true;
        if (id == cfg_.depth_begin) has_depth = true;
        if (id == cfg_.ins_begin)   has_ins   = true;
    }
    if ((has_ref && !has_ins) || has_depth) {
        return decode_dense(input_ids, output_ids,
                            (const float*)dense_logits.data,
                            dense_logits.w,
                            merger_h, merger_w, image_w, image_h);
    }

    // 3. 无特殊 token — 原样返回
    return output_ids;
}

bool YoutuDensePrediction::has_coord_tokens(const std::vector<int>& output) const
{
    for (int id : output) {
        if (id >= cfg_.coord_x0_id &&
            id <= cfg_.coord_x0_id + cfg_.coord_max * 2 + 1)
            return true;
    }
    return false;
}

std::vector<int> YoutuDensePrediction::convert_coords(
    const std::vector<int>& output, float scale_x, float scale_y) const
{
    std::vector<int> out;
    out.reserve(output.size());

    for (int tid : output) {
        if (tid < cfg_.coord_x0_id ||
            tid > cfg_.coord_x0_id + cfg_.coord_max * 2 + 1)
        {
            out.push_back(tid);
            continue;
        }

        int offset = tid - cfg_.coord_x0_id;
        bool is_y  = (offset & 1) == 1;
        int  idx   = offset >> 1;

        if (idx < 0 || idx > cfg_.coord_max) {
            out.push_back(tid);
            continue;
        }

        int scaled;
        if (!is_y)
            scaled = (int)std::round(idx * scale_x);
        else
            scaled = (int)std::round(idx * scale_y);

        scaled = std::max(0, std::min(scaled, cfg_.coord_max));
        out.push_back(cfg_.coord_x0_id + (scaled << 1) + (is_y ? 1 : 0));
    }
    return out;
}

bool YoutuDensePrediction::contains_subseq(
    const std::vector<int>& seq, const std::vector<int>& sub) const
{
    if (sub.empty() || sub.size() > seq.size()) return false;
    for (size_t i = 0; i <= seq.size() - sub.size(); i++) {
        bool match = true;
        for (size_t j = 0; j < sub.size(); j++) {
            if (seq[i + j] != sub[j]) { match = false; break; }
        }
        if (match) return true;
    }
    return false;
}

bool YoutuDensePrediction::contains_subseq_any(
    const std::vector<int>& seq,
    const std::vector<std::vector<int>>& subs) const
{
    for (auto& sub : subs)
        if (contains_subseq(seq, sub)) return true;
    return false;
}

std::vector<int> YoutuDensePrediction::decode_dense(
    const std::vector<int>& input_ids,
    const std::vector<int>& output_ids,
    const float* logits, int vocab_size,
    int merger_h, int merger_w,
    int image_w, int image_h) const
{
    // 1. 定位所有 <|image_pad|> token
    std::vector<int> image_positions;
    for (size_t i = 0; i < input_ids.size(); i++)
        if (input_ids[i] == cfg_.image_token_id)
            image_positions.push_back((int)i);

    int num_image_tokens = (int)image_positions.size();
    if (num_image_tokens == 0) return output_ids;

    // 2. 判断路径: 深度估计 还是 分割
    bool is_depth = false;
    for (int id : output_ids)
        if (id == cfg_.depth_begin) { is_depth = true; break; }

    // ---------- 深度估计 ----------
    if (is_depth) {
        int depth_classes = cfg_.custom1_count;   // 1000

        // 提取 depth logits: [depth_classes, num_image_tokens] channel-first
        std::vector<float> depth_logits(depth_classes * num_image_tokens);
        for (int c = 0; c < depth_classes; c++) {
            int col = cfg_.custom1_begin + c;
            for (int t = 0; t < num_image_tokens; t++) {
                int pos = image_positions[t];
                depth_logits[c * num_image_tokens + t] =
                    logits[pos * vocab_size + col];
            }
        }

        // 上采样 2× 到视觉编码器网格尺寸
        int vision_h = merger_h * cfg_.spatial_merge_size;
        int vision_w = merger_w * cfg_.spatial_merge_size;
        auto upsampled = bilinear_interpolate(
            depth_logits.data(), depth_classes,
            merger_h, merger_w, vision_h, vision_w);

        // argmax → flat mask
        std::vector<int> depth_mask(vision_h * vision_w);
        for (int y = 0; y < vision_h; y++) {
            for (int x = 0; x < vision_w; x++) {
                int   best   = 0;
                float best_v = upsampled[y * vision_w + x];
                for (int c = 1; c < depth_classes; c++) {
                    float v = upsampled[c * vision_h * vision_w + y * vision_w + x];
                    if (v > best_v) { best_v = v; best = c; }
                }
                depth_mask[y * vision_w + x] = best;
            }
        }
        return encode_rle(depth_mask);
    }

    // ---------- 语义分割 ----------
    // 3. 提取所有 <ref> … </ref> 标签组
    std::vector<std::vector<int>> ref_spans;
    for (size_t i = 0; i < output_ids.size(); i++) {
        if (output_ids[i] != cfg_.ref_begin) continue;
        for (size_t j = i + 1; j < output_ids.size(); j++) {
            if (output_ids[j] == cfg_.ref_end) {
                if (j > i + 1)
                    ref_spans.emplace_back(
                        output_ids.begin() + i + 1,
                        output_ids.begin() + j);
                i = j;
                break;
            }
        }
    }
    if (ref_spans.empty()) return output_ids;

    int num_classes = (int)ref_spans.size();

    // 4. 检测 <OTHERS> 类别
    bool has_others     = false;
    int  others_idx     = -1;
    std::vector<int> others_token = {cfg_.others_id};
    for (int c = 0; c < num_classes; c++) {
        if (ref_spans[c] == others_token) {
            has_others = true;
            others_idx = c;
            break;
        }
    }

    // 5. 计算每类平均 logit: [num_classes, num_image_tokens] channel-first
    std::vector<float> class_logits(num_classes * num_image_tokens);
    for (int c = 0; c < num_classes; c++) {
        auto& label_ids = ref_spans[c];
        if (label_ids.empty()) continue;
        float scale = 1.f / (float)label_ids.size();
        for (int t = 0; t < num_image_tokens; t++) {
            int   pos = image_positions[t];
            const float* row = logits + pos * vocab_size;
            float total = 0.f;
            for (int lid : label_ids) total += row[lid];
            class_logits[c * num_image_tokens + t] = total * scale;
        }
    }

    // 6. 温度缩放 + 归一化
    if (has_others) {
        // Sigmoid 独立各类, OTHERS 固定 0.5
        for (int c = 0; c < num_classes; c++) {
            float* ptr = class_logits.data() + c * num_image_tokens;
            for (int t = 0; t < num_image_tokens; t++)
                ptr[t] = 1.f / (1.f + std::exp(-ptr[t]));
        }
        std::fill(class_logits.begin() + others_idx * num_image_tokens,
                  class_logits.begin() + (others_idx + 1) * num_image_tokens,
                  0.5f);
    } else {
        // temperature = 0.2, softmax across classes
        static const float kInvTemp = 1.f / 0.2f;   // = 5.0
        for (int t = 0; t < num_image_tokens; t++) {
            float max_val = -1e38f;
            for (int c = 0; c < num_classes; c++) {
                float v = class_logits[c * num_image_tokens + t];
                if (v > max_val) max_val = v;
            }
            float sum = 0.f;
            for (int c = 0; c < num_classes; c++) {
                float v = std::exp(
                    (class_logits[c * num_image_tokens + t] - max_val) * kInvTemp);
                class_logits[c * num_image_tokens + t] = v;
                sum += v;
            }
            float inv_sum = 1.f / sum;
            for (int c = 0; c < num_classes; c++)
                class_logits[c * num_image_tokens + t] *= inv_sum;
        }
    }

    // 7. 双线性缩放到原始图像尺寸 → argmax → flat mask
    auto probs = bilinear_interpolate(
        class_logits.data(), num_classes,
        merger_h, merger_w, image_h, image_w);

    std::vector<int> seg_mask(image_h * image_w);
    for (int y = 0; y < image_h; y++) {
        for (int x = 0; x < image_w; x++) {
            int   best   = 0;
            float best_v = probs[y * image_w + x];
            for (int c = 1; c < num_classes; c++) {
                float v = probs[c * image_h * image_w + y * image_w + x];
                if (v > best_v) { best_v = v; best = c; }
            }
            seg_mask[y * image_w + x] = best;
        }
    }

    return encode_rle(seg_mask);
}

std::vector<int> YoutuDensePrediction::encode_int_as_digits(int value) const
{
    if (value == 0) return {cfg_.digit_start_id};

    std::vector<int> digits;
    while (value > 0) {
        digits.push_back(cfg_.digit_start_id + (value % 10));
        value /= 10;
    }
    std::reverse(digits.begin(), digits.end());
    return digits;
}

std::vector<int> YoutuDensePrediction::encode_rle(
    const std::vector<int>& flat_mask) const
{
    if (flat_mask.empty()) return {};

    // 计算 run-lengths
    struct Run { int value; int count; };
    std::vector<Run> runs;
    int current_value = flat_mask[0];
    int current_run   = 1;
    for (size_t i = 1; i < flat_mask.size(); i++) {
        if (flat_mask[i] == current_value) {
            current_run++;
        } else {
            runs.push_back({current_value, current_run});
            current_value = flat_mask[i];
            current_run   = 1;
        }
    }
    runs.push_back({current_value, current_run});

    // 构建 body: <mask_rle> v , c </mask_rle> [ , … ]
    std::vector<int> body;
    for (size_t i = 0; i < runs.size(); i++) {
        body.push_back(cfg_.mask_rle_id);

        auto value_digits = encode_int_as_digits(runs[i].value);
        body.insert(body.end(), value_digits.begin(), value_digits.end());
        body.push_back(cfg_.comma_id);

        auto count_digits = encode_int_as_digits(runs[i].count);
        body.insert(body.end(), count_digits.begin(), count_digits.end());
        body.push_back(cfg_.mask_rle_end_id);

        if (i != runs.size() - 1)
            body.push_back(cfg_.comma_id);
    }

    // 外层包装: <mask> body </mask>
    std::vector<int> result;
    result.push_back(cfg_.mask_begin);
    result.insert(result.end(), body.begin(), body.end());
    result.push_back(cfg_.mask_end);
    return result;
}
