#pragma once
// ===================================================================
// Youtu-VL 密集预测后处理
// ===================================================================
// 处理模型输出的特殊 token（坐标、分割 mask、深度等），将其转换为
// 可读格式或 RLE 编码的 ID 序列。
//
// 三种模式:
//   1. 坐标缩放（检测/定位） — 检测到 <x_N>/<y_N>
//   2. 密集解码（分割/深度） — 检测到 <ref> 或 <depth>
//   3. RLE 掩码输出          — 将 mask 编码为 <mask_rle> 格式
// ===================================================================

#include <vector>

#include <mat.h>

struct DensePredConfig {
    // Token IDs
    int coord_x0_id      = 278267;
    int coord_max        = 2047;
    int ref_begin        = 283371;
    int ref_end          = 283372;
    int ins_begin        = 283365;
    int mask_begin       = 27;
    int mask_end         = 713;
    int depth_begin      = 440;
    int comma_id         = 11;
    int digit_start_id   = 15;
    int mask_rle_id      = 7;
    int mask_rle_end_id  = 8;
    int others_id        = 283375;
    int image_token_id   = 128264;
    int custom1_begin      = 282363;  // <custom_1> begin, 1000 depth classes
    int custom1_count      = 1000;

    // Vision encoder params
    int patch_size         = 16;
    int spatial_merge_size = 2;
};

class YoutuDensePrediction
{
public:
    explicit YoutuDensePrediction(const DensePredConfig& cfg);

    // ------------------------------------------------------------------
    // 主入口：处理生成的 token，返回后处理后的 token 序列
    // ------------------------------------------------------------------
    // input_ids      : 输入序列 (含 <|image_pad|> 位置)
    // output_ids     : 模型生成的 token
    // dense_logits   : 首轮 forward logits [seq_len, vocab_size]
    // merger_h, merger_w : PatchMerger 输出网格尺寸
    // image_w, image_h   : 原始图像宽高
    // 返回: 后处理后的 token（坐标缩放 / 附加 mask token）
    std::vector<int> process(
        const std::vector<int>& input_ids,
        const std::vector<int>& output_ids,
        const ncnn::Mat& dense_logits,
        int merger_h, int merger_w,
        int image_w, int image_h) const;

private:
    // --- 坐标处理 ---
    bool has_coord_tokens(const std::vector<int>& output) const;
    std::vector<int> convert_coords(const std::vector<int>& output,
                                     float scale_x, float scale_y) const;

    // --- 序列匹配 ---
    bool contains_subseq(const std::vector<int>& seq,
                          const std::vector<int>& sub) const;
    bool contains_subseq_any(const std::vector<int>& seq,
                              const std::vector<std::vector<int>>& subs) const;

    // --- 密集解码 (分割 / 深度) ---
    std::vector<int> decode_dense(
        const std::vector<int>& input_ids,
        const std::vector<int>& output_ids,
        const float* logits, int vocab_size,
        int merger_h, int merger_w,
        int image_w, int image_h) const;

    // --- RLE 编码 ---
    std::vector<int> encode_rle(const std::vector<int>& flat_mask) const;
    std::vector<int> encode_int_as_digits(int value) const;

    DensePredConfig cfg_;
};
