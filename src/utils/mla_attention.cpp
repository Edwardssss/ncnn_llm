#include "mla_attention.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

static void rms_norm_inplace(float* x, const float* weight, int n, int dim, float eps = 1e-6f)
{
    #pragma omp parallel for if(n > 4)
    for (int i = 0; i < n; i++)
    {
        float* row = x + i * dim;
        float sum_sq = 1e-10f;
        for (int j = 0; j < dim; j++)
            sum_sq += row[j] * row[j];

        float inv_norm = 1.0f / sqrtf(sum_sq / (float)dim);
        for (int j = 0; j < dim; j++)
            row[j] = row[j] * inv_norm * weight[j];
    }
}

static void matmul_nt(const float* A, const float* B, float* C, int M, int K, int N)
{
    #pragma omp parallel for if(M > 4)
    for (int i = 0; i < M; i++)
    {
        const float* Ai = A + i * K;
        float* Ci = C + i * N;
        for (int j = 0; j < N; j++)
        {
            const float* Bj = B + j * K;
            float sum = 0.f;
            for (int k = 0; k < K; k++)
                sum += Ai[k] * Bj[k];
            Ci[j] = sum;
        }
    }
}

static void matmul_nn(const float* A, const float* B, float* C, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        const float* Ai = A + i * K;
        float* Ci = C + i * N;
        for (int j = 0; j < N; j++)
        {
            float sum = 0.f;
            for (int k = 0; k < K; k++)
                sum += Ai[k] * B[k * N + j];
            Ci[j] = sum;
        }
    }
}

static void softmax_last_dim(float* x, int rows, int cols)
{
    #pragma omp parallel for if(rows > 4)
    for (int i = 0; i < rows; i++)
    {
        float* row = x + i * cols;
        float max_val = row[0];
        for (int j = 1; j < cols; j++)
            if (row[j] > max_val) max_val = row[j];

        float sum = 0.f;
        for (int j = 0; j < cols; j++)
        {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        float inv = 1.f / sum;
        for (int j = 0; j < cols; j++)
            row[j] *= inv;
    }
}

static void apply_interleave_rope(float* q_or_k, int num_heads, int this_seq,
                                   int qk_head_dim, int nope_dim, int rope_dim,
                                   const float* cos, const float* sin, int cos_sin_dim,
                                   int start_pos = 0)
{
    int half_rope = rope_dim / 2;
    for (int h = 0; h < num_heads; h++)
    {
        for (int s = 0; s < this_seq; s++)
        {
            float* x = q_or_k
                + h * this_seq * qk_head_dim
                + s * qk_head_dim
                + nope_dim;

            std::vector<float> reorder(rope_dim);
            for (int i = 0; i < half_rope; i++)
            {
                reorder[i]            = x[i * 2];
                reorder[i + half_rope] = x[i * 2 + 1];
            }
            memcpy(x, reorder.data(), rope_dim * sizeof(float));

            for (int i = 0; i < half_rope; i++)
            {
                float a = x[i];
                float b = x[i + half_rope];
                float c = cos[(start_pos + s) * cos_sin_dim + i];
                float si = sin[(start_pos + s) * cos_sin_dim + i];
                x[i]             = a * c - b * si;
                x[i + half_rope] = b * c + a * si;
            }

            for (int i = 0; i < half_rope; i++)
            {
                reorder[i * 2]     = x[i];
                reorder[i * 2 + 1] = x[i + half_rope];
            }
            memcpy(x, reorder.data(), rope_dim * sizeof(float));
        }
    }
}

MlaAttention::MlaAttention()
{
    one_blob_only = false;
    support_inplace = false;
}

int MlaAttention::load_param(const ncnn::ParamDict& pd)
{
    hidden_size_        = pd.get(0, 2560);
    num_heads_          = pd.get(1, 32);
    qk_nope_head_dim_   = pd.get(2, 128);
    qk_rope_head_dim_   = pd.get(3, 64);
    v_head_dim_         = pd.get(4, 128);
    kv_lora_rank_       = pd.get(5, 512);
    q_lora_rank_        = pd.get(6, 1536);
    intermediate_size_  = pd.get(7, 9728);
    layer_idx_          = pd.get(8, 0);

    qk_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
    scaling_     = 1.0f / sqrtf((float)qk_head_dim_);

    return 0;
}

int MlaAttention::load_model(const ncnn::ModelBin& mb)
{
    // Load weights as 1D first then reshape (workaround for ncnn mb.load(w,h) issue)
    q_a_proj_weight_               = mb.load(hidden_size_ * q_lora_rank_, 0).reshape(hidden_size_, q_lora_rank_);
    q_a_layernorm_weight_          = mb.load(q_lora_rank_, 0).reshape(q_lora_rank_, 1);
    q_b_proj_weight_               = mb.load(q_lora_rank_ * num_heads_ * qk_head_dim_, 0).reshape(q_lora_rank_, num_heads_ * qk_head_dim_);
    kv_a_proj_weight_              = mb.load(hidden_size_ * (kv_lora_rank_ + qk_rope_head_dim_), 0).reshape(hidden_size_, kv_lora_rank_ + qk_rope_head_dim_);
    kv_a_layernorm_weight_         = mb.load(kv_lora_rank_, 0).reshape(kv_lora_rank_, 1);
    kv_b_proj_weight_              = mb.load(kv_lora_rank_ * num_heads_ * (qk_nope_head_dim_ + v_head_dim_), 0).reshape(kv_lora_rank_, num_heads_ * (qk_nope_head_dim_ + v_head_dim_));
    o_proj_weight_                 = mb.load(num_heads_ * v_head_dim_ * hidden_size_, 0).reshape(num_heads_ * v_head_dim_, hidden_size_);
    input_layernorm_weight_        = mb.load(hidden_size_, 0).reshape(hidden_size_, 1);
    post_attention_layernorm_weight_ = mb.load(hidden_size_, 0).reshape(hidden_size_, 1);
    gate_proj_weight_              = mb.load(hidden_size_ * intermediate_size_, 0).reshape(hidden_size_, intermediate_size_);
    up_proj_weight_                = mb.load(hidden_size_ * intermediate_size_, 0).reshape(hidden_size_, intermediate_size_);
    down_proj_weight_              = mb.load(intermediate_size_ * hidden_size_, 0).reshape(intermediate_size_, hidden_size_);

    fprintf(stderr, "MLA load_model: input_ln_w[0:5]=");
    for (int i=0;i<5;i++) fprintf(stderr, "%.6f ", ((float*)input_layernorm_weight_.data)[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "MLA load_model: q_a_proj[0:3]=");
    for (int i=0;i<3;i++) fprintf(stderr, "%.6f ", ((float*)q_a_proj_weight_.data)[i]);
    fprintf(stderr, "\n");

    return 0;
}

int MlaAttention::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                           std::vector<ncnn::Mat>& top_blobs,
                           const ncnn::Option& opt) const
{
    const ncnn::Mat& hidden       = bottom_blobs[0];
    const ncnn::Mat& mask         = bottom_blobs[1];
    const ncnn::Mat& cos_cache    = bottom_blobs[2];
    const ncnn::Mat& sin_cache    = bottom_blobs[3];
    const ncnn::Mat& kv_latent_in = bottom_blobs[4];

    const int seq         = hidden.h;
    const int prev_kv_len = kv_latent_in.empty() ? 0 : kv_latent_in.h;
    const int total_kv_len = prev_kv_len + seq;
    const int kv_compressed = kv_lora_rank_ + qk_rope_head_dim_;

    const float* h   = (const float*)hidden;
    const float* m   = (const float*)mask;
    const float* cos = (const float*)cos_cache;
    const float* sin = (const float*)sin_cache;

    std::vector<float> normed(seq * hidden_size_);
    memcpy(normed.data(), h, seq * hidden_size_ * sizeof(float));
    rms_norm_inplace(normed.data(), (const float*)input_layernorm_weight_, seq, hidden_size_);

    std::vector<float> q_hidden(seq * q_lora_rank_);
    matmul_nt(normed.data(), (const float*)q_a_proj_weight_, q_hidden.data(),
              seq, hidden_size_, q_lora_rank_);
    rms_norm_inplace(q_hidden.data(), (const float*)q_a_layernorm_weight_, seq, q_lora_rank_);

    std::vector<float> q_full(seq * num_heads_ * qk_head_dim_);
    matmul_nt(q_hidden.data(), (const float*)q_b_proj_weight_, q_full.data(),
              seq, q_lora_rank_, num_heads_ * qk_head_dim_);

    std::vector<float> kv_cur(seq * kv_compressed);
    matmul_nt(normed.data(), (const float*)kv_a_proj_weight_, kv_cur.data(),
              seq, hidden_size_, kv_compressed);

    std::vector<float> Q(num_heads_ * seq * qk_head_dim_);
    std::vector<float> K(num_heads_ * total_kv_len * qk_head_dim_);
    std::vector<float> V(num_heads_ * total_kv_len * v_head_dim_);
    std::vector<float> kv_latent_out(total_kv_len * kv_compressed);

    // Build kv_latent_out: OLD entries first, then NEW entries (chronological order)
    if (prev_kv_len > 0)
    {
        memcpy(&kv_latent_out[0],
               (const float*)kv_latent_in,
               prev_kv_len * kv_compressed * sizeof(float));
    }

    for (int s = 0; s < seq; s++)
    {
        memcpy(&kv_latent_out[(prev_kv_len + s) * kv_compressed],
               &kv_cur[s * kv_compressed],
               kv_compressed * sizeof(float));
    }

    for (int s = 0; s < seq; s++)
    {
        for (int hh = 0; hh < num_heads_; hh++)
        {
            int q_src = s * num_heads_ * qk_head_dim_ + hh * qk_head_dim_;
            int q_dst = hh * seq * qk_head_dim_ + s * qk_head_dim_;
            memcpy(&Q[q_dst], &q_full[q_src], qk_nope_head_dim_ * sizeof(float));
            memcpy(&Q[q_dst + qk_nope_head_dim_],
                   &q_full[q_src + qk_nope_head_dim_],
                   qk_rope_head_dim_ * sizeof(float));
        }
    }

    for (int pos = 0; pos < total_kv_len; pos++)
    {
        const float* latent  = &kv_latent_out[pos * kv_compressed];
        const float* k_lat   = latent;
        const float* k_rot_r = latent + kv_lora_rank_;

        std::vector<float> k_lat_normed(kv_lora_rank_);
        memcpy(k_lat_normed.data(), k_lat, kv_lora_rank_ * sizeof(float));
        rms_norm_inplace(k_lat_normed.data(), (const float*)kv_a_layernorm_weight_, 1, kv_lora_rank_);

        int expand_dim = num_heads_ * (qk_nope_head_dim_ + v_head_dim_);
        std::vector<float> expanded(expand_dim);
        matmul_nt(k_lat_normed.data(), (const float*)kv_b_proj_weight_, expanded.data(),
                  1, kv_lora_rank_, expand_dim);

        for (int hh = 0; hh < num_heads_; hh++)
        {
            int k_nope_src = hh * (qk_nope_head_dim_ + v_head_dim_);
            int k_dst = hh * total_kv_len * qk_head_dim_ + pos * qk_head_dim_;
            memcpy(&K[k_dst], &expanded[k_nope_src], qk_nope_head_dim_ * sizeof(float));

            int k_rope_dst = k_dst + qk_nope_head_dim_;
            memcpy(&K[k_rope_dst], k_rot_r, qk_rope_head_dim_ * sizeof(float));

            int v_src = k_nope_src + qk_nope_head_dim_;
            int v_dst = hh * total_kv_len * v_head_dim_ + pos * v_head_dim_;
            memcpy(&V[v_dst], &expanded[v_src], v_head_dim_ * sizeof(float));
        }
    }

    int rope_half = qk_rope_head_dim_ / 2;
    apply_interleave_rope(Q.data(), num_heads_, seq,
                           qk_head_dim_, qk_nope_head_dim_, qk_rope_head_dim_,
                           cos, sin, cos_cache.w, /*start_pos=*/total_kv_len - seq);
    apply_interleave_rope(K.data(), num_heads_, total_kv_len,
                           qk_head_dim_, qk_nope_head_dim_, qk_rope_head_dim_,
                           cos, sin, cos_cache.w, /*start_pos=*/0);

    int out_dim = num_heads_ * v_head_dim_;
    std::vector<float> attn(seq * out_dim);

    #pragma omp parallel for
    for (int hh = 0; hh < num_heads_; hh++)
    {
        const float* Qh = Q.data() + hh * seq * qk_head_dim_;
        const float* Kh = K.data() + hh * total_kv_len * qk_head_dim_;
        const float* Vh = V.data() + hh * total_kv_len * v_head_dim_;

        std::vector<float> scores(seq * total_kv_len);
        matmul_nt(Qh, Kh, scores.data(), seq, qk_head_dim_, total_kv_len);

        for (int i = 0; i < seq * total_kv_len; i++)
            scores[i] = scores[i] * scaling_ + m[i];
        softmax_last_dim(scores.data(), seq, total_kv_len);

        float* attn_h = attn.data() + hh * v_head_dim_;
        for (int s = 0; s < seq; s++)
        {
            for (int d = 0; d < v_head_dim_; d++)
            {
                float sum = 0.f;
                for (int k = 0; k < total_kv_len; k++)
                    sum += scores[s * total_kv_len + k] * Vh[k * v_head_dim_ + d];
                attn_h[s * out_dim + d] = sum;
            }
        }
    }

    std::vector<float> attn_proj(seq * hidden_size_);
    matmul_nt(attn.data(), (const float*)o_proj_weight_, attn_proj.data(),
              seq, out_dim, hidden_size_);

    std::vector<float> pa(seq * hidden_size_);
    for (int i = 0; i < seq * hidden_size_; i++)
        pa[i] = h[i] + attn_proj[i];

    rms_norm_inplace(pa.data(), (const float*)post_attention_layernorm_weight_, seq, hidden_size_);

    std::vector<float> gate_act(seq * intermediate_size_);
    std::vector<float> up_act(seq * intermediate_size_);
    matmul_nt(pa.data(), (const float*)gate_proj_weight_, gate_act.data(),
              seq, hidden_size_, intermediate_size_);
    matmul_nt(pa.data(), (const float*)up_proj_weight_, up_act.data(),
              seq, hidden_size_, intermediate_size_);

    for (int i = 0; i < seq * intermediate_size_; i++)
    {
        float g = gate_act[i];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        gate_act[i] = g * sigmoid_g * up_act[i];
    }

    std::vector<float> ffn_out(seq * hidden_size_);
    matmul_nt(gate_act.data(), (const float*)down_proj_weight_, ffn_out.data(),
              seq, intermediate_size_, hidden_size_);

    ncnn::Mat& out0 = top_blobs[0];
    out0.create(hidden_size_, seq, 4u, opt.blob_allocator);
    float* out0_ptr = (float*)out0.data;
    for (int i = 0; i < seq * hidden_size_; i++)
        out0_ptr[i] = h[i] + attn_proj[i] + ffn_out[i];

    ncnn::Mat& out1 = top_blobs[1];
    out1.create(kv_compressed, total_kv_len, 4u, opt.blob_allocator);
    memcpy(out1.data, kv_latent_out.data(), total_kv_len * kv_compressed * sizeof(float));

    return 0;
}

ncnn::Layer* MlaAttention_creator(void*)
{
    return new MlaAttention;
}

void MlaAttention_destroyer(ncnn::Layer* layer, void*)
{
    delete layer;
}
