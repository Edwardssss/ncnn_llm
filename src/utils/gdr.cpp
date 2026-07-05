#include "gdr.h"
#include "mla_attention.h"
#include <cmath>
#include <cstring>

static void l2norm(const float* x, float* out, int n, int dim, float eps)
{
    for (int i = 0; i < n; i++)
    {
        const float* row_in = x + i * dim;
        float* row_out = out + i * dim;

        float sum = 0.f;
        for (int j = 0; j < dim; j++)
        {
            sum += row_in[j] * row_in[j];
        }
        float inv_norm = 1.f / sqrtf(sum + eps);

        for (int j = 0; j < dim; j++)
        {
            row_out[j] = row_in[j] * inv_norm;
        }
    }
}

static float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

static float softplusf(float x)
{
    if (x > 20.f)
        return x;
    if (x < -20.f)
        return expf(x);
    return logf(1.f + expf(x));
}

static void torch_recurrent_gated_delta_rule(
    const float* query, const float* key, const float* value,
    const float* g, const float* beta,
    float* core_attn_out,
    float* last_recurrent_state,
    int batch_size, int num_heads, int seq_len,
    int k_head_dim, int v_head_dim,
    bool use_qk_l2norm_in_kernel)
{
    std::vector<float> query_norm;
    std::vector<float> key_norm;

    const float* q_ptr = query;
    const float* k_ptr = key;

    int qk_size = batch_size * num_heads * seq_len * k_head_dim;
    if (use_qk_l2norm_in_kernel)
    {
        query_norm.resize(qk_size);
        key_norm.resize(qk_size);

        l2norm(query, query_norm.data(), batch_size * num_heads * seq_len, k_head_dim, 1e-6f);
        l2norm(key, key_norm.data(), batch_size * num_heads * seq_len, k_head_dim, 1e-6f);

        q_ptr = query_norm.data();
        k_ptr = key_norm.data();
    }

    float scale = 1.f / sqrtf((float)k_head_dim);

    memset(core_attn_out, 0, batch_size * num_heads * seq_len * v_head_dim * sizeof(float));

    for (int t = 0; t < seq_len; t++)
    {
        for (int b = 0; b < batch_size; b++)
        {
            for (int h = 0; h < num_heads; h++)
            {
                const float* q_t = q_ptr + ((b * num_heads + h) * seq_len + t) * k_head_dim;
                const float* k_t = k_ptr + ((b * num_heads + h) * seq_len + t) * k_head_dim;
                const float* v_t = value + ((b * num_heads + h) * seq_len + t) * v_head_dim;

                float g_t = g[(b * num_heads + h) * seq_len + t];
                float beta_t = beta[(b * num_heads + h) * seq_len + t];

                float* state = last_recurrent_state + (b * num_heads + h) * k_head_dim * v_head_dim;
                float* out_t = core_attn_out + ((b * num_heads + h) * seq_len + t) * v_head_dim;

                float g_t_exp = expf(g_t);

                for (int i = 0; i < k_head_dim * v_head_dim; i++)
                {
                    state[i] *= g_t_exp;
                }

                std::vector<float> kv_mem(v_head_dim, 0.f);
                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    for (int dk = 0; dk < k_head_dim; dk++)
                    {
                        kv_mem[dv] += state[dk * v_head_dim + dv] * k_t[dk];
                    }
                }

                std::vector<float> delta(v_head_dim);
                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    delta[dv] = (v_t[dv] - kv_mem[dv]) * beta_t;
                }

                for (int dk = 0; dk < k_head_dim; dk++)
                {
                    for (int dv = 0; dv < v_head_dim; dv++)
                    {
                        state[dk * v_head_dim + dv] += k_t[dk] * delta[dv];
                    }
                }

                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    float sum = 0.f;
                    for (int dk = 0; dk < k_head_dim; dk++)
                    {
                        sum += state[dk * v_head_dim + dv] * q_t[dk] * scale;
                    }
                    out_t[dv] = sum;
                }
            }
        }
    }
}

GatedDeltaRule::GatedDeltaRule()
{
    one_blob_only = false;
    support_inplace = false;

    num_k_heads = 128;
    num_v_heads = 128;
}

int GatedDeltaRule::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    const ncnn::Mat& A_log = bottom_blobs[0];
    const ncnn::Mat& dt_bias = bottom_blobs[1];
    const ncnn::Mat& b = bottom_blobs[2];
    const ncnn::Mat& a = bottom_blobs[3];
    const ncnn::Mat& query = bottom_blobs[4];
    const ncnn::Mat& key = bottom_blobs[5];
    const ncnn::Mat& value = bottom_blobs[6];
    const ncnn::Mat& initial_state = bottom_blobs[7];

    int num_heads = query.h;
    int seq_len = query.c;
    int k_head_dim = query.w;
    int v_head_dim = value.w;

    bool use_qk_l2norm_in_kernel = true;

    ncnn::Mat& top_blob = top_blobs[0];
    top_blob.create(k_head_dim, num_heads, seq_len, 4u, opt.blob_allocator);

    ncnn::Mat& state_out = top_blobs[1];
    state_out.create(v_head_dim, k_head_dim, num_heads, 4u, opt.blob_allocator);

    if (top_blob.empty() || state_out.empty())
        return -100;

    const float* query_data = (const float*)query.data;
    const float* key_data = (const float*)key.data;
    const float* value_data = (const float*)value.data;

    std::vector<float> query_t(num_heads * seq_len * k_head_dim);
    std::vector<float> key_t(num_heads * seq_len * k_head_dim);
    std::vector<float> value_t(num_heads * seq_len * v_head_dim);

    for (int t = 0; t < seq_len; t++)
    {
        for (int h = 0; h < num_heads; h++)
        {
            for (int d = 0; d < k_head_dim; d++)
            {
                int src_idx = (t * num_heads + h) * k_head_dim + d;
                int dst_idx = (h * seq_len + t) * k_head_dim + d;
                query_t[dst_idx] = query_data[src_idx];
                key_t[dst_idx] = key_data[src_idx];
            }
            for (int d = 0; d < v_head_dim; d++)
            {
                int src_idx = (t * num_heads + h) * v_head_dim + d;
                int dst_idx = (h * seq_len + t) * v_head_dim + d;
                value_t[dst_idx] = value_data[src_idx];
            }
        }
    }

    const float* b_data = (const float*)b.data;
    const float* a_data = (const float*)a.data;
    const float* A_log_data = (const float*)A_log.data;
    const float* dt_bias_data = (const float*)dt_bias.data;

    std::vector<float> beta(num_heads * seq_len);
    std::vector<float> g(num_heads * seq_len);

    for (int h = 0; h < num_heads; h++)
    {
        for (int t = 0; t < seq_len; t++)
        {
            float b_val = b_data[t * num_heads + h];
            beta[h * seq_len + t] = sigmoidf(b_val);
        }
    }

    for (int h = 0; h < num_heads; h++)
    {
        float A_log_val = A_log_data[h];
        float dt_bias_val = dt_bias_data[h];
        float exp_A = expf(A_log_val);

        for (int t = 0; t < seq_len; t++)
        {
            float a_val = a_data[t * num_heads + h];
            float sp_val = softplusf(a_val + dt_bias_val);
            g[h * seq_len + t] = -exp_A * sp_val;
        }
    }

    int batch_size = 1;

    std::vector<float> core_attn_out(num_heads * seq_len * v_head_dim);
    std::vector<float> last_recurrent_state(num_heads * k_head_dim * v_head_dim);

    if (!initial_state.empty())
    {
        memcpy(last_recurrent_state.data(), initial_state.data,
               num_heads * k_head_dim * v_head_dim * sizeof(float));
    }
    else
    {
        memset(last_recurrent_state.data(), 0,
               num_heads * k_head_dim * v_head_dim * sizeof(float));
    }

    torch_recurrent_gated_delta_rule(
        query_t.data(), key_t.data(), value_t.data(),
        g.data(), beta.data(),
        core_attn_out.data(),
        last_recurrent_state.data(),
        batch_size, num_heads, seq_len,
        k_head_dim, v_head_dim,
        use_qk_l2norm_in_kernel);

    float* top_data = (float*)top_blob.data;
    for (int h = 0; h < num_heads; h++)
    {
        for (int t = 0; t < seq_len; t++)
        {
            for (int d = 0; d < v_head_dim; d++)
            {
                int src_idx = (h * seq_len + t) * v_head_dim + d;
                int dst_idx = (t * num_heads + h) * v_head_dim + d;
                top_data[dst_idx] = core_attn_out[src_idx];
            }
        }
    }

    memcpy((float*)state_out.data, last_recurrent_state.data(),
           num_heads * k_head_dim * v_head_dim * sizeof(float));

    return 0;
}

ShortConv::ShortConv()
{
    one_blob_only = false;
    support_inplace = false;
}

int ShortConv::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    const ncnn::Mat& weight_mat = bottom_blobs[0];
    const ncnn::Mat& mixed_qkv = bottom_blobs[1];
    const ncnn::Mat& conv_state = bottom_blobs[2];

    int seq_len = mixed_qkv.h;
    int groups = mixed_qkv.w;
    int kernel_size = weight_mat.w;

    ncnn::Mat stated_mixed_qkv;
    if (conv_state.empty())
    {
        stated_mixed_qkv.create(groups, kernel_size - 1 + seq_len, 4u, opt.blob_allocator);
        memset(stated_mixed_qkv.row(0), 0, (kernel_size - 1) * groups * sizeof(float));
        memcpy(stated_mixed_qkv.row(kernel_size - 1), mixed_qkv, mixed_qkv.h * groups * sizeof(float));
    }
    else
    {
        stated_mixed_qkv.create(groups, conv_state.h + seq_len, 4u, opt.blob_allocator);
        memcpy(stated_mixed_qkv.row(0), conv_state, conv_state.h * groups * sizeof(float));
        memcpy(stated_mixed_qkv.row(conv_state.h), mixed_qkv, mixed_qkv.h * groups * sizeof(float));
    }

    int state_len = kernel_size;
    int total_len = conv_state.empty() ? (kernel_size - 1 + seq_len) : (conv_state.h + seq_len);
    ncnn::Mat last_conv_state(groups, state_len, 4u, opt.blob_allocator);
    memcpy(last_conv_state.data, stated_mixed_qkv.row(total_len - state_len),
           state_len * groups * sizeof(float));

    ncnn::Mat& top_blob = top_blobs[0];
    top_blob.create(groups, seq_len, 4u, opt.blob_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < groups; g++)
    {
        const float* w_ptr = weight_mat.channel(g);

        for (int i = 0; i < seq_len; i++)
        {
            float sum = 0.f;

            int prefix_len = conv_state.empty() ? (kernel_size - 1) : conv_state.h;
            int base = prefix_len + i;

            for (int k = 0; k < kernel_size; k++)
            {
                int src_i = base - (kernel_size - 1) + k;
                sum += stated_mixed_qkv.row(src_i)[g] * w_ptr[k];
            }

            top_blob.row(i)[g] = sum * (1.f / (1.f + expf(-sum)));
        }
    }

    top_blobs[1] = last_conv_state;

    return 0;
}

static ncnn::Layer* GatedDeltaRule_creator(void*)
{
    return new GatedDeltaRule;
}

static void GatedDeltaRule_destroyer(ncnn::Layer* layer, void*)
{
    delete layer;
}

static ncnn::Layer* ShortConv_creator(void*)
{
    return new ShortConv;
}

static void ShortConv_destroyer(ncnn::Layer* layer, void*)
{
    delete layer;
}

void register_gdr_layers(ncnn::Net& net)
{
    net.register_custom_layer("GatedDeltaRule", GatedDeltaRule_creator, GatedDeltaRule_destroyer);
    net.register_custom_layer("ShortConv", ShortConv_creator, ShortConv_destroyer);
    net.register_custom_layer("MlaAttention", MlaAttention_creator, MlaAttention_destroyer);
}