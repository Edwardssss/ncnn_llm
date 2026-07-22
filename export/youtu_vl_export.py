"""Youtu-VL 模型权重导出 (HF safetensors → ncnn .param/.bin)
===========================================================

从 Youtu-VL-4B-Instruct HuggingFace 权重构建所有 ncnn 子网络。

生成文件:
  youtu_embed.ncnn.{param,bin}         — Token 嵌入 (Embed 283386×2560)
  youtu_lm_head.ncnn.{param,bin}       — 语言模型头 (Gemm, tied embeddings)
  youtu_merger_rmsnorm.ncnn.{param,bin} — 视觉融合 RMSNorm
  youtu_merger_mlp.ncnn.{param,bin}    — 视觉融合 MLP (fc1→GELU→fc2)
  youtu_decoder.ncnn.{param,bin}       — MLA Decoder (40层)
  youtu_vision_full.ncnn.{param,bin}   — 视觉编码器 (pnnx 导出, 需要 pnnx)

用法:
  python export/youtu_vl_export.py --model ./Youtu-VL-4B-Instruct --output ncnn_models/  [--no-vision]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

FLAG_RAW_FP32 = b'\x00\x00\x00\x00'  # ncnn ModelBin raw float32 标志

# ────────────────────────── Weight helpers ──────────────────────────

def _load_safetensors(model_dir):
    """加载 HF 模型权重索引和所有分片。"""
    with open(model_dir / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]

    from safetensors import safe_open
    shards = {}
    for fn in sorted(model_dir.glob("model-*.safetensors")):
        shards[fn.name] = safe_open(str(fn), framework="pt", device="cpu")
    return wm, shards


def _get_tensor(wm, shards, name):
    """从分片中读取单个 tensor 并转为 float32 numpy。"""
    sname = wm[name]
    return shards[sname].get_tensor(name).float().numpy()


# ──────────────────── 文本 & 融合模块 ────────────────────

def _build_text_embed(out_dir, wm, shards):
    embed_w = _get_tensor(wm, shards, "model.embed_tokens.weight")
    nparam = ("7767517\n2 3\n"
              "Input  in0  0 1 in0\n"
              f"Embed  embed  1 1 in0 out0  0=2560 1=283386 2=0 3={283386*2560}\n")
    (out_dir / "youtu_embed.ncnn.param").write_text(nparam)
    with open(out_dir / "youtu_embed.ncnn.bin", "wb") as f:
        f.write(FLAG_RAW_FP32)
        f.write(embed_w.tobytes())
    print(f"  embed: {embed_w.shape}")
    return embed_w


def _build_lm_head(out_dir, embed_weight):
    nparam = ("7767517\n2 3\n"
              "Input  in0  0 1 in0\n"
              "Gemm  out  1 1 in0 out0  "
              "0=1 1=0 2=0 3=1 4=0 5=1 6=0 7=0 8=283386 9=2560\n")
    (out_dir / "youtu_lm_head.ncnn.param").write_text(nparam)
    with open(out_dir / "youtu_lm_head.ncnn.bin", "wb") as f:
        f.write(FLAG_RAW_FP32)
        f.write(embed_weight.tobytes())
    print(f"  lm_head: {embed_weight.shape}")


def _build_merger(out_dir, wm, shards):
    ln_q   = _get_tensor(wm, shards, "merger.ln_q.weight")       # [1152]
    mlp0_w = _get_tensor(wm, shards, "merger.mlp.0.weight")      # [4608,4608]
    mlp0_b = _get_tensor(wm, shards, "merger.mlp.0.bias")        # [4608]
    mlp2_w = _get_tensor(wm, shards, "merger.mlp.2.weight")      # [2560,4608]
    mlp2_b = _get_tensor(wm, shards, "merger.mlp.2.bias")        # [2560]

    # RMSNorm 子模型
    rms_param = ("7767517\n2 2\n"
                 "Input  in0  0 1 in0\n"
                 "RMSNorm  rmsnorm  1 1 in0 out0  0=1152 1=1e-06\n")
    (out_dir / "youtu_merger_rmsnorm.ncnn.param").write_text(rms_param)
    with open(out_dir / "youtu_merger_rmsnorm.ncnn.bin", "wb") as f:
        f.write(FLAG_RAW_FP32)
        f.write(ln_q.tobytes())

    # MLP 子模型  (MemoryData → Gemm → GELU → MemoryData → Gemm)
    mlp_param = ("7767517\n8 8\n"
                 "Input  in0  0 1 in0\n"
                 "MemoryData  mlp0_w  0 1 1  0=4608 1=4608 21=0\n"
                 "MemoryData  mlp0_b  0 1 2  0=4608 21=0\n"
                 "Gemm  fc1  3 1 in0 1 2 3  0=4608 1=4608 2=0 3=0 4=0 5=1 6=1 7=0 8=4608 9=4608\n"
                 "GELU  gelu  1 1 3 4\n"
                 "MemoryData  mlp2_w  0 1 5  0=4608 1=2560 21=0\n"
                 "MemoryData  mlp2_b  0 1 6  0=2560 21=0\n"
                 "Gemm  fc2  3 1 4 5 6 out0  0=2560 1=4608 2=0 3=0 4=0 5=1 6=1 7=0 8=2560 9=4608\n")
    (out_dir / "youtu_merger_mlp.ncnn.param").write_text(mlp_param)
    with open(out_dir / "youtu_merger_mlp.ncnn.bin", "wb") as f:
        f.write(FLAG_RAW_FP32)
        f.write(mlp0_w.tobytes())
        f.write(mlp0_b.tobytes())
        f.write(mlp2_w.tobytes())
        f.write(mlp2_b.tobytes())

    print(f"  merger: mlp0 {mlp0_w.shape}, mlp2 {mlp2_w.shape}")


def _build_decoder(out_dir, wm, shards):
    N = 40           # num_layers
    H = 2560         # hidden_size
    I = 9728         # intermediate_size
    HEADS = 32
    NO = 128         # qk_nope_head_dim
    RO = 64          # qk_rope_head_dim
    VH = 128         # v_head_dim
    KV = 512         # kv_lora_rank
    QL = 1536        # q_lora_rank

    with open(out_dir / "youtu_decoder.ncnn.bin", "wb") as bf:
        for li in range(N):
            p = f"model.layers.{li}."
            for name in [
                f"{p}self_attn.q_a_proj.weight", f"{p}self_attn.q_a_layernorm.weight",
                f"{p}self_attn.q_b_proj.weight", f"{p}self_attn.kv_a_proj_with_mqa.weight",
                f"{p}self_attn.kv_a_layernorm.weight", f"{p}self_attn.kv_b_proj.weight",
                f"{p}self_attn.o_proj.weight", f"{p}input_layernorm.weight",
                f"{p}post_attention_layernorm.weight", f"{p}mlp.gate_proj.weight",
                f"{p}mlp.up_proj.weight", f"{p}mlp.down_proj.weight",
            ]:
                bf.write(_get_tensor(wm, shards, name).tobytes())
        bf.write(_get_tensor(wm, shards, "model.norm.weight").tobytes())

    # 构建 param: 4 固定输入 + N 层 MLA + 1 层 final_norm
    layers = []
    for li in range(N):
        hid_in = 0 if li == 0 else 5 + 2 * (li - 1)
        kv_in  = 4 if li == 0 else 5 + 2 * (li - 1) + 1
        attn_out = 5 + 2 * li
        kv_out   = 5 + 2 * li + 1
        layers.append(
            f"MlaAttention  mla_{li}  5 2 "
            f"in{hid_in} in1 in2 in3 in{kv_in} out{attn_out} out{kv_out}  "
            f"0={H} 1={HEADS} 2={NO} 3={RO} 4={VH} 5={KV} 6={QL} 7={I} 8={li}"
        )
    last_attn = 5 + 2 * (N - 1)
    last_kv   = 5 + 2 * (N - 1) + 1
    norm_out  = last_kv + 1
    total_blobs = norm_out + 1
    layers.append(f"RMSNorm  final_norm  1 1 out{last_attn} out{norm_out}  0={H} 1=1e-06")

    header = "7767517\n%d %d\nInput  in0  0 1 in0\nInput  in1  0 1 in1\nInput  in2  0 1 in2\nInput  in3  0 1 in3\n" % (4 + len(layers), total_blobs)
    (out_dir / "youtu_decoder.ncnn.param").write_text(
        header + "\n".join(layers) + "\nOutput  out0  1 0\n"
    )
    print(f"  decoder: {N} layers, {total_blobs} blobs")


# ─────────────────────── Vision encoder ───────────────────────

def _export_vision_encoder(model_dir, out_dir):
    """使用 pnnx 导出全注意力 SigLIP2 视觉编码器 (无 RoPE / 无窗口注意力)。"""
    import importlib.util

    # 从 HF 模型目录动态导入 configuration 和 modeling (避免复制大文件)
    def _load(mod_name, filepath):
        spec = importlib.util.spec_from_file_location(mod_name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("configuration_siglip2", str(model_dir / "configuration_siglip2.py"))
    _load("modeling_siglip2", str(model_dir / "modeling_siglip2.py"))
    from configuration_siglip2 import Siglip2VisionConfig
    from modeling_siglip2 import Siglip2VisionTransformer
    import safetensors.torch as st

    print("  loading vision config & model...")

    # 消除 PyTorch ≥2.6 的 aten::resolve_conj
    try:
        torch._C._set_conj_trace(False)
        torch._C._set_neg_trace(False)
    except Exception:
        pass

    # 强制使用手动 matmul+softmax (避开 SDPA / aten::resolve_conj)
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, scale=None):
        if scale is None:
            scale = 1.0 / math.sqrt(query.size(-1))
        w = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_mask is not None:
            w = w + attn_mask
        return torch.matmul(torch.softmax(w, dim=-1), value)
    torch.nn.functional.scaled_dot_product_attention = _manual_sdpa

    cfg = Siglip2VisionConfig.from_pretrained(str(model_dir))
    cfg._attn_implementation = "eager"
    model = Siglip2VisionTransformer(cfg).eval()

    # 加载 HF 权重
    with open(model_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    sd = {}
    for fn in sorted(set(idx["weight_map"].values())):
        d = st.load_file(str(model_dir / fn))
        for k, v in d.items():
            if k.startswith("siglip2.vision_model."):
                sd[k.replace("siglip2.vision_model.", "")] = v.float()
    model.load_state_dict(sd, strict=False)
    print(f"    loaded {len(sd)} vision params")

    # 替换 encoder.forward → 全注意力 (无 RoPE, 无 window attention)
    def _full_attn_forward(self, inputs_embeds, **_):
        import transformers.modeling_outputs as mo
        h = inputs_embeds
        for layer in self.layers:
            out = layer(h, attention_mask=None,
                        cu_seqlens=torch.tensor([0, 256], dtype=torch.int32),
                        rotary_pos_emb=None, position_embeddings=None)
            h = out[0]
        return mo.BaseModelOutput(last_hidden_state=h)
    model.encoder.forward = _full_attn_forward.__get__(model.encoder)

    # 修补 Vision_EagerAttention: 跳过 RoPE + 跳过 attention mask
    for layer in model.encoder.layers:
        def _patched_attn(self, hidden_states, cu_seqlens,
                          rotary_pos_emb=None, position_embeddings=None):
            L = hidden_states.shape[0]
            q = self.q_proj(hidden_states).view(L, self.num_heads, self.head_dim)
            k = self.k_proj(hidden_states).view(L, self.num_heads, self.head_dim)
            v = self.v_proj(hidden_states).view(L, self.num_heads, self.head_dim)
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
            return self.out_proj(out.squeeze(0).transpose(0, 1).reshape(L, -1).to(hidden_states.dtype)), None
        layer.self_attn.forward = _patched_attn.__get__(layer.self_attn)

    # 包装: pixel_values → last_hidden_state
    class VisionWrapper(torch.nn.Module):
        def __init__(self, vision):
            super().__init__()
            self.vision = vision
        def forward(self, pixels):
            return self.vision(pixels, attention_mask=None,
                               spatial_shapes=torch.tensor([[16, 16]], dtype=torch.long)).last_hidden_state

    wrapper = VisionWrapper(model).eval()
    pixels = torch.randn(1, 256, 768)

    print("  exporting via pnnx ...")
    import pnnx
    pnnx.export(wrapper, str(out_dir / "youtu_vision_full.pt"),
                inputs=(pixels,), fp16=False, device="cpu", check_trace=False)

    # 后处理: 删除 aten::resolve_conj / aten::resolve_neg (安全直通层)
    param_path = out_dir / "youtu_vision_full.ncnn.param"
    with open(param_path) as f:
        lines = f.readlines()
    magic = lines[0].rstrip('\n')

    alias = {}
    keep = []
    for l in lines[2:]:
        parts = l.split()
        if not parts:
            continue
        if parts[0] in ('aten::resolve_conj', 'aten::resolve_neg'):
            if len(parts) >= 5:
                alias[parts[4]] = parts[3]
            continue
        keep.append(parts)

    for k in list(alias):
        v = alias[k]
        while v in alias:
            v = alias[v]
        alias[k] = v
    for row in keep:
        for i, tok in enumerate(row):
            if tok in alias:
                row[i] = alias[tok]

    # 重新计算 header
    input_rows = sum(1 for r in keep if r[0] == 'Input')
    output_rows = sum(1 for r in keep if r[0] == 'Output')
    layer_count = len(keep) - input_rows - output_rows
    blob_idxs = set()
    for row in keep:
        n_in = int(row[2]) if len(row) > 2 and row[2].isdigit() else 0
        n_out = int(row[3]) if len(row) > 3 and row[3].isdigit() else 0
        for tok in row[4:4 + n_in + n_out]:
            if tok.isdigit():
                blob_idxs.add(int(tok))
    bottom_count = max(blob_idxs) + 1 if blob_idxs else 1

    with open(param_path, 'w') as f:
        f.write(magic + '\n')
        f.write(f'{layer_count} {bottom_count}\n')
        for row in keep:
            f.write(' '.join(row) + '\n')

    # 检查残留不支持的算子
    bad = [r[0] for r in keep if r[0].startswith(('aten::', 'torch.', 'pnnx.'))]
    if bad:
        print(f"  WARNING: {len(bad)} unsupported ops remain: {bad[:5]}")
    else:
        print(f"  vision: {layer_count} layers, {bottom_count} blobs (stripped {len(alias)} resolve_* ops)")

    # 更新 model.json 中的 vision encoder 路径
    mj_path = out_dir / "model.json"
    if mj_path.exists():
        with open(mj_path) as f:
            cfg = json.load(f)
        v = cfg["setting"]["vision"]
        v["vision_encoder_param"] = "youtu_vision_full.ncnn.param"
        v["vision_encoder_bin"]   = "youtu_vision_full.ncnn.bin"
        with open(mj_path, "w") as f:
            json.dump(cfg, f, indent=4)
            f.write("\n")
        print("  model.json updated")


# ─────────────────────────── 入口 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Youtu-VL 模型权重导出")
    parser.add_argument("--model",  default="./Youtu-VL-4B-Instruct",
                        help="HF 模型目录 (必须包含 model.safetensors.index.json)")
    parser.add_argument("--output", default="ncnn_models",
                        help="ncnn 权重输出目录")
    parser.add_argument("--no-vision", action="store_true",
                        help="跳过视觉编码器导出 (需要 pnnx)")
    args = parser.parse_args()

    model_dir = Path(args.model).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (model_dir / "model.safetensors.index.json").exists():
        print(f"ERROR: {model_dir} 不是有效的 HF 模型目录 (缺少 model.safetensors.index.json)")
        sys.exit(1)

    # 需要 pnnx 包
    if not args.no_vision:
        try:
            import pnnx
        except ImportError:
            print("ERROR: pnnx 未安装 (pip install pnnx)。使用 --no-vision 跳过视觉编码器导出。")
            sys.exit(1)

    print("[1/2] 构建文本 & 融合模块权重")
    wm, shards = _load_safetensors(model_dir)
    embed_w = _build_text_embed(out_dir, wm, shards)
    _build_lm_head(out_dir, embed_w)
    _build_merger(out_dir, wm, shards)
    _build_decoder(out_dir, wm, shards)
    for s in shards.values():
        s.close()

    if not args.no_vision:
        print("[2/2] 导出视觉编码器 (pnnx)")
        _export_vision_encoder(model_dir, out_dir)

    print(f"\nDone. 模型文件位于: {out_dir}")


if __name__ == "__main__":
    main()
