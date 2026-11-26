"""Gemma 4 Unified teacher for calibration."""

import math

import torch
from torch.nn import functional as F

from activation import SITES, expand_mlp

PREFIX = "model.language_model."
E4M3 = torch.float8_e4m3fn


def rms(value, weight=None):
    x = value.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    return (x if weight is None else x * weight).to(value.dtype)


def rope(length, width, theta, fraction, device):
    angles = int(fraction * width // 2)
    frequency = 1 / theta ** (torch.arange(0, 2 * angles, 2).float() / width)
    frequency = F.pad(frequency, (0, width // 2 - angles))
    phase = torch.outer(torch.arange(length).float(), frequency).repeat(1, 2)
    return phase.cos().to(device, torch.bfloat16), phase.sin().to(device, torch.bfloat16)


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos[:, None] + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin[:, None]


def bf16_carrier(weight, attention):
    # The BF16 carrier prefill uses the expanded stand-in weight codes.
    work = weight.float()
    tiny = torch.finfo(torch.float32).tiny
    if attention:
        scale = (work.abs().amax(-1, keepdim=True) / 448).clamp_min(tiny).bfloat16().float()
        return ((work / scale).to(E4M3).float() * scale).bfloat16().float(), 1.
    global_scale = (work.abs().amax() / (6 * 448)).clamp_min(tiny)
    blocks = (work / global_scale).unflatten(-1, (-1, 16))
    scale = (blocks.abs().amax(-1, keepdim=True) / 6).clamp_min(tiny)
    scale = scale.to(E4M3).float()
    normalized = blocks / scale
    boundaries = work.new_tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    levels = work.new_tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    codes = levels[torch.bucketize(normalized.abs(), boundaries)] * normalized.sign().to(torch.int8)
    return (codes * scale).flatten(1).bfloat16().float(), float(global_scale)


def fp8_carrier(checkpoint, name, attention, device):
    if attention:
        return (checkpoint.get_tensor(name + ".weight").to(device),
                checkpoint.get_tensor(name + ".weight_scale").to(device, torch.float32).flatten())
    shape = checkpoint.get_slice(name + ".weight_packed").get_shape()
    codes = torch.empty(shape[0], 2 * shape[1], dtype=E4M3, device=device)
    scales = torch.empty(shape[0], dtype=torch.float32, device=device)
    for start in range(0, shape[0], 512):
        end = min(start + 512, shape[0])
        expanded = expand_mlp(checkpoint, name, start, end, device=device)
        scale = (expanded.abs().amax(-1, keepdim=True) / 256).clamp_min(1e-30)
        codes[start:end] = (expanded / scale).to(E4M3)
        scales[start:end] = scale.flatten()
    global_scale = 1. / float(checkpoint.get_tensor(name + ".weight_global_scale"))
    return codes, scales * global_scale


def teacher(checkpoint, config, prompts, mode, embedding, *, observe=None, maxima=None, cache=None):
    """Share each layer's expanded weights across contexts and return final-norm rows."""
    width = config["hidden_size"]
    device = embedding.device
    activation = mode == "activation"
    embedding_scale = torch.tensor(math.sqrt(width), dtype=torch.bfloat16, device=device)
    states = []
    for ids, replacement, _ in prompts:
        x = embedding[torch.tensor(ids, device=device)] * embedding_scale
        if replacement is not None:
            start, soft = replacement
            x[start:start + len(soft)] = soft
        states.append(x)

    maximum = max(map(len, states))
    positions = torch.arange(maximum, device=device)
    distance = positions[:, None] - positions[None, :]
    window = config["sliding_window"] if activation else 256
    full_mask = distance >= 0
    masks = [full_mask & (distance < window), full_mask]
    heads = config["num_attention_heads"]
    config_by_type = [(config["head_dim"], config["num_key_value_heads"], "sliding_attention"),
                      (config["global_head_dim"], config["num_global_key_value_heads"], "full_attention")]
    ropes = [rope(maximum, dim, config["rope_parameters"][kind]["rope_theta"],
                  config["rope_parameters"][kind].get("partial_rotary_factor", 1.), device)
             for dim, _, kind in config_by_type]

    # Q4 uses Gate/Up statistics only through layer 45.
    layers = config["layer_types"][:46] if mode == "q4" else config["layer_types"]
    for layer, kind in enumerate(layers):
        full = kind == "full_attention"
        head_dim, kv_heads, _ = config_by_type[full]
        cos, sin = ropes[full]
        prefix = f"{PREFIX}layers.{layer}."

        def param(name):
            return checkpoint.get_tensor(prefix + name).to(device, torch.float32)

        weights = {}
        input_scales = {}
        for site, names in SITES.items():
            if not activation:
                input_scale = max(maxima[f"L{layer}.{names[0]}"].max().item() / 256, 1e-30)
            for name in names:
                if full and name == "v_proj":
                    continue
                attention = site in ("qkv", "o")
                projection = prefix + ("self_attn." if attention else "mlp.") + name
                if activation:
                    weight = checkpoint.get_tensor(projection + ".weight").to(device)
                    weights[name] = bf16_carrier(weight, attention)
                else:
                    if cache is not None and projection in cache:
                        codes, scale = cache[projection]
                    else:
                        codes, scale = fp8_carrier(checkpoint, projection, attention, device)
                        if cache is not None:
                            cache[projection] = codes, scale
                    weights[name] = codes.float(), scale
                    input_scales[name] = input_scale

        def linear(name, x):
            w, scale = weights[name]
            if activation:
                return ((x.float() @ w.T) * scale).bfloat16()
            sx = input_scales[name]
            codes = (x.float() / sx).clamp(-448, 448).to(E4M3).float()
            return (((codes @ w.T) * sx) * scale).bfloat16()

        input_norm = param("input_layernorm.weight")
        q_norm, k_norm = (param(f"self_attn.{name}_norm.weight") for name in ("q", "k"))
        post_attn, pre_ffn, post_ffn = (param(name + "_layernorm.weight") for name in
                                        ("post_attention", "pre_feedforward", "post_feedforward"))
        scalar_name = prefix + "layer_scalar"
        scalar = checkpoint.get_tensor(scalar_name).to(device) if scalar_name in checkpoint.keys() else 1.
        for column, x in enumerate(states):
            n = len(x)
            sample = prompts[column][2]
            y = rms(x, input_norm)
            if activation:
                observe(layer, "q_proj", y)
            q = linear("q_proj", y).reshape(n, heads, head_dim)
            k = linear("k_proj", y).reshape(n, kv_heads, head_dim)
            v = k if full else linear("v_proj", y).reshape_as(k)
            q = rotate(rms(q, q_norm), cos[:n], sin[:n]).transpose(0, 1)
            k = rotate(rms(k, k_norm), cos[:n], sin[:n]).transpose(0, 1)
            v = rms(v).transpose(0, 1)
            if not activation and not full:
                k, v = (value.clamp(-448, 448).to(E4M3).bfloat16() for value in (k, v))
            k, v = (value.repeat_interleave(heads // kv_heads, dim=0) for value in (k, v))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=masks[full][:n, :n], scale=1.)
            y = y.transpose(0, 1).reshape(n, -1)
            if activation:
                observe(layer, "o_proj", y)
            x = x + rms(linear("o_proj", y), post_attn)
            y = rms(x, pre_ffn)
            if observe is not None:
                observe(layer, "gate_proj", y if sample is None else y[sample])
            g, u = linear("gate_proj", y), linear("up_proj", y)
            if activation:
                y = F.gelu(g, approximate="tanh") * u
                observe(layer, "down_proj", y)
            else:
                gf = g.float()
                y = ((u.float() * ((torch.erf(gf * 0.70710678118) + 1.) * gf)) / 2.).bfloat16()
            states[column] = (x + rms(linear("down_proj", y), post_ffn)) * scalar
        if mode != "continuations":
            print(f"{mode}: layer {layer + 1}/{len(layers)}", flush=True)
    if mode != "q4":
        norm = checkpoint.get_tensor(PREFIX + "norm.weight").to(device, torch.float32)
        return [rms(x, norm) for x in states]


def greedy(checkpoint, config, windows, maxima, suppressed, device, steps=17):
    embedding = checkpoint.get_tensor(PREFIX + "embed_tokens.weight").to(device)
    head = embedding.float()
    cache = {} if embedding.is_cuda else None
    softcap = config["final_logit_softcapping"]
    sequences = [list(ids) for ids in windows]
    for step in range(steps):
        prompts = [(ids, None, None) for ids in sequences]
        rows = teacher(checkpoint, config, prompts, "continuations", embedding, maxima=maxima, cache=cache)
        last = torch.stack([x[-1] for x in rows]).float()
        logits = softcap * torch.tanh(F.linear(last, head) / softcap)
        logits[:, suppressed] = -torch.inf
        for ids, token in zip(sequences, logits.argmax(-1).tolist(), strict=True):
            ids.append(token)
        print(f"continuations: token {step + 1}/{steps}", flush=True)
    return [ids[len(window):] for ids, window in zip(sequences, windows, strict=True)]
