import pathlib, shutil, os

_DEFAULT = "/home/terence/.cache/huggingface/modules/transformers_modules/nvidia/LocateAnything_hyphen_3B/c32291ca5e996f5a7a485845b4f57a233936bba0"
MOD = pathlib.Path(os.environ.get("MOD_PATH", _DEFAULT))
GU = MOD / "generate_utils.py"
MO = MOD / "modeling_locateanything.py"
shutil.copy(GU, str(GU) + ".bak")
shutil.copy(MO, str(MO) + ".bak")


def replace_func(src: str, name: str, newfunc: str) -> str:
    lines = src.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(f"def {name}("))
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("def ") or lines[j].startswith("class ") or lines[j].startswith("@"):
            end = j
            break
    return "\n".join(lines[:start] + newfunc.split("\n") + lines[end:])


NEW_DECODE = '''def decode_bbox_avg(
    logits,
    probs,
    token_ids,
    keep_k=5,
    start_thresh=0.7,
    end_thresh=0.2,
    generation_mode: str = 'hybrid',
):
    """
    Decode bounding box coordinates using top-k weighted average. Returns
    (token_tensor, score) where score is the mean probability the model assigned
    to the four chosen coordinate tokens (None when not a real coord box).
    """
    coord_start_token_id = token_ids['coord_start_token_id']
    coord_end_token_id = token_ids['coord_end_token_id']
    box_start_token_id = token_ids['box_start_token_id']
    box_end_token_id = token_ids['box_end_token_id']
    none_token_id = token_ids['none_token_id']

    device = logits.device

    box_type = is_valid_box_frame(
        probs,
        token_ids,
        start_thresh=start_thresh,
        end_thresh=end_thresh,
        topk=keep_k
    )
    if box_type == 'empty_box':
        return torch.tensor([
            box_start_token_id,
            none_token_id,
            box_end_token_id,
            token_ids['null_token_id'],
            token_ids['null_token_id'],
            token_ids['null_token_id']
        ], dtype=torch.long, device=probs.device), None
    elif box_type == 'illegal_box':
        return None, None

    pos_probs, pos_ids = torch.topk(probs[1:5], k=keep_k, dim=-1)
    mask = (pos_ids >= coord_start_token_id) & (pos_ids <= coord_end_token_id)
    has_valid = mask.any(dim=-1)
    if not has_valid.all():
        return None, None

    first_valid_idx = mask.long().argmax(dim=-1, keepdim=True)
    first_valid_probs = pos_probs.gather(-1, first_valid_idx).squeeze(-1)
    first_valid_ids = pos_ids.gather(-1, first_valid_idx).squeeze(-1)
    if generation_mode == 'hybrid':
        valid_counts = mask.sum(dim=-1)
        LARGE_NUM, SMALL_NUM = 999999, -999999
        valid_ids_for_max = torch.where(mask, pos_ids, torch.tensor(SMALL_NUM, device=device))
        valid_ids_for_min = torch.where(mask, pos_ids, torch.tensor(LARGE_NUM, device=device))
        valid_max = valid_ids_for_max.max(dim=-1)[0]
        valid_min = valid_ids_for_min.min(dim=-1)[0]
        is_abnormal = (first_valid_probs < 0.9) & (valid_counts > 1) & ((valid_max - valid_min) > 60)
        final_coords = torch.where(is_abnormal, torch.tensor(0, device=pos_ids.device), first_valid_ids)
    elif generation_mode == 'fast':
        final_coords = first_valid_ids

    start_t = torch.tensor([box_start_token_id], dtype=final_coords.dtype, device=device)
    end_t = torch.tensor([box_end_token_id], dtype=final_coords.dtype, device=device)

    score = float(first_valid_probs.mean().item())
    return torch.cat([start_t, final_coords, end_t]), score'''


NEW_SAMPLE = '''def sample_tokens(
    logits: torch.Tensor,
    generated: torch.Tensor,
    token_ids,
    **generate_kwargs,
):
    batch_size, seq_len, vocab_size = logits.shape

    repetition_penalty = generate_kwargs.get('repetition_penalty', 1.0)
    temperature = generate_kwargs.get('temperature', 0)
    top_p = generate_kwargs.get('top_p', None)
    top_k = generate_kwargs.get('top_k', None)

    if repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated, repetition_penalty)

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if seq_len == 1:
        return probs, confidence, x0, None, None

    box_avg = []
    box_score = []
    fallback_box = torch.zeros(1, dtype=x0.dtype, device=x0.device)

    for b in range(batch_size):
        decoded_box, score = decode_bbox_avg(
            logits[b], probs[b], token_ids, keep_k=generate_kwargs.get('keep_k_avg', 4),
            generation_mode=generate_kwargs.get('generation_mode', 'hybrid'),
        )
        if decoded_box is not None:
            box_avg.append(decoded_box)
            box_score.append(score if score is not None else 0.0)
        else:
            out_ref = decode_ref(logits[b], probs[b], token_ids)
            if out_ref is not None:
                box_avg.append(torch.tensor(out_ref, dtype=x0.dtype, device=x0.device))
            else:
                box_avg.append(fallback_box)
            box_score.append(0.0)

    box_avg = torch.stack(box_avg)
    box_score = torch.tensor(box_score, dtype=torch.float, device=x0.device)

    return probs, confidence, x0, box_avg, box_score'''


gu = GU.read_text()
gu = replace_func(gu, "decode_bbox_avg", NEW_DECODE)
gu = replace_func(gu, "sample_tokens", NEW_SAMPLE)
GU.write_text(gu)
print("generate_utils.py patched")

# --- modeling_locateanything.py: targeted unique replacements ---
mo = MO.read_text()

repls = []

repls.append((
    """            probs, confidence, x0, box_avg = sample_tokens(
                next_token_logits, generated, self.token_ids, keep_k=5, **generate_kwargs
            )

            is_box_empty = (box_avg[0] == 0).all()
            new_tokens = x0[0] if is_box_empty else box_avg[0]

            out_pattern = handle_pattern(new_tokens, self.token_ids, generation_mode)
            out_type = out_pattern['type']
            out_token = torch.tensor(out_pattern['tokens'], dtype=x0.dtype, device=x0.device)

            return out_type, out_token""",
    """            probs, confidence, x0, box_avg, box_score = sample_tokens(
                next_token_logits, generated, self.token_ids, keep_k=5, **generate_kwargs
            )

            is_box_empty = (box_avg[0] == 0).all()
            new_tokens = x0[0] if is_box_empty else box_avg[0]

            out_pattern = handle_pattern(new_tokens, self.token_ids, generation_mode)
            out_type = out_pattern['type']
            out_token = torch.tensor(out_pattern['tokens'], dtype=x0.dtype, device=x0.device)

            box_conf = float(box_score[0].item()) if out_type == 'coord_box' else None
            return out_type, out_token, box_conf""",
))

repls.append((
    """            probs, confidence, x0, _ = sample_tokens(
                next_token_logits, generated, self.token_ids, **generate_kwargs
            )""",
    """            probs, confidence, x0, _, _ = sample_tokens(
                next_token_logits, generated, self.token_ids, **generate_kwargs
            )""",
))

repls.append((
    """            if use_mtp:
                out_type, out_token = _sample_token_in_mtp(generated, outputs)
            else:
                out_type, out_token = _sample_token_in_ar(generated, outputs)""",
    """            if use_mtp:
                out_type, out_token, box_conf = _sample_token_in_mtp(generated, outputs)
            else:
                out_type, out_token, tok_conf = _sample_token_in_ar(generated, outputs)

            # Per-box confidence collection (aligned with parsed <box>...</box>).
            if out_type == 'coord_box':
                box_scores.append(box_conf if box_conf is not None else 0.0)
            elif out_type == 'error_box':
                ar_coord_conf.clear()
            elif out_type == 'coord_ar':
                ar_coord_conf.append(tok_conf)
            elif out_type == 'box_end_ar':
                if ar_coord_conf:
                    box_scores.append(sum(ar_coord_conf) / len(ar_coord_conf))
                ar_coord_conf.clear()""",
))

repls.append((
    "        sampling_history = []",
    "        sampling_history = []\n        box_scores = []\n        ar_coord_conf = []",
))

repls.append((
    """            return response[0], sampling_history, out_info

        return response[0]""",
    """            return response[0], sampling_history, out_info, box_scores

        return response[0], box_scores""",
))

# ar return (now the only remaining bare "return out_type, out_token")
repls.append((
    "            return out_type, out_token\n",
    "            return out_type, out_token, float(confidence[0].item())\n",
))

for old, new in repls:
    cnt = mo.count(old)
    assert cnt == 1, f"expected 1 match, got {cnt} for:\n{old[:80]}"
    mo = mo.replace(old, new)

MO.write_text(mo)
print("modeling_locateanything.py patched")
