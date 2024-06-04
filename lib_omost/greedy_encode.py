import torch
import copy
from typing import Callable, NamedTuple


class CLIPTokens(NamedTuple):
    clip_l_tokens: list[int]
    clip_g_tokens: list[int] | None = None

    @property
    def length(self) -> int:
        return len(self.clip_l_tokens)

    def __repr__(self) -> str:
        return f"CLIPTokens(clip_l_tokens({len(self.clip_l_tokens)}), clip_g_tokens={len(self.clip_g_tokens) if self.clip_g_tokens else None})"


class EncoderOutput(NamedTuple):
    cond: torch.Tensor
    pooler: torch.Tensor


TokenizeFunc = Callable[[str], CLIPTokens]
EncodeFunc = Callable[[CLIPTokens], EncoderOutput]


def greedy_partition(items: list[CLIPTokens], max_sum: int) -> list[list[CLIPTokens]]:
    bags: list[list[CLIPTokens]] = []
    current_bag: list[CLIPTokens] = []
    current_sum: int = 0

    for item in items:
        num = item.length
        if current_sum + num > max_sum:
            if current_bag:
                bags.append(current_bag)
            current_bag = [item]
            current_sum = num
        else:
            current_bag.append(item)
            current_sum += num

    if current_bag:
        bags.append(current_bag)

    return bags


def get_77_tokens_in_torch(subprompt_inds, tokenizer):
    # Note that all subprompt are theoretically less than 75 tokens (without bos/eos)
    result = (
        [tokenizer.bos_token_id]
        + subprompt_inds[:75]
        + [tokenizer.eos_token_id]
        + [tokenizer.pad_token_id] * 75
    )
    result = result[:77]
    result = torch.tensor([result]).to(device=device, dtype=torch.int64)
    return result


def merge_with_prefix(bag):
    merged_ids_t1 = copy.deepcopy(prefix_ids_t1)
    merged_ids_t2 = copy.deepcopy(prefix_ids_t2)

    for item in bag:
        merged_ids_t1.extend(item["ids_t1"])
        merged_ids_t2.extend(item["ids_t2"])

    return dict(
        ids_t1=get_77_tokens_in_torch(merged_ids_t1, self.tokenizer),
        ids_t2=get_77_tokens_in_torch(merged_ids_t2, self.tokenizer_2),
    )


def double_encode(pair_of_inds):
    inds = [pair_of_inds["ids_t1"], pair_of_inds["ids_t2"]]
    text_encoders = [self.text_encoder, self.text_encoder_2]

    pooled_prompt_embeds = None
    prompt_embeds_list = []

    for text_input_ids, text_encoder in zip(inds, text_encoders):
        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)

        # Only last pooler_output is needed
        pooled_prompt_embeds = prompt_embeds.pooler_output

        # "2" because SDXL always indexes from the penultimate layer.
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def encode_bag_of_subprompts_greedy(prefixes: list[str], suffixes: list[str]):
    # Begin with tokenizing prefixes

    prefix_length = 0
    prefix_ids_t1 = []
    prefix_ids_t2 = []

    for prefix in prefixes:
        ids_t1 = self.tokenizer(
            prefix, truncation=False, add_special_tokens=False
        ).input_ids
        ids_t2 = self.tokenizer_2(
            prefix, truncation=False, add_special_tokens=False
        ).input_ids
        assert len(ids_t1) == len(ids_t2)
        prefix_length += len(ids_t1)
        prefix_ids_t1 += ids_t1
        prefix_ids_t2 += ids_t2

    # Then tokenizing suffixes

    allowed_suffix_length = 75 - prefix_length
    suffix_targets = []

    for subprompt in suffixes:
        # Note that all subprompt are theoretically less than 75 tokens (without bos/eos)
        # So we can safely just crop it to 75
        ids_t1 = self.tokenizer(
            subprompt, truncation=False, add_special_tokens=False
        ).input_ids[:75]
        ids_t2 = self.tokenizer_2(
            subprompt, truncation=False, add_special_tokens=False
        ).input_ids[:75]
        assert len(ids_t1) == len(ids_t2)
        suffix_targets.append(dict(length=len(ids_t1), ids_t1=ids_t1, ids_t2=ids_t2))

    # Then merge prefix and suffix tokens

    suffix_targets = greedy_partition(suffix_targets, max_sum=allowed_suffix_length)
    targets = [merge_with_prefix(b) for b in suffix_targets]

    # Encode!

    conds, poolers = [], []

    for target in targets:
        cond, pooler = double_encode(target)
        conds.append(cond)
        poolers.append(pooler)

    conds_merged = torch.concat(conds, dim=1)
    poolers_merged = poolers[0]

    return dict(cond=conds_merged, pooler=poolers_merged)
