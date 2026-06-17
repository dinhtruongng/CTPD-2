from typing import Dict, List, Optional


def tokenize_batch_element(
    prompt: str,
    chosen: str,
    rejected: str,
    truncation_mode: str,
    tokenizer,
    max_length: int,
    max_prompt_length: int,
    rejected_weight=None,
    chosen_weight=None,
) -> Dict:
    """Tokenize a single batch element.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
      in case the prompt + chosen or prompt + rejected responses is/are too long. First
      we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
      the sum of the length of the prompt and the chosen/rejected response, with -100 for the
      prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False, return_offsets_mapping=True)
    # len(chosen_tokens['input_ids'])  104
    rejected_tokens = tokenizer(rejected, add_special_tokens=False, return_offsets_mapping=True)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    if rejected_weight is not None:
        assert len(rejected_weight) == len(rejected_tokens["input_ids"])

    if chosen_weight is not None:
        assert len(chosen_weight) == len(chosen_tokens["input_ids"])

    assert tokenizer.eos_token_id not in prompt_tokens["input_ids"], (
        f"Prompt contains EOS token: {prompt}"
    )
    assert tokenizer.eos_token_id not in chosen_tokens["input_ids"], (
        f"Chosen response contains EOS token: {chosen}"
    )
    assert tokenizer.eos_token_id not in rejected_tokens["input_ids"], (
        f"Rejected response contains EOS token: {rejected}"
    )

    chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
    chosen_tokens["attention_mask"].append(1)

    rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
    rejected_tokens["attention_mask"].append(1)

    longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        if truncation_mode == "keep_start":
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == "keep_end":
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    # if that's still too long, truncate the response
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        # print('truncate=====', len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))
        chosen_tokens = {k: v[: max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {
            k: v[: max_length - max_prompt_length] for k, v in rejected_tokens.items()
        }

    # Create labels
    chosen_sequence_tokens = {
        k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens if k != "offset_mapping"
    }
    rejected_sequence_tokens = {
        k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens if k != "offset_mapping"
    }
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
        prompt_tokens["input_ids"]
    )
    chosen_sequence_tokens["offset_mapping"] = chosen_tokens["offset_mapping"]
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
        prompt_tokens["input_ids"]
    )
    rejected_sequence_tokens["offset_mapping"] = rejected_tokens["offset_mapping"]

    batch = {}

    if rejected_weight is not None:
        batch["rejected_weight"] = (
            [0] * len(prompt_tokens["input_ids"])
            + rejected_weight[: len(rejected_tokens["input_ids"]) - 1]
            + [0]
        )
    else:
        batch["rejected_weight"] = (
            [0] * len(prompt_tokens["input_ids"])
            + [1] * (len(rejected_tokens["input_ids"]) - 1)
            + [0]
        )

    if chosen_weight is not None:
        batch["chosen_weight"] = (
            [0] * len(prompt_tokens["input_ids"])
            + chosen_weight[: len(chosen_tokens["input_ids"]) - 1]
            + [0]
        )
    else:
        batch["chosen_weight"] = (
            [0] * len(prompt_tokens["input_ids"])
            + [1] * (len(chosen_tokens["input_ids"]) - 1)
            + [0]
        )

    assert len(batch["chosen_weight"]) == len(chosen_sequence_tokens["labels"])
    assert len(batch["rejected_weight"]) == len(rejected_sequence_tokens["labels"])

    batch["prompt"] = prompt
    batch["chosen"] = prompt + " " + chosen
    batch["rejected"] = prompt + " " + rejected
    batch["chosen_response_only"] = chosen
    batch["rejected_response_only"] = rejected

    for k, toks in {
        "chosen": chosen_sequence_tokens,
        "rejected": rejected_sequence_tokens,
        "prompt": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}_{type_key}"] = tokens

    return batch


def tokenize_batch(
    prompts: List[str],
    chosens: List[str],
    rejecteds: List[str],
    truncation_mode: str,
    tokenizer,
    max_length: int,
    max_prompt_length: int,
    rejected_weights: Optional[List[List[float]]] = None,
    chosen_weights: Optional[List[List[float]]] = None,
) -> Dict:
    """Tokenize a batch of elements.

    Process multiple examples at once for use with datasets.map(batch=True).
    Handles truncation if prompt + chosen or prompt + rejected responses are too long.
    Creates labels for chosen/rejected responses with -100 for prompt tokens.

    Args:
        prompts: List of prompt strings
        chosens: List of chosen response strings
        rejecteds: List of rejected response strings
        truncation_mode: How to truncate prompts ("keep_start" or "keep_end")
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length after truncation
        rejected_weights: Optional list of weight lists for rejected responses
        chosen_weights: Optional list of weight lists for chosen responses

    Returns:
        Dictionary with batch data
    """
    batch_size = len(prompts)
    results = {
        "chosen": [],
        "rejected": [],
    }

    # Initialize result dictionaries for all token types
    for prefix in ["chosen", "rejected", "prompt"]:
        for suffix in ["input_ids", "attention_mask", "labels", "offset_mapping"]:
            if not (
                (prefix == "prompt" and suffix == "labels")
                or (prefix == "prompt" and suffix == "offset_mapping")
            ):  # prompt doesn't have labels
                results[f"{prefix}_{suffix}"] = []

    results["chosen_weight"] = []
    results["rejected_weight"] = []

    # Process each example in the batch
    for i in range(batch_size):
        prompt = prompts[i]
        chosen = chosens[i]
        rejected = rejecteds[i]
        chosen_weight = None if chosen_weights is None else chosen_weights[i]
        rejected_weight = None if rejected_weights is None else rejected_weights[i]

        # Get single element result
        element_result = tokenize_batch_element(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            truncation_mode=truncation_mode,
            tokenizer=tokenizer,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            rejected_weight=rejected_weight,
            chosen_weight=chosen_weight,
        )

        # Append full text results
        results["chosen"].append(element_result["chosen"])
        results["rejected"].append(element_result["rejected"])

        # Append tokenized results
        for key in element_result:
            if key not in [
                "prompt",
                "chosen",
                "rejected",
                "chosen_response_only",
                "rejected_response_only",
            ]:
                results[key].append(element_result[key])

    return results
