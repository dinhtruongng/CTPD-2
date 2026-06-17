from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils.utils import drop_leading_zeros_batch, drop_zr_cols_and_padded, pad_to_length


def tisdpo_KDAlign_loss():
    pass


def token_level_dpo_loss(
    ptoken_chosen_logps_margin,  # torch.FloatTensor,  # shape: (batch_size, num_parent_tokens)
    ptoken_rejected_logps_margin,  # torch.FloatTensor,  # shape: (batch_size, num_parent_tokens)
    beta: float = 0.5,
):  # -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    chosen_values = ptoken_chosen_logps_margin
    rejected_values = ptoken_rejected_logps_margin

    chosen_rejected_logps_margin = ptoken_chosen_logps_margin - ptoken_rejected_logps_margin
    logits = chosen_rejected_logps_margin

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps_tisdpo(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    weights: torch.FloatTensor = None,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        weights: Weights for each token. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per
        (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        token_level: If True, return the log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    if labels.dtype != torch.long:
        labels = labels.long()

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = labels != -100

    labels[labels == -100] = 0

    vocab_ps = logits.softmax(-1)
    vocab_logps = vocab_ps.log()

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (vocab_ps * (vocab_logps - reference_vocab_logps)).sum(-1)

    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(
        reference_vocab_logps, dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps
    weights = weights[:, 1:].clone()

    if average_log_prob:
        return (
            (logps_margin * weights * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_position_kl * weights * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_token_logps * weights * loss_mask).sum(-1) / loss_mask.sum(-1),
        )
    else:
        return (
            (logps_margin * weights * loss_mask).sum(-1),
            (per_position_kl * weights * loss_mask).sum(-1),
            (per_token_logps * weights * loss_mask).sum(-1),
        )


def tisdpo_loss(
    chosen_logps_margin: torch.FloatTensor,
    rejected_logps_margin: torch.FloatTensor,
    chosen_position_kl: torch.FloatTensor,
    rejected_position_kl: torch.FloatTensor,
    beta: float,
    alpha: float = 0.5,
    token_level: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    if token_level:
        chosen_values = chosen_logps_margin - chosen_position_kl
        rejected_values = rejected_logps_margin - rejected_position_kl
    else:
        chosen_values = chosen_logps_margin
        rejected_values = rejected_logps_margin

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if token_level:
        logits = chosen_rejected_logps_margin - alpha * (chosen_position_kl - rejected_position_kl)
    else:
        logits = chosen_rejected_logps_margin

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards


def preference_loss(
    policy_chosen_logps,  # torch.FloatTensor,
    policy_rejected_logps,  # torch.FloatTensor,
    reference_chosen_logps,  # torch.FloatTensor,
    reference_rejected_logps,  # torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    ipo: bool = False,
    reference_free: bool = False,
):  # -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1 / (2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def mask_from_neg100(x):
    """
    Replace padding (-100) with 0, but set the last padding token
    before real tokens start to 1.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_tokens).
        pad_val (int): The padding value to replace. Default is -100.

    Returns:
        torch.Tensor: The processed tensor.
    """
    # Find the first real token index
    pad_val = -100
    out = (x != pad_val).long()

    first_real_token_idx = (x != pad_val).float().argmax(dim=1)

    marker_col_idx = (first_real_token_idx - 1).clamp(min=0)

    # Set the last padding (before real tokens) to 1
    row_idx = torch.arange(x.size(0), device=x.device)
    if x.size(1) > 0:
        out[row_idx, marker_col_idx] = 1

    return out


def prompt_remove(logits, labels, input_ids):
    masked_labels = mask_from_neg100(labels)
    masked_logits = masked_labels.unsqueeze(-1) * logits

    no_prompt_logits = drop_zr_cols_and_padded(masked_logits)

    masked_input_ids = masked_labels * input_ids

    drop_zero = drop_leading_zeros_batch(masked_input_ids)

    no_prompt_labels = pad_sequence(drop_zero, batch_first=True, padding_value=0)

    return no_prompt_logits, no_prompt_labels


def gather_with_padding(
    A,  # torch.Tensor,
    index_tensor,  # torch.Tensor,
    padding_value: int = -1,
    fill_value: float = 0.0,
):  # -> torch.Tensor:
    """
    Gathers values from a 2D tensor A based on a 3D index tensor,
    handling padding indices.

    Args:
        A (torch.Tensor): The source data tensor with shape (batch_size, seq_len). Normally should be
        logits from -1 to n, with subsequent 0s
        index_tensor (torch.Tensor): The 3D tensor with shape
            (batch_size, maxdim2, maxdim3) containing indices into the
            last dimension of A for each batch element. -> parent tensor
        padding_value (int): The integer value in index_tensor that represents
                             padding and should be ignored during gathering. Defaults to -1.
        fill_value (float): The value to fill in the output tensor where
                            the index_tensor had the padding_value. Defaults to 0.0.

    Returns:
        torch.Tensor: A tensor with the same shape as index_tensor, where valid
                      indices are replaced by corresponding values from A, and
                      padding indices are replaced by fill_value. The dtype will
                      match the dtype of tensor A.
    """
    # print("original A: ", A.shape)
    # print("original index: ", index_tensor.shape)
    # Ensure index_tensor is Long type for indexing
    if index_tensor.dtype != torch.long:
        index_tensor = index_tensor.long()

    # Get shapes
    batch_size, seq_len = A.shape
    _, maxdim2, maxdim3 = index_tensor.shape  # Or use index_tensor.shape[1] directly

    # 1. Create a mask for padding indices
    padding_mask = index_tensor == padding_value

    # 2. Create safe indices by replacing padding_value with a valid index (e.g., 0)
    #    Clone first to avoid modifying the original index_tensor
    safe_indices = index_tensor.clone()
    safe_indices[padding_mask] = 0  # Replace padding_value with 0 temporarily

    # --- Input Validation (Optional but recommended) ---
    # Check if 0 is actually a valid index if seq_len could be 0
    if seq_len == 0:
        if not torch.all(padding_mask):  # If there are non-padding indices but seq_len is 0
            raise ValueError(
                "Cannot gather from tensor A with seq_len=0 unless all indices are padding."
            )
        # If all indices are padding and seq_len is 0, we can return zeros directly
        return torch.full_like(index_tensor, fill_value=fill_value, dtype=A.dtype)

    # Check if any non-padding index is out of bounds for A's seq_len dimension
    if torch.any((safe_indices >= seq_len) & (~padding_mask)):
        offending_indices = safe_indices[(safe_indices >= seq_len) & (~padding_mask)]
        raise IndexError(
            f"Index out of bounds. Max index allowed: {seq_len - 1}, but found indices: {offending_indices.unique().tolist()}"
        )
    # --- End Input Validation ---

    # 3. Expand A to be compatible with the index tensor's dimensions for gather
    #    Shape: (batch_size, 1, seq_len) -> (batch_size, maxdim2, seq_len)
    A_expanded = A.unsqueeze(1).expand(batch_size, maxdim2, seq_len)

    # 4. Gather values from A_expanded using safe_indices
    #    Gather along dimension 2 (the original seq_len dimension)
    #    Output shape will be the same as safe_indices: (batch_size, maxdim2, maxdim3)
    gathered_values = torch.gather(A_expanded, 2, safe_indices)

    # 5. Apply the mask: set elements corresponding to original padding indices to fill_value
    #    Ensure the output tensor has the same dtype as A
    result_tensor = gathered_values.to(A.dtype)  # Match A's dtype
    result_tensor[padding_mask] = fill_value

    return result_tensor


def get_ptoken_logps(
    logits,  # torch.FloatTensor,  # shape: (batch_size, num_tokens, vocab_size) # already normalized (-1, n)
    labels,  # torch.LongTensor,  # shape: (batch_size, num_tokens) # only the tokens in the response (-1, n)
    p_tokens_list,  # torch.LongTensor,  # batch_size * num_parent_tokens *  # data: offset of ptoken: index of token in seq len
    average_log_prob=0,
):  # -> torch.Tensor:  # shape: batch_size * max_parent_tokens
    assert logits.shape[:-1] == labels.shape, f"logits: {logits.shape}, labels: {labels.shape}"

    # mask?
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    # loss_mask = labels == -100

    labels[labels == -100] = 0

    per_token_prob = torch.gather(logits, dim=2, index=labels.unsqueeze(2).long()).squeeze(2)

    p_token_with_prob = gather_with_padding(
        per_token_prob, p_tokens_list, padding_value=-1, fill_value=0.0
    )
    if average_log_prob == 1:
        result = torch.mean(p_token_with_prob, dim=-1)
    else:
        result = torch.sum(p_token_with_prob, dim=-1)

    return result


def _get_batch_logps(
    logits,  # torch.FloatTensor,
    labels,  # torch.LongTensor,
    weights,  # torch.FloatTensor = None,
    average_log_prob: bool = False,
    token_level: bool = False,
):  # -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2).long()
    ).squeeze(2)
    # import ipdb; ipdb.set_trace()
    if token_level:
        weights = weights[:, 1:].clone()
        batch_logps = (per_token_logps * loss_mask * weights).sum(-1)
    else:
        batch_logps = (per_token_logps * loss_mask).sum(-1)

    if average_log_prob:
        return batch_logps / loss_mask.sum(-1)
    else:
        return batch_logps


def _get_batch_logps_KDtisdpo(
    parent_token_logps,  # torch.HalfTensor,  # shape: (batch_size, num_parent_tokens)
    reference_parent_token_logps,  # torch.HalfTensor,  # shape: (batch_size, num_parent_tokens)
    weights,  # torch.FloatTensor = None,  # shape: (batch_size, num_parent_tokens)
    # average_log_prob: bool = False
):  # -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        weights: Weights for each token. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per
        (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        token_level: If True, return the log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    logps_margin = ((parent_token_logps - reference_parent_token_logps) * weights).sum(
        -1
    )  # shape: (batch_size, )
    logps = (parent_token_logps * weights).sum(-1)  # shape: (batch_size, )

    return logps_margin, logps


def fast_pad_tensor(input_tensor, max_token, max_span, pad_value=-1):
    batch_size, token_size, span_size = input_tensor.shape

    # Create the output tensor filled with pad_value
    output = input_tensor.new_full((batch_size, max_token, max_span), pad_value)

    # Copy the original values into the top-left part
    output[:, :token_size, :span_size] = input_tensor

    return output


def concatenated_inputs(batch: Dict, mode: str) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(
        batch[f"chosen_{mode}_input_ids"].shape[1], batch[f"rejected_{mode}_input_ids"].shape[1]
    )
    max_num_parents = max(
        batch[f"chosen_{mode}_parent_list"].shape[1], batch[f"rejected_{mode}_parent_list"].shape[1]
    )
    # print("max_num_parent: ", max_num_parents)
    max_span = max(
        batch[f"chosen_{mode}_parent_list"].shape[2], batch[f"rejected_{mode}_parent_list"].shape[2]
    )
    concatenated_batch = {}
    keys = [k for k in batch if mode in k]
    keys.extend([k for k in batch if "weight" in k])
    for k in keys:
        # if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
        if k.startswith("chosen"):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            if "weight" in k:
                # print(k)
                # print(concatenated_key)
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_num_parents, pad_value=pad_value
                )
            elif "parent_list" in k:
                concatenated_batch[concatenated_key] = fast_pad_tensor(
                    batch[k], max_num_parents, max_span, pad_value=-1
                )
            elif ("parent_dict" in k) or ("offset_mapping" in k):
                concatenated_batch[concatenated_key] = batch[k]
            else:
                # print(k)
                # print(type(batch[k]))
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )
    for k in keys:
        # if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
        if k.startswith("rejected"):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("rejected", "concatenated")
            if "weight" in k:
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_num_parents, pad_value=pad_value),
                    ),
                    dim=0,
                )
            elif "parent_list" in k:
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        fast_pad_tensor(batch[k], max_num_parents, max_span, pad_value=-1),
                    ),
                    dim=0,
                )
            elif ("parent_dict" in k) or ("offset_mapping" in k):
                concatenated_batch[concatenated_key] += batch[k]
            else:
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                )
    return concatenated_batch


def concatenated_tisdpo_inputs(batch: Dict, mode: str) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(
        batch[f"chosen_{mode}_input_ids"].shape[1], batch[f"rejected_{mode}_input_ids"].shape[1]
    )
    max_num_parents = max(
        batch[f"chosen_{mode}_parent_list"].shape[1], batch[f"rejected_{mode}_parent_list"].shape[1]
    )

    max_weight = max(batch["chosen_weight"].shape[1], batch["rejected_weight"].shape[1])
    print("max_weight: ", max_weight)
    print("chosen shape: ", batch["chosen_weight"].shape)
    print("rejected shape: ", batch["rejected_weight"].shape)
    # print("max_num_parent: ", max_num_parents)
    max_span = max(
        batch[f"chosen_{mode}_parent_list"].shape[2], batch[f"rejected_{mode}_parent_list"].shape[2]
    )
    concatenated_batch = {}
    keys = [k for k in batch if mode in k]
    keys.extend([k for k in batch if "weight" in k])
    for k in keys:
        # if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
        if k.startswith("chosen"):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            if "weight" in k:
                # print(k)
                # print(concatenated_key)
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_weight, pad_value=pad_value
                )
                print("after chosen shape ", concatenated_batch["concatenated_weight"].shape)
            elif "parent_list" in k:
                concatenated_batch[concatenated_key] = fast_pad_tensor(
                    batch[k], max_num_parents, max_span, pad_value=-1
                )
            elif ("parent_dict" in k) or ("offset_mapping" in k):
                concatenated_batch[concatenated_key] = batch[k]
            else:
                # print(k)
                # print(type(batch[k]))
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )
    for k in keys:
        # if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
        if k.startswith("rejected"):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("rejected", "concatenated")
            if "weight" in k:
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_weight, pad_value=pad_value),
                    ),
                    dim=0,
                )
            elif "parent_list" in k:
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        fast_pad_tensor(batch[k], max_num_parents, max_span, pad_value=-1),
                    ),
                    dim=0,
                )
            elif ("parent_dict" in k) or ("offset_mapping" in k):
                concatenated_batch[concatenated_key] += batch[k]
            else:
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                )
    return concatenated_batch
