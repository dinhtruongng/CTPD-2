from datasets import load_dataset

ds = load_dataset(
    "src/prefkd/generated-data_qwen14b/llama/student_all/full"
)  # returns a Dataset or DatasetDict


def trim_padding(batch):
    """
    For each example:
    • keep only the first N elements of *_true_weight,
      where N == len(placeholder_weight)
    """
    batch["rejected_true_weight"] = [
        tw[: len(w)] for tw, w in zip(batch["rejected_true_weight"], batch["rejected_weight"])
    ]
    batch["chosen_true_weight"] = [
        tw[: len(w)] for tw, w in zip(batch["chosen_true_weight"], batch["chosen_weight"])
    ]
    return batch


clean_ds = ds.map(trim_padding, batched=True, num_proc=8)  # num_proc optional

# print(len(clean_ds["train"]["rejected_true_weight"][20]))
# print(len(clean_ds["train"]["rejected_weight"][20]))

# ds = ds.remove_columns(["irrelevant_feature"])
# print("original: \n", ds)

cols_to_drop = ["chosen_weight", "rejected_weight"]
clean_ds["train"] = clean_ds["train"].remove_columns(cols_to_drop)
clean_ds["test"] = clean_ds["test"].remove_columns(cols_to_drop)
# print("drop: \n", ds)

rename_map = {
    "rejected_true_weight": "rejected_weight",
    "chosen_true_weight": "chosen_weight",
}
clean_ds = clean_ds.rename_columns(rename_map)
# print("final: \n", ds)

clean_ds.save_to_disk("data/ultra-feedback/weight14B/student_all/full")
