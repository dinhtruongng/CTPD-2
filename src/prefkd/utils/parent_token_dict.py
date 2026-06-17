import numpy as np

MAX_LEN = 512
MAX_PROMPT_LEN = 128


def find_parent_token(student_offset_map, teacher_offset_map, teacher_token_id_dict=None):
    """
    Find parent tokens between student and teacher tokenizations.

    Parameters:
    student_offset_map (List[Tuple[int, int]]): Offset map for student tokenization
    teacher_offset_map (List[Tuple[int, int]]): Offset map for teacher tokenization
    teacher_token_id_dict (Dict[int, int], optional): Dictionary mapping token indices to token IDs for teacher

    Returns:
    Tuple[Dict, Dict]: Dictionaries mapping parent token spans to lists of child token indices
    """
    a = student_offset_map[-1][-1]
    b = teacher_offset_map[-1][-1]
    n = max(a, b)
    cnt = np.zeros(n + 1)

    # Count occurrences of indices in both offset maps
    for items in student_offset_map:
        cnt[items[0]] += 1
    for items in teacher_offset_map:
        cnt[items[0]] += 1
    cnt[n] = 2

    # Extract parent indices
    parent_index = [0]
    for items in student_offset_map:
        if cnt[items[1]] == 2:
            parent_index.append(int(items[1]))

    # Create parent token spans
    parent = []
    for i in range(len(parent_index) - 1):
        parent.append((parent_index[i], parent_index[i + 1]))

    # Build student parent dictionary
    student_parent_dict = {}
    sind = 0
    for i in range(len(parent)):
        student_parent_dict[parent[i]] = []
        while sind < len(student_offset_map):
            if student_offset_map[sind][1] <= parent[i][1]:
                student_parent_dict[parent[i]].append(sind)
            if student_offset_map[sind][1] == parent[i][1]:
                sind += 1
                break
            sind += 1

    # Build teacher parent dictionary
    teacher_parent_dict = {}
    tind = 0
    for i in range(len(parent)):
        teacher_parent_dict[parent[i]] = []
        while tind < len(teacher_offset_map):
            if teacher_offset_map[tind][1] <= parent[i][1]:
                if teacher_token_id_dict is not None:
                    teacher_parent_dict[parent[i]].append((tind, teacher_token_id_dict[tind]))
                else:
                    teacher_parent_dict[parent[i]].append(tind)
            if teacher_offset_map[tind][1] == parent[i][1]:
                tind += 1
                break
            tind += 1

    return student_parent_dict, teacher_parent_dict


def batch_find_parent_token(examples, mode):
    """
    Process a batch of examples to find parent tokens.
    Designed for use with Hugging Face datasets.map().

    Parameters:
    examples (Dict[str, List]): A dictionary where keys are column names
                                (e.g., 'student_offset_map_column', 'teacher_offset_map_column')
                                and values are lists of data for the batch.

    Returns:
    Dict[str, List]: A dictionary with new columns for student and teacher parent mappings.
    """
    # These will store the results for each example in the batch
    batch_student_parent_dicts = []
    batch_teacher_parent_dicts = []

    # Assuming your dataset columns are named 'student_offsets' and 'teacher_offsets'.
    # Adjust these names if your columns are named differently.
    student_offset_maps_batch = examples[
        f"{mode}_student_offset_mapping"
    ]  # Replace with your actual column name
    teacher_offset_maps_batch = examples[
        f"{mode}_teacher_offset_mapping"
    ]  # Replace with your actual column name

    num_examples = len(student_offset_maps_batch)

    for i in range(num_examples):
        student_map = student_offset_maps_batch[i]
        teacher_map = teacher_offset_maps_batch[i]

        s_parent_dict, t_parent_dict = find_parent_token(
            student_map,
            teacher_map,
        )
        batch_student_parent_dicts.append(s_parent_dict)
        batch_teacher_parent_dicts.append(t_parent_dict)

    # The map function expects a dictionary where keys are new column names
    examples[f"{mode}_student_parent_dict"] = batch_student_parent_dicts
    examples[f"{mode}_teacher_parent_dict"] = batch_teacher_parent_dicts
    return examples


## TO DO: add batch processed version
