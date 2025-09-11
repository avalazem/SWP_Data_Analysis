import numpy as np
import json

def create_contrast_vector(contrast_rule):
    """
    Creates a contrast vector (48-dim) based on a given contrast rule.
    It assigns 1s and -1s and then scales the values only if the vector
    does not sum to zero to ensure a zero sum.
    """
    vector = np.zeros(48)
    
    group_indices = {
    'audio': list(range(0, 24)),
    'visual': list(range(24, 48)),
    'speech': list(range(0, 12)) + list(range(24, 36)),
    'write': list(range(12, 24)) + list(range(36, 48)),
    'pseudo': [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39],
    'real': [4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47],
    'long': [0, 1, 4, 5, 6, 7, 12, 13, 16, 17, 18, 19, 24, 25, 28, 29, 30, 31, 36, 37, 40, 41, 42, 43],
    'short': [2, 3, 8, 9, 10, 11, 14, 15, 20, 21, 22, 23, 26, 27, 32, 33, 34, 35, 38, 39, 44, 45, 46, 47],
    'complex': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46],
    'simple': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47],
    'high': [4, 5, 8, 9, 16, 17, 20, 21, 28, 29, 32, 33, 40, 41, 44, 45],
    'low': [6, 7, 10, 11, 18, 19, 22, 23, 30, 31, 34, 35, 42, 43, 46, 47]
}

    parts = contrast_rule.split('|')
    main_contrast_str = parts[0].strip()
    filters = [f.strip() for f in parts[1:]]
    pos_group, neg_group = main_contrast_str.split('>')
    pos_group = pos_group.strip()
    neg_group = neg_group.strip()

    base_indices = set(range(48))
    for f in filters:
        if f in group_indices:
            base_indices = base_indices.intersection(set(group_indices[f]))
    
    pos_indices = sorted(list(set(group_indices.get(pos_group, [])).intersection(base_indices)))
    neg_indices = sorted(list(set(group_indices.get(neg_group, [])).intersection(base_indices)))
    
    for idx in pos_indices:
        vector[idx] = 1
    for idx in neg_indices:
        vector[idx] = -1

    if np.isclose(np.sum(vector), 0):
        return vector
    else:
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("One or both groups are empty after filtering.")
        
        pos_weight = 1 / n_pos
        neg_weight = 1 / n_neg
        
        scaled_vector = np.zeros(48)
        for idx in pos_indices:
            scaled_vector[idx] = pos_weight
        for idx in neg_indices:
            scaled_vector[idx] = -neg_weight
        
        return scaled_vector


def create_contrast_dict(contrast_rule, task="swp"):
    """
    Generates a dictionary for a contrast rule, including the weighted vector.

    Args:
        contrast_rule (str): The contrast rule, e.g., 'audio > visual'.
        task (str, optional): The name of the task. Defaults to "swp".

    Returns:
        dict: A dictionary containing the contrast definition.
    """
    try:
        weights_vector = create_contrast_vector(contrast_rule)
    except ValueError as e:
        print(f"Skipping rule '{contrast_rule}' due to error: {e}")
        return {}
    
    # Use the contrast_rule directly as the dictionary key
    dict_key = contrast_rule
    
    # Set the name field to be the same as the contrast_rule
    name = contrast_rule
    
    parts = contrast_rule.split('|')
    main_contrast_str = parts[0].strip()
    pos_group, neg_group = main_contrast_str.split('>')
    
    description = f"Contrast comparing {pos_group.strip()} and {neg_group.strip()} conditions"
    
    contrast_dict = {
        dict_key: {
            "task": task,
            "name": name,
            "weights": weights_vector.tolist(),
            "description": description
        }
    }
    return contrast_dict

def generate_contrast_json_object(contrast_rules, task="swp"):
    """
    Generates a single Python dictionary from a list of contrast rules.
    """
    all_contrasts = {}
    for rule in contrast_rules:
        single_contrast = create_contrast_dict(rule, task)
        all_contrasts.update(single_contrast)
    return all_contrasts

# Define a list of example contrast rules
example_rules = [
    # audio > visual
    "audio > visual",
    "audio > visual | speech",
    "audio > visual | write",
    
    # speech > write
    "speech > write",
    "speech > write | audio",
    "speech > write | visual",
    
    # long > short
    "long > short",
    "long > short | audio",
    "long > short | visual",
    "long > short | speech",
    "long > short | write",
    "long > short | audio | speech",
    "long > short | audio | write",
    "long > short | visual | speech",
    "long > short | visual | write",
    
    # real > pseudo
    "real > pseudo",
    "real > pseudo | audio",
    "real > pseudo | visual",
    "real > pseudo | speech",
    "real > pseudo | write",
    "real > pseudo | audio | speech",
    "real > pseudo | audio | write",
    "real > pseudo | visual | speech",
    "real > pseudo | visual | write",
    
    # complex > simple
    "complex > simple",
    "complex > simple | audio",
    "complex > simple | visual",
    "complex > simple | speech",
    "complex > simple | write",
    "complex > simple | audio | speech",
    "complex > simple | audio | write",
    "complex > simple | visual | speech",
    "complex > simple | visual | write",
    
    # high > low | real
    "high > low | real | audio",
    "high > low | real | visual",
    "high > low | real | audio | speech",
    "high > low | real | audio | write",
    "high > low | real | visual | speech",
    "high > low | real | visual | write",
]

# 1. Generate the Python dictionary
contrasts_data = generate_contrast_json_object(example_rules)

# 2. Save the dictionary as a JSON file, with a compact format for the weights
file_path = "contrasts.json"
with open(file_path, "w") as json_file:
    # Use json.dumps to get a string representation with custom formatting
    formatted_json_string = json.dumps(contrasts_data, indent=4)
    
    # Replace the default list formatting with a compact single-line version
    for key, value in contrasts_data.items():
        weights_list_str = str(value['weights']).replace(' ', '')
        formatted_json_string = formatted_json_string.replace(json.dumps(value['weights']), weights_list_str)

    json_file.write(formatted_json_string)

print(f"✅ Successfully saved {len(contrasts_data)} contrasts to '{file_path}'.")