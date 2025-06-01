import numpy as np
import pandas as pd

# CONDITIONS dictionary
CONDITIONS = {
   "input_modality": ["visual", "audio"],
    "output_modality": ["speech", "write"],
    "lexicality": ["pseudo", "real"],
    "length": ["short", "long"],
    "frequency": ["low", "high"],
    "morph_complexity": ["simple", "complex"]
}

# Define the expected order of factors in regressor names
# and special handling for frequency based on lexicality
_ORDERED_FACTOR_NAMES = [
    "input_modality", "output_modality", "lexicality", 
    "length", "frequency", "morph_complexity"
]
_FACTOR_LEXICALITY = "lexicality"
_VALUE_PSEUDO = "pseudo"
_FACTOR_FREQUENCY = "frequency"
# Task regressors are expected to start with one of the input modalities
_POSSIBLE_TASK_STARTERS = CONDITIONS["input_modality"]


def _parse_key_to_features(key_str: str) -> dict:
    """
    Parses a contrast key string (e.g., "audio_high") into a dictionary of features.
    Example: "audio_high" -> {"input_modality": "audio", "frequency": "high"}
    """
    target_features = {}
    if not key_str:
        return target_features
    
    key_parts = key_str.split('_')
    for part in key_parts:
        found_category_for_part = False
        for category, allowed_values in CONDITIONS.items():
            if part in allowed_values:
                if category in target_features and target_features[category] != part:
                    print(f"Warning: Ambiguous or conflicting part '{part}' in key '{key_str}'. "
                          f"Category '{category}' was '{target_features[category]}', attempted to set to '{part}'. "
                          "Please ensure keys define a consistent set of features. Behavior may be unpredictable.")
                target_features[category] = part
                found_category_for_part = True
                break # Part assigned to a category
        if not found_category_for_part:
            print(f"Warning: Part '{part}' in key '{key_str}' not found in any CONDITIONS category. "
                  "This part will be ignored for matching.")
    return target_features


def _parse_regressor_to_features(col_name: str) -> dict | None:
    """
    Parses a design matrix column name into its constituent features if it's a task regressor.
    Returns None if it's considered a confound or doesn't match the expected structure.
    Example: "audio_speech_pseudo_long_complex_run1" -> 
             {'input_modality': 'audio', 'output_modality': 'speech', ...}
    """
    if not any(col_name.startswith(starter) for starter in _POSSIBLE_TASK_STARTERS):
        return None # Likely a confound or non-task regressor

    parts = col_name.split('_')
    features = {}
    current_part_idx = 0

    for factor_name in _ORDERED_FACTOR_NAMES:
        if factor_name == _FACTOR_FREQUENCY and \
           features.get(_FACTOR_LEXICALITY) == _VALUE_PSEUDO:
            continue  # Skip frequency for pseudo words

        if current_part_idx >= len(parts):
            return None # Not enough parts in col_name for expected structure

        part_value = parts[current_part_idx]

        if part_value not in CONDITIONS.get(factor_name, []):
            # If the part_value is not in the list of allowed values for the current factor,
            # it means we've likely hit a suffix (e.g., 'run1') or it's not a valid task regressor.
            # We assume that if we successfully parsed the minimum number of factors (5 or 6),
            # any subsequent parts are suffixes.
            num_parsed_factors = len(features)
            is_pseudo = features.get(_FACTOR_LEXICALITY) == _VALUE_PSEUDO
            min_expected_condition_parts = 5 if is_pseudo else 6
            
            if num_parsed_factors >= min_expected_condition_parts:
                break # Stop parsing, assume remaining parts are suffixes
            else:
                return None # Not a valid task regressor structure before suffixes

        features[factor_name] = part_value
        current_part_idx += 1
    
    # Final check: ensure minimum number of features were parsed for it to be a valid condition
    is_pseudo_final_check = features.get(_FACTOR_LEXICALITY) == _VALUE_PSEUDO
    expected_min_factors = 5 if is_pseudo_final_check else 6
    if len(features) < expected_min_factors:
         # This can happen if a name is too short, e.g. "audio_speech_real"
        return None

    return features


def _check_features_match(regressor_features: dict, target_key_features: dict) -> bool:
    """
    Checks if all features in target_key_features are present and match in regressor_features.
    """
    if not target_key_features: # An empty key (e.g. from an empty negative_key_str) should not match.
        return False
    for target_factor, target_value in target_key_features.items():
        if regressor_features.get(target_factor) != target_value:
            return False
    return True


def load_contrast_vector(contrast_name: str, design_matrix: pd.DataFrame) -> np.ndarray:
    """
    Generates a contrast vector based on feature matching against design_matrix columns.

    Args:
        contrast_name: str, e.g., "audio_real" (positive key vs others)
                       or "audio > visual" (positive key vs negative key).
        design_matrix: pd.DataFrame, where columns are regressor names.

    Returns:
        np.ndarray: The contrast vector.
    """
    positive_key_str = ""
    negative_key_str = "" # Only used if ">" is in contrast_name

    if ">" in contrast_name:
        parts = contrast_name.split(">", 1)
        if len(parts) == 2:
            positive_key_str = parts[0].strip()
            negative_key_str = parts[1].strip()
            if not positive_key_str or not negative_key_str:
                raise ValueError(
                    f"Contrast name '{contrast_name}' has an empty positive or negative key. "
                    "Required format: 'positive_key > negative_key'."
                )
        else: # Should not happen with split(">", 1)
            raise ValueError(
                f"Contrast name '{contrast_name}' format is invalid. "
                "Use 'positive_key' or 'positive_key > negative_key'."
            )
    else:
        positive_key_str = contrast_name.strip()
        if not positive_key_str:
            raise ValueError("Contrast name for single key cannot be empty.")

    positive_target_features = _parse_key_to_features(positive_key_str)
    negative_target_features = _parse_key_to_features(negative_key_str) if negative_key_str else {}

    n_regressors = design_matrix.shape[1]
    contrast_vector = np.zeros(n_regressors)
    
    found_any_positive_match = False
    found_any_negative_match = False

    for idx, col_name in enumerate(design_matrix.columns):
        regressor_features = _parse_regressor_to_features(col_name)

        if regressor_features is None: # Confound or unparseable
            contrast_vector[idx] = 0
            continue

        # Logic for "P > N" contrasts
        if negative_key_str: # Implies "P > N" format
            is_positive_match = _check_features_match(regressor_features, positive_target_features)
            is_negative_match = _check_features_match(regressor_features, negative_target_features)

            if is_positive_match:
                contrast_vector[idx] += 1
                found_any_positive_match = True
            if is_negative_match: # Use 'if' not 'elif' in case a regressor could match aspects of both
                contrast_vector[idx] -= 1
                found_any_negative_match = True
        # Logic for single "P" key (contrast P vs all other task conditions)
        else:
            if _check_features_match(regressor_features, positive_target_features):
                contrast_vector[idx] = 1
                found_any_positive_match = True
            else:
                # It's a task regressor but didn't match the positive key
                contrast_vector[idx] = -1 
                # No specific found_any_negative_match here, as it's an implicit negative

    # Warnings
    if not found_any_positive_match and positive_target_features:
        print(f"Warning: Positive key features from '{positive_key_str}' did not match any task regressors.")
    if negative_key_str and not found_any_negative_match and negative_target_features:
        print(f"Warning: Negative key features from '{negative_key_str}' did not match any task regressors.")

    if np.sum(np.abs(contrast_vector)) == 0 and (positive_target_features or negative_target_features) :
        print(f"Warning: Contrast vector for '{contrast_name}' is all zeros. "
              "Check if keys correctly specify distinct and present conditions, "
              "or if positive/negative keys effectively cancel out for all regressors.")
    elif positive_target_features : # Print success only if a positive key was processed and vector is non-zero or warning issued
        print(f"Generated contrast vector for '{contrast_name}'. Sum: {np.sum(contrast_vector)}, Non-zero elements: {np.count_nonzero(contrast_vector)}")
            
    return contrast_vector
