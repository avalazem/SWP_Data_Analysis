import pandas as pd
from nilearn.image import mean_img
from nilearn.glm.first_level import FirstLevelModel
import numpy as np


def setup_paths_and_load_data(subject_id, run_id, project_root_dir):
    
    """ Defines paths, validates them, and loads event data. """
    print(f"Loading data for subject {subject_id}, run {run_id}...")
    subject_id_padded = f"{subject_id:02d}"
    subject_id_for_events = str(subject_id)

    # BIDS-like paths for derivatives
    bids_derivatives_dir = project_root_dir / "single-word-processing" / "data" / "derivatives"
    if not bids_derivatives_dir.exists():
        alt_bids_derivatives_dir = project_root_dir.parent / "data" / "derivatives"
        if alt_bids_derivatives_dir.exists():
            bids_derivatives_dir = alt_bids_derivatives_dir
        else:
            # Fallback to a common relative path if primary structures not found
            bids_derivatives_dir = project_root_dir / ".." / "data" / "derivatives"
            print(f"Warning: Primary derivatives directory not found. Trying relative path: {bids_derivatives_dir}")


    anat_file = bids_derivatives_dir / f"sub-{subject_id_padded}/ses-1/anat/sub-{subject_id_padded}_ses-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    # Ensure func_files is a list, even if it's just one file per run for FirstLevelModel
    func_files = [bids_derivatives_dir / f"sub-{subject_id_padded}/ses-1/func/sub-{subject_id_padded}_ses-1_task-swp_dir-pa_run-{run_id:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"]
    
    events_dir = project_root_dir / "event_tsvs"
    if not events_dir.exists(): # Fallback
        events_dir = project_root_dir / ".." / "event_tsvs"
        print(f"Warning: Primary events directory not found. Trying relative path: {events_dir}")
    events_file = events_dir / f"sub_{subject_id_padded}_run_{run_id:02d}_events.tsv" # Use subject_id_padded and padded run_id

    # Validate paths
    if not anat_file.exists(): print(f"ERROR: Anat file not found: {anat_file}"); return None
    if not func_files[0].exists(): print(f"ERROR: Func file not found: {func_files[0]}"); return None
    if not events_file.exists(): print(f"ERROR: Events file not found: {events_file}"); return None

    print(f"  Anatomical image: {anat_file}")
    print(f"  Functional image(s): {func_files[0]}")
    print(f"  Events file: {events_file}")

    events_df = pd.read_table(events_file)
    
    mean_func_img = mean_img(func_files[0], copy_header=True)
    print("  Mean functional image calculated.")
    
    return {
        "anat_file": anat_file,
        "func_files": func_files,
        "events_file": events_file,
        "events_df": events_df,
        "mean_func_img": mean_func_img
    }

    
def fit_glm_model(func_files, events_df, glm_params):
    
    """Fits the GLM and returns the model and design matrix."""
    print("Fitting GLM model...")
    fmri_glm = FirstLevelModel(**glm_params)
    fmri_glm = fmri_glm.fit(func_files, events_df) # func_files should be a list
    design_matrix = fmri_glm.design_matrices_[0]
    print("  GLM fitting complete. Design matrix extracted.")
    return fmri_glm, design_matrix

    
def load_contrast_vector(contrast_name, design_matrix):
    """
    Generates a contrast vector for the given contrast_name using the design_matrix.
    contrast_name: str, e.g., "condition1>condition2"
    design_matrix: pd.DataFrame
    """
    parts = contrast_name.split(">")
    if len(parts) != 2:
        raise ValueError(f"Contrast name '{contrast_name}' should be in 'positive_key>negative_key' format.")
    
    positive_key = parts[0].strip()
    negative_key = parts[1].strip()

    n_regressors = design_matrix.shape[1]
    contrast_vector = np.zeros(n_regressors)
    conditions = design_matrix.columns
    
    found_pos = False
    found_neg = False

    for idx, col_name in enumerate(conditions):
        # Using 'startswith' for flexibility if full regressor names are complex (e.g. 'visual_real_long...')
        if col_name.startswith(positive_key):
            contrast_vector[idx] = 1
            found_pos = True
        elif col_name.startswith(negative_key):
            contrast_vector[idx] = -1
            found_neg = True
            
    if not found_pos:
        print(f"Warning: Positive key '{positive_key}' not found as a prefix in design matrix columns for contrast '{contrast_name}'.")
    if not found_neg:
        print(f"Warning: Negative key '{negative_key}' not found as a prefix in design matrix columns for contrast '{contrast_name}'.")
    
    if not found_pos and not found_neg:
        print(f"Error: Neither positive ('{positive_key}') nor negative ('{negative_key}') keys found. Contrast vector is all zeros.")
        # Optionally raise an error: raise ValueError("Contrast keys not found in design matrix.")
    elif np.sum(np.abs(contrast_vector)) == 0:
        print(f"Warning: Contrast vector for '{contrast_name}' is all zeros despite potential key matches. Check keys and design matrix.")
    else:
        print(f"Successfully generated contrast vector for '{contrast_name}'")
    return contrast_vector