import os
import matplotlib.pyplot as plt
from nilearn.image import mean_img
from nilearn.plotting import plot_anat, plot_img, plot_stat_map
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix as nilearn_plot_design_matrix
from pathlib import Path
import numpy as np
from nilearn.plotting import plot_contrast_matrix as nilearn_plot_contrast_matrix
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_glass_brain
import argparse

# Default GLM parameters
GLM_PARAMS = {
    "t_r": 1.81,
    "noise_model": "ar1",
    "standardize": False,
    "hrf_model": "spm",
    "drift_model": "cosine",
    "high_pass": 0.01,
}

# Argument parser setup
parser = argparse.ArgumentParser(description="Process fMRI data for a single subject and run.")
parser.add_argument("--subject", type=int, required=True, help="Subject number (e.g., 1)")
parser.add_argument("--run", type=int, required=False, help="Run number (e.g., 1). If not provided, all runs (1-6) will be processed.")
parser.add_argument("--contrast", type=str, required=True, help="Contrast name (e.g., 'visual>auditory')")
parser.add_argument("--output_dir", type=str, default="results", help="Base output directory for results")
parser.add_argument("--alpha", type=float, nargs='+', default=[0.05], help="Alpha levels for thresholding")
parser.add_argument("--num_runs", type=int, default=6, help="Total number of runs if --run is not specified.")


def setup_paths_and_load_data(subject_id, run_id, project_root_dir):
    """Defines paths, validates them, loads event data, and calculates mean functional image."""
    print(f"Setting up paths and loading data for subject {subject_id}, run {run_id}...")
    subject_id_padded = f"{subject_id:02d}"
    subject_id_for_events = str(subject_id)

    # BIDS-like paths for derivatives
    bids_derivatives_dir = project_root_dir / "single-word-processing" / "data" / "derivatives"
    if not bids_derivatives_dir.exists():
        alt_bids_derivatives_dir = project_root_dir.parent / "single-word-processing" / "data" / "derivatives"
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
    #print("  Events data loaded (first 5 rows):")
    #print(events_df.head())
    
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

# Function to define contrast vector based on contrast name and design matrix
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

def plot_design_matrix_to_file(design_matrix, output_filepath):
    """Plots the design matrix and saves it to a file."""
    print(f"Plotting design matrix to {output_filepath}...")
    nilearn_plot_design_matrix(design_matrix, output_file=output_filepath)
    plt.close() # Close the figure to free memory
    print("  Design matrix plot saved.")

def plot_contrast_matrix_to_file(contrast_vector, design_matrix, output_filepath):
    """Plots the contrast matrix and saves it to a file."""
    print(f"Plotting contrast matrix to {output_filepath}...")
    nilearn_plot_contrast_matrix(contrast_vector, design_matrix, output_file=output_filepath)
    plt.close()
    print("  Contrast matrix plot saved.")

def plot_diagnostic_images_to_file(mean_func_img, anat_file, base_output_filepath_prefix):
    """Plots and saves mean functional and anatomical images."""
    print("Plotting diagnostic images...")
    plot_img(mean_func_img, 
             colorbar=True, 
             cbar_tick_format="%i", 
             cmap="gray",
             output_file=base_output_filepath_prefix.with_suffix(".mean_func_img.png"))
    plot_anat(anat_file, 
              colorbar=True, 
              cbar_tick_format="%i",
              output_file=base_output_filepath_prefix.with_suffix(".anat_img.png"))
    plt.close('all')
    print("  Mean functional and anatomical images saved.")

def compute_threshold_plot_stat_maps_to_file(fmri_glm, contrast_vector, original_contrast_name, contrast_name_safe, mean_func_img, current_alpha, cluster_threshold, base_output_filepath_prefix):
    """Computes statistical maps, thresholds them, and plots/saves them."""
    print("Computing and plotting statistical maps...")
    # Compute z-map
    z_map = fmri_glm.compute_contrast(contrast_vector, output_type="z_score")
    print(f"  Z-map computed for contrast: {contrast_name_safe}")

    # Threshold the z-map
    print(f"  Thresholding z-map with alpha={current_alpha}, cluster_threshold={cluster_threshold}...")
    clean_map, threshold = threshold_stats_img(
        z_map, alpha=current_alpha, 
        height_control="fdr",
        cluster_threshold=cluster_threshold, 
        two_sided=False,
    )
    print(f"  Thresholded map generated. Threshold value: {threshold:.3f}")

    # Plot stat map
    stat_map_plotting_config = {"bg_img": mean_func_img, 
                                "display_mode": "z", 
                                "cut_coords": 3, 
                                "black_bg": True}
    title_stat_map = (f"{original_contrast_name} (p<{current_alpha:.3f} FDR; thresh: {threshold:.3f}; clusters > {cluster_threshold})")
    
    stat_map_filepath = base_output_filepath_prefix.with_suffix(f".stat_map_alpha{current_alpha}.png")
    plot_stat_map(clean_map, threshold=threshold, 
                  title=title_stat_map,
                  figure=plt.figure(figsize=(10, 4)), 
                  output_file=stat_map_filepath,
                  **stat_map_plotting_config)
    print(f"  Statistical map saved to {stat_map_filepath}")

    # Plot glass brain
    glass_brain_plotting_config = {"display_mode": "ortho", 
                                   "cut_coords": (0,0,0), 
                                   "colorbar": True, 
                                   "annotate": True, 
                                   "draw_cross": False, 
                                   "black_bg": False}
    glass_brain_filepath = base_output_filepath_prefix.with_suffix(f".glass_brain_alpha{current_alpha}.png")
    plot_glass_brain(clean_map, threshold=threshold, title=title_stat_map,
                     figure=plt.figure(figsize=(10, 8)), 
                     output_file=glass_brain_filepath,
                     **glass_brain_plotting_config)
    print(f"  Glass brain plot saved to {glass_brain_filepath}")
    plt.close('all')


# Main analysis function for a single subject, run, and contrast
def run_first_level_analysis(subject_id, run_id, contrast_name, output_dir_base, alpha_levels, project_root_dir, glm_params):
    """
    Performs first-level fMRI analysis for a given subject, run, and contrast.
    """
    print(f"--- Starting Analysis for Subject: {subject_id}, Run: {run_id}, Contrast: {contrast_name} ---")

    # 0. Setup output directory and base filename
    output_dir_base_path = Path(output_dir_base)
    # Per-run output directory
    run_specific_output_dir = output_dir_base_path / f"sub-{subject_id:02d}" / f"run-{run_id:02d}"
    run_specific_output_dir.mkdir(exist_ok=True, parents=True)
    
    contrast_name_safe = contrast_name.replace(">", "_vs_") # Used for filenames
    base_fn_name_prefix = f"sub-{subject_id:02d}_run-{run_id:02d}_contrast-{contrast_name_safe}"
    # This is now a prefix for filenames, specific file suffixes will be added by plotting functions
    base_output_filepath_prefix = run_specific_output_dir / base_fn_name_prefix

    # 1. Load Data
    data_loaded = setup_paths_and_load_data(subject_id, run_id, project_root_dir)
    if not data_loaded:
        print(f"ERROR: Data loading failed for sub-{subject_id}, run-{run_id}. Skipping analysis.")
        return

    # 2. Fit GLM
    fmri_glm, design_matrix = fit_glm_model(data_loaded["func_files"], data_loaded["events_df"], glm_params)
    
    # 3. Plot Design Matrix
    dm_plot_path = base_output_filepath_prefix.with_suffix(".design_matrix.png")
    plot_design_matrix_to_file(design_matrix, dm_plot_path)

    # 4. Define and Plot Contrast
    contrast_vector = load_contrast_vector(contrast_name, design_matrix)
    cm_plot_path = base_output_filepath_prefix.with_suffix(".contrast_matrix.png")
    plot_contrast_matrix_to_file(contrast_vector, design_matrix, cm_plot_path)

    # 5. Plot Diagnostic Images (Mean Func, Anat)
    plot_diagnostic_images_to_file(data_loaded["mean_func_img"], data_loaded["anat_file"], base_output_filepath_prefix)

    # 6. Compute, Threshold, and Plot Statistical Maps
    # Using the first alpha for simplicity, can be extended to loop through alpha_levels
    current_alpha = alpha_levels[0] 
    cluster_threshold = 5 # Default, consider making it an argument if it varies
    
    compute_threshold_plot_stat_maps_to_file(
        fmri_glm, contrast_vector, contrast_name, contrast_name_safe, 
        data_loaded["mean_func_img"], current_alpha, cluster_threshold, 
        base_output_filepath_prefix
    )
    
    print(f"--- Finished Analysis for Subject: {subject_id}, Run: {run_id}, Contrast: {contrast_name} ---\n")


if __name__ == "__main__":
    args = parser.parse_args()

    # Determine project root directory (assuming script is in fMRI_Data_Analysis/code/)
    script_location = Path(__file__).resolve().parent
    project_root = script_location.parent 
    print(f"Project root directory identified as: {project_root}")
    print(f"Output will be saved under: {Path(args.output_dir).resolve()}")


    runs_to_process = [args.run] if args.run is not None else range(1, args.num_runs + 1)

    if args.subject and args.contrast:
        for run_id_to_process in runs_to_process:
            run_first_level_analysis(
                subject_id=args.subject,
                run_id=run_id_to_process,
                contrast_name=args.contrast,
                output_dir_base=Path(args.output_dir),
                alpha_levels=args.alpha,
                project_root_dir=project_root,
                glm_params=GLM_PARAMS
            )
        print("--- All specified analyses finished. ---")
    else:
        # This case should not be reached if subject and contrast are required=True
        print("Error: --subject and --contrast arguments are mandatory.")
        parser.print_help()