from pathlib import Path

from utils import (
    setup_paths_and_load_data
)

from viz import (
    plot_design_matrix_to_file,
    plot_contrast_matrix_to_file,
    plot_diagnostic_images_to_file,
    compute_threshold_plot_stat_maps_to_file
)

from compute_contrast import load_contrast_vector

from compute_model import fit_glm_model


# Main first-level analysis function for a single subject, multiple runs (concatenated), and contrast

def run_first_level_analysis(subject_id, run_ids, contrast_name, output_dir_base, alpha_levels, project_root_dir, glm_params): # Added run_ids
    """
    Performs first-level fMRI analysis for a given subject, concatenating specified runs, for a given contrast.
    """
    if not isinstance(run_ids, list):
        run_ids = [run_ids] # Ensure run_ids is a list, even if a single run number is passed

    runs_label_str = "_".join(map(lambda r: f"{r}", run_ids))
    print(f"--- Starting Analysis for Subject: {subject_id}, Runs: {runs_label_str}, Contrast: {contrast_name} ---")

    # 0. Setup output directory and base filename
    output_dir_base_path = Path(output_dir_base)
    # Per-analysis (potentially multi-run) output directory
    analysis_specific_output_dir = output_dir_base_path / f"sub-{subject_id:02d}" / f"{contrast_name.replace(' ', '_')}"
    analysis_specific_output_dir.mkdir(exist_ok=True, parents=True)
    
    contrast_name_safe = contrast_name.replace(">", "_vs_") # Used for filenames
    base_fn_name_prefix = f"sub-{subject_id:02d}_{contrast_name_safe}"
    base_output_filepath_prefix = analysis_specific_output_dir / base_fn_name_prefix

    # 1. Load Data for all specified runs
    data_loaded = setup_paths_and_load_data(subject_id, run_ids, project_root_dir)
    if not data_loaded:
        print(f"ERROR: Data loading failed for sub-{subject_id}, runs-{runs_label_str}. Skipping analysis.")
        return
    
    # 2. Fit GLM (fit_glm_model should accept lists of func_files, events_dfs), and confound_dfs_list
    fmri_glm, design_matrix = fit_glm_model(data_loaded["func_files"], data_loaded["events_dfs"], data_loaded["confound_data"], glm_params)
    
    
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
    cluster_threshold = 10 # Default, consider making it an argument if it varies
    
    compute_threshold_plot_stat_maps_to_file(
        fmri_glm, contrast_vector, contrast_name, contrast_name_safe, 
        data_loaded["mean_func_img"], current_alpha, cluster_threshold, 
        base_output_filepath_prefix
    )
    
    print(f"--- Finished Analysis for Subject: {subject_id}, Contrast: {contrast_name} ---\n")

