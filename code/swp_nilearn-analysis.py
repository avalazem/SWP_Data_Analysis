from pathlib import Path
import argparse

from utils import (
    setup_paths_and_load_data,
    fit_glm_model,
    load_contrast_vector
)

from viz import (
    plot_design_matrix_to_file,
    plot_contrast_matrix_to_file,
    plot_diagnostic_images_to_file,
    compute_threshold_plot_stat_maps_to_file
)

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