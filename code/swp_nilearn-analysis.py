from pathlib import Path
import argparse

from analyses import (
    run_first_level_analysis
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
parser.add_argument("--contrast", type=str, required=True, help="Contrast name (e.g., 'visual>auditory')")
parser.add_argument("--output_dir", type=str, default="results", help="Base output directory for results")
parser.add_argument("--alpha", type=float, nargs='+', default=[0.05], help="Alpha levels for thresholding")
parser.add_argument("--num_runs", type=int, default=6, help="Total number of runs if --run is not specified.")



if __name__ == "__main__":
    args = parser.parse_args()

    # Determine project root directory (assuming script is in fMRI_Data_Analysis/code/)
    script_location = Path(__file__).resolve().parent
    project_root = script_location.parent
    print(f"Project root directory identified as: {project_root}")
    print(f"Output will be saved under: {Path(args.output_dir).resolve()}")

    # Generate run_ids from num_runs
    run_ids = [f"{i:02d}" for i in range(1, args.num_runs + 1)]

    if args.subject and args.contrast:
        run_first_level_analysis(
            subject_id=args.subject,
            contrast_name=args.contrast,
            output_dir_base=Path(args.output_dir),
            alpha_levels=args.alpha,
            project_root_dir=project_root,
            glm_params=GLM_PARAMS,
            run_ids=run_ids
            )
        print("--- All specified analyses finished. ---")
    else:
        # This case should not be reached if subject and contrast are required=True
        print("Error: --subject and --contrast arguments are mandatory.")
        parser.print_help()