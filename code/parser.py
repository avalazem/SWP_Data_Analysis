import argparse
import sys

def parse_arguments():
    """
    Parses command-line arguments and returns the parsed args object.
    """
    # --- Argument parser setup ---
    parser = argparse.ArgumentParser(description="Process fMRI data for a single subject and run.")

    # General Processing Arguments Group
    exp_args = parser.add_argument_group("General Processing Arguments")
    exp_args.add_argument("--subject", type=int, default=3, help="Subject number (e.g., 1)")
    exp_args.add_argument("--session", type=int, default=1, help="Session number (e.g., 1)")
    exp_args.add_argument("--task", type=str, default='swp', help="Task name is required in BIDS, e.g., 'swp'")
    exp_args.add_argument("--num-runs", type=int, default=6, help="Number of runs to process (default: 1)")

    # Contrast Arguments Group
    contrast_args = parser.add_argument_group("Contrast Arguments")
    contrast_args.add_argument("--contrast-file", type=str, default="contrasts.json", help="Path to the contrast file (default: contrasts.json)")
    contrast_args.add_argument("--contrast-name", type=str, default="real > pseudo", help="Name of the contrast to analyze (default: vis_aud)")

    # Statistical Thresholding Arguments Group
    stat_args = parser.add_argument_group("Statistical Thresholding Arguments")
    stat_args.add_argument("--threshold_z", type=float, default=2, help="Alpha level for statistical thresholding (default: 0.05)")
    stat_args.add_argument("--alpha", type=float, default=0.05, help="Alpha level for statistical thresholding (default: 0.05)")
    stat_args.add_argument("--cluster-threshold", type=int, default=1, help="Cluster size threshold for statistical maps (default: 10)")

    # Path Arguments Group
    path_args = parser.add_argument_group("Path Arguments")
    path_args.add_argument("--path2root", type=str, default='..', help="Path to input data directory")

    # GLM Parameters Group (for help message organization)
    glm_args = parser.add_argument_group("GLM Parameters")
    glm_args.add_argument("--t-r", type=float, default=1.81, help="GLM parameter: Repetition time (default: 1.81)")
    glm_args.add_argument("--noise-model", type=str, default="ar1", help="GLM parameter: Noise model (default: ar1)")
    glm_args.add_argument("--standardize", action="store_true", help="GLM parameter: Standardize (default: False)")
    glm_args.add_argument("--hrf-model", type=str, default="spm", help="GLM parameter: HRF model (default: spm)")
    glm_args.add_argument("--drift-model", type=str, default="cosine", help="GLM parameter: Drift model (default: cosine)")
    glm_args.add_argument("--high-pass", type=float, default=0.01, help="GLM parameter: High pass filter cutoff (default: 0.01)")
    
    return parser.parse_args()

def get_arg_groups(args):
    """
    Separates parsed arguments into logical groups.
    """
    exp_params = {
        'subject': args.subject,
        'session': args.session,
        'task': args.task,
        'num_runs': args.num_runs
    }

    glm_params = {
        't_r': args.t_r,
        'noise_model': args.noise_model,
        'standardize': args.standardize,
        'hrf_model': args.hrf_model,
        'drift_model': args.drift_model,
        'high_pass': args.high_pass
    }
    
    return exp_params, glm_params