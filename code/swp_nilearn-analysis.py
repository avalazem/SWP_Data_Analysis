import os
from pathlib import Path
import argparse

from analyses import fit_GLM
from analyses import plot_contrast
from utils import load_BIDS_data
from viz import plot_diagnostic_images_to_file
from contrasts import ContrastManager
from nilearn.image import mean_img
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- Argument parser setup ---
parser = argparse.ArgumentParser(description="Process fMRI data for a single subject and run.")

# General Processing Arguments Group
general_args = parser.add_argument_group("General Processing Arguments")
general_args.add_argument("--subject", type=int, default=1, help="Subject number (e.g., 1)")
general_args.add_argument("--task", type=str, default='swp', help="Task name is required in BIDS")
general_args.add_argument("--num-runs", type=int, default=6, help="Number of runs to process (default: 1)")

# Contrast Arguments Group
contrast_args = parser.add_argument_group("Contrast Arguments")
contrast_args.add_argument("--contrast-file", type=str, default="contrasts.json", help="Path to the contrast file (default: contrasts.json)")
contrast_args.add_argument("--contrast-name", type=str, default="viz_gt_aud", help="Name of the contrast to analyze (default: vis_aud)")

# Statistical Thresholding Arguments Group
stat_args = parser.add_argument_group("Statistical Thresholding Arguments")
stat_args.add_argument("--alpha", type=float, default=0.05, help="Alpha level for statistical thresholding (default: 0.05)")
stat_args.add_argument("--cluster-threshold", type=int, default=10, help="Cluster size threshold for statistical maps (default: 10)")

# Path Arguments Group
path_args = parser.add_argument_group("Path Arguments")
path_args.add_argument("--path2root", type=str, default='..', help="Path to input data directory")

# GLM Parameters Group (for help message organization)
glm_group = parser.add_argument_group("GLM Parameters")
glm_group.add_argument("--t-r", type=float, default=1.81, help="GLM parameter: Repetition time (default: 1.81)")
glm_group.add_argument("--noise-model", type=str, default="ar1", help="GLM parameter: Noise model (default: ar1)")
glm_group.add_argument("--standardize", action="store_true", help="GLM parameter: Standardize (default: False)")
glm_group.add_argument("--hrf-model", type=str, default="spm", help="GLM parameter: HRF model (default: spm)")
glm_group.add_argument("--drift-model", type=str, default="cosine", help="GLM parameter: Drift model (default: cosine)")
glm_group.add_argument("--high-pass", type=float, default=0.01, help="GLM parameter: High pass filter cutoff (default: 0.01)")

# Parse the arguments
args = parser.parse_args()

glm_params = {}
for action in glm_group._group_actions:
    if hasattr(args, action.dest):
        glm_params[action.dest] = getattr(args, action.dest)

confound_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

# LOAD DATA - Anatomy, functional and events
dict_BIDS_data = load_BIDS_data(args.subject,
                                 range(1, args.num_runs+1),  # Adjusted to use runs from 1 to num_runs
                                  args.path2root,
                                   confound_columns)

# Plot diagnostic images
mean_func_img = mean_img(dict_BIDS_data["fns_func"][0], copy_header=True)
plot_diagnostic_images_to_file(mean_func_img,
                                dict_BIDS_data["fn_anat"],
                                args.path2root)

# FIT MODEL: GLM
model_glm = fit_GLM(dict_BIDS_data['fns_func'],
                    dict_BIDS_data['dfs_events'],
                    dict_BIDS_data['dfs_confounds'],
                    glm_params,
                    args.path2root,
                    save_model=True,
                    plot_design_matrix=True)


# CONTRAST ANALYSIS
manager = ContrastManager(args.contrast_file)

# Retrieve a contrast
vis_aud_contrast = manager.get_contrast(args.contrast_name)
        
# Plot the contrast
plot_contrast(model_glm, args.contrast_name, vis_aud_contrast['weights'],
              mean_func_img, dict_BIDS_data["fn_anat"],
              args.path2root,
              args.alpha, args.cluster_threshold,
              save_plots=True)