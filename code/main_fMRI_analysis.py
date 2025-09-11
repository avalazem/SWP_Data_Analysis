import os
from pathlib import Path
import sys
from analyses import fit_GLM
from analyses import plot_contrast
from utils import load_BIDS_data
from viz import plot_diagnostic_images_to_file
from viz import plot_design_matrix_to_file
from contrasts import ContrastManager
from nilearn.image import mean_img
from parser import parse_arguments, get_arg_groups

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Parse the arguments from the separate parser file
    args = parse_arguments()
    exp_params, glm_params = get_arg_groups(args)

    n_subjects = 1 if isinstance(exp_params['subject'], int) else len(exp_params['subject'])
    confound_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

    # LOAD DATA - Anatomy, functional and events
    run_ids = list(range(1, args.num_runs + 1)) if args.num_runs else []
    dict_BIDS_data = load_BIDS_data(exp_params,
                                    run_ids,
                                    args.path2root,
                                    confound_columns)

    # Plot diagnostic images
    mean_func_img = mean_img(dict_BIDS_data["fns_func"][0], copy_header=True)
    plot_diagnostic_images_to_file(exp_params,
                                   [mean_func_img]*n_subjects,
                                   dict_BIDS_data["fn_anat"],
                                   args.path2root)

    # FIT MODEL: GLM
    glm_params['smoothing_fwhm'] = 8.0  # Example smoothing parameter
    model_glm = fit_GLM(exp_params,
                        dict_BIDS_data['fns_func'],
                        dict_BIDS_data['dfs_events'],
                        dict_BIDS_data['dfs_confounds'],
                        glm_params,
                        args.path2root,
                        save_model=True)
    
    # Plot the design matrix
    plot_design_matrix_to_file(model_glm, exp_params, args.path2root)

    # CONTRAST ANALYSIS
    manager = ContrastManager(args.contrast_file)
    contrast = manager.get_contrast(args.contrast_name)

    # Hack to ensure the contrast weights are the same length as the design matrix
    if len(contrast['weights']) < model_glm.design_matrices_[0].shape[1]:
        contrast['weights'] += [0] * (model_glm.design_matrices_[0].shape[1] - len(contrast['weights']))

    # Plot the contrast
    plot_contrast(exp_params,
                  model_glm, args.contrast_name, contrast['weights'],
                  mean_func_img,
                  args.path2root,
                  args.threshold_z, args.cluster_threshold,
                  save_plots=True)

if __name__ == '__main__':
    main()