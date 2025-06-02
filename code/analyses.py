import pickle
import os
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_stat_map
from nilearn.plotting import plot_glass_brain
import matplotlib.pyplot as plt
import numpy as np

from viz import plot_design_matrix_to_file, plot_contrast_matrix_to_file


from nilearn.glm.first_level import FirstLevelModel


# Main first-level analysis function for a single subject, multiple runs (concatenated), and contrast

def fit_GLM(fns_func, dfs_events, dfs_confounds,
            glm_params, path2root, save_model=True,
            plot_design_matrix=False): # Added run_ids
    """
    Performs first-level fMRI analysis for a given subject, concatenating specified runs, for a given contrast.
    """
    
    path2output = os.path.join(path2root, "output")
    fn_glm = fns_func[0].split('/')[-1].replace('.nii.gz', '_glm.pkl')  # Use the first functional file name for GLM
    fmri_glm_file = os.path.join(path2output, fn_glm)
    if os.path.exists(fmri_glm_file):
        print(f"GLM model already exists at {fmri_glm_file}. Loading existing model...")
        with open(fmri_glm_file, 'rb') as f:
            fmri_glm = pickle.load(f)
        return fmri_glm    

    # Fit GLM (fit_glm_model should accept lists of func_files, events_dfs), and confound_dfs_list
    print("Fitting GLM model...")
    fmri_glm = FirstLevelModel(**glm_params)
    fmri_glm.fit(fns_func, dfs_events, dfs_confounds)
    print("  GLM fitting complete. Design matrix extracted.")

    if save_model:
        # Create output directory if it doesn't exist
        
        os.makedirs(path2output, exist_ok=True)
        
        # Save the fitted model
        with open(fmri_glm_file, 'wb') as f:
            pickle.dump(fmri_glm, f)
        
        print(f"Fitted GLM model saved to {fmri_glm_file}")


    if plot_design_matrix:
        # Plot the design matrix
        path2figures = os.path.join(path2root, "figures")
        fn_design_matrix = fns_func[0].split('/')[-1].replace('.nii.gz', '_design_matrix.png')  # Use the first functional file name for GLM
        design_matrix = fmri_glm.design_matrices_[0]
        dm_plot_path = os.path.join(path2figures, fn_design_matrix)
        plot_design_matrix_to_file(design_matrix, dm_plot_path)

    
    return fmri_glm
   
def plot_contrast(exp_args, fmri_glm, contrast_name, contrast_vector,
                   mean_func_img, path2root,
                     current_alpha=0.05, cluster_threshold=10, save_plots=True):
    """
    Plots the contrast for the fitted GLM model.
    """
    # build file and folder names based on experiment arguments
    subject_id, session, task = exp_args['subject'], exp_args['session'], exp_args['task']
    fn_base = f"contrast-{contrast_name}_sub-{subject_id:02d}_ses-{session}_task-{task}"
    print("Plotting diagnostic images...")
    folder_figures = os.path.join(path2root,
                                   "figures",
                                   f"sub-{subject_id:02d}_ses-{session}",
                                     "contrasts")
    os.makedirs(folder_figures, exist_ok=True)
    print(f"  Saving Contrast images to {folder_figures}...")

    # Plot design matrix
    print(f"Plotting contrast: {contrast_name}...")
    contrast_vector = np.array(contrast_vector)  # Ensure it's a numpy array

    fn_contrast_matrix = f"contrast_matrix_{fn_base}.png"
    cm_plot_path = os.path.join(folder_figures,  fn_contrast_matrix)
    plot_contrast_matrix_to_file(contrast_vector, fmri_glm.design_matrices_[0], cm_plot_path)
    
    
    # Compute and plot statistical maps
    print("Computing and plotting statistical maps...")
    # Compute z-map
    z_map = fmri_glm.compute_contrast(contrast_vector, output_type="z_score")
    print(f"  Z-map computed for contrast: {contrast_name}")

    # Threshold the z-map
    print(f"  Thresholding z-map with alpha={current_alpha}, cluster_threshold={cluster_threshold}...")
    
    clean_map, threshold = threshold_stats_img(
        z_map, 
        alpha=current_alpha, 
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
    title_stat_map = (f"{contrast_name} (p<{current_alpha:.3f} FDR; thresh: {threshold:.3f}; clusters > {cluster_threshold} voxels)")
    
    stat_map_filepath = os.path.join(folder_figures, f"stat_map_alpha{current_alpha}_{fn_base}.png")
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
    
    fn_glass_brain = f"glass_brain_{contrast_name}_alpha{current_alpha}_{fn_base}.png"
    glass_brain_filepath = os.path.join(folder_figures, fn_glass_brain)
    plot_glass_brain(clean_map, threshold=threshold, 
                     title=title_stat_map, 
                     output_file=glass_brain_filepath,
                     **glass_brain_plotting_config)
    print(f"  Glass brain plot saved to {glass_brain_filepath}")
    plt.close('all')

