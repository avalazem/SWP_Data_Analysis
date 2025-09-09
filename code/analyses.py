import pickle
import os
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_stat_map
from nilearn.plotting import plot_glass_brain
from nilearn.plotting import view_img_on_surf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.validation import check_is_fitted, NotFittedError
from nilearn.plotting import plot_img_on_surf

from viz import plot_design_matrix_to_file, plot_contrast_matrix_to_file


from nilearn.glm.first_level import FirstLevelModel


# Main first-level analysis function for a single subject, multiple runs (concatenated), and contrast
def fit_GLM(exp_args, fns_func, dfs_events, dfs_confounds,
            glm_params, path2root, save_model=True):
    """
    Performs first-level fMRI analysis for a given subject, concatenating specified runs, for a given contrast.
    """
    # build file and folder names based on experiment arguments
    subject_id, session, task = exp_args['subject'], exp_args['session'], exp_args['task']
    
    if isinstance(subject_id, list):
    # If it's a list, join the formatted subject IDs
        subject_ids_str = "_".join([f"{sub:02d}" for sub in subject_id])
    else:
        # If it's a single integer, format it directly
        subject_ids_str = f"{subject_id:02d}"
    fn_base = f"sub-{subject_ids_str}_ses-{session}_task-{task}"

    path2output = os.path.join(path2root, "output", "glm_models")
    fn_glm = f'glm_{fn_base}.pkl'  # Use the first functional file name for GLM
    
    fmri_glm_file = os.path.join(path2output, fn_glm)

    needs_fitting = False

    if os.path.exists(fmri_glm_file):
        print(f"GLM model already exists at {fmri_glm_file}. Loading existing model...")
        with open(fmri_glm_file, 'rb') as f:
            fmri_glm = pickle.load(f)
        
        try:
            check_is_fitted(fmri_glm)
            print("  Loaded model is already fitted.")
        except NotFittedError:
            print("  WARNING: Loaded model is not fitted. It will be re-fitted.")
            needs_fitting = True
    else:
        print("No existing GLM model found. Creating and fitting a new one.")
        fmri_glm = FirstLevelModel(**glm_params)
        needs_fitting = True

    if needs_fitting:
        # Fit GLM (re-fit if loaded, as pickle can be unreliable for fitted state)
        print("Fitting GLM model...")
        fmri_glm.fit(fns_func, dfs_events, dfs_confounds)
        print("  GLM fitting complete.")
        if save_model:
            # Create output directory if it doesn't exist
            os.makedirs(path2output, exist_ok=True)
            # Save the fitted model
            with open(fmri_glm_file, 'wb') as f:
                pickle.dump(fmri_glm, f)
            print(f"Fitted GLM model saved to {fmri_glm_file}")

    return fmri_glm
   
def plot_contrast(exp_args, fmri_glm, contrast_name, contrast_vector,
                   mean_func_img, path2root,
                    threshold_z=3.1,
                    cluster_threshold=10,
                    save_plots=True):
    """
    Plots the contrast for the fitted GLM model.
    """
    # build file and folder names based on experiment arguments
    subject_id, session, task = exp_args['subject'], exp_args['session'], exp_args['task']

    if isinstance(subject_id, list):
        subject_ids_str = "_".join([f"{sub:02d}" for sub in subject_id])
        fn_base = f"contrast-{contrast_name}_sub-{subject_ids_str}_ses-{session}_task-{task}"
        folder_figures = os.path.join(path2root,
                                      "figures",
                                      f"sub-{subject_ids_str}_ses-{session}",
                                      "contrasts")
    else:
        fn_base = f"contrast-{contrast_name}_sub-{subject_id:02d}_ses-{session}_task-{task}"
        folder_figures = os.path.join(path2root,
                                      "figures",
                                      f"sub-{subject_id:02d}_ses-{session}",
                                      "contrasts")
    
    print("Plotting images...") 
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

    #fdr_corrected_map, threshold = threshold_stats_img(...alpha..)
    # Plot stat map
    stat_map_plotting_config = {"bg_img": mean_func_img, 
                                "display_mode": "z", 
                                "cut_coords": 3, 
                                "black_bg": True,
                                "symmetric_cbar": True,
                                "cmap": "cold_hot"}
    
    title_stat_map = (f"{contrast_name} (thresh: {threshold_z:.3f}; clusters > {cluster_threshold} voxels)")
    
    stat_map_filepath = os.path.join(folder_figures, f"stat_map_threshold_z_{threshold_z}_{fn_base}.png")
    plot_stat_map(z_map, threshold=threshold_z, 
                  title=title_stat_map,
                  figure=plt.figure(figsize=(10, 4)), 
                  output_file=stat_map_filepath,
                  **stat_map_plotting_config)
    print(f"  Statistical map saved to {stat_map_filepath}")

    # Plot glass brain
    glass_brain_plotting_config = {"display_mode": "lyrz", 
                                   "plot_abs": False,
                                   "symmetric_cbar": True,
                                   "cut_coords": (0,0,0), 
                                   "colorbar": True, 
                                   "annotate": True, 
                                   "draw_cross": False, 
                                   "black_bg": False}
    
    fn_glass_brain = f"glass_brain_threshold_z_{threshold_z}_{fn_base}.png"
    glass_brain_filepath = os.path.join(folder_figures, fn_glass_brain)
    plot_glass_brain(z_map, threshold=threshold_z, 
                     title=title_stat_map,
                     output_file=glass_brain_filepath,
                     **glass_brain_plotting_config)
    print(f"  Glass brain plot saved to {glass_brain_filepath}")


    fn_surf_brain = f"surf_brain_threshold_z_{threshold_z}_{fn_base}.png"
    
    plot_img_on_surf(
    stat_map=z_map,
    views=["lateral", "medial"],
    hemispheres=["left", "right"],
    bg_on_data=True,
    threshold=threshold_z,
    colorbar=True,
    cmap='cold_hot',
    inflate=True,
    output_file=os.path.join(folder_figures, f"{fn_surf_brain}.png"))
    
    surf_brain_filepath = os.path.join(folder_figures, fn_surf_brain)
    view=view_img_on_surf(z_map, 
                     threshold=threshold_z,
                     surf_mesh='fsaverage')
    
    view.save_as_html(f"{surf_brain_filepath}.html")


    
    
    plt.close('all')

