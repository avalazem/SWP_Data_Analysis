import os
import matplotlib.pyplot as plt
from nilearn.plotting import plot_design_matrix as nilearn_plot_design_matrix
from nilearn.plotting import plot_design_matrix_correlation
from nilearn.plotting import plot_contrast_matrix as nilearn_plot_contrast_matrix
from nilearn.plotting import plot_anat, plot_img, plot_stat_map
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_glass_brain
from nilearn.image import mean_img

def plot_design_matrix_to_file(fmri_glm, exp_args, path2root):
    """Plots the design matrix and saves it to a file."""
    subject_id, session, task = exp_args['subject'], exp_args['session'], exp_args['task']

    # Check if subject_id is a list and format accordingly
    if isinstance(subject_id, list):
        subject_ids_str = "_".join([f"{sub:02d}" for sub in subject_id])
        fn_base = f"sub-{subject_ids_str}_ses-{session}_task-{task}"
    else:
        subject_ids_str = f"{subject_id:02d}"
        fn_base = f"sub-{subject_ids_str}_ses-{session}_task-{task}"

    folder_figures = os.path.join(path2root, "figures", f"sub-{subject_ids_str}_ses-{session}", "design_matrices")

    os.makedirs(folder_figures, exist_ok=True)
    print(f"  Saving design matrix plot to {folder_figures}...")
    
    design_matrix = fmri_glm.design_matrices_[0]
    nilearn_plot_design_matrix(design_matrix, output_file=os.path.join(folder_figures,
                                                                        f"design_matrix_{fn_base}.png"))
    plt.close() # Close the figure to free memory
    plot_design_matrix_correlation(design_matrix, 
                                   output_file=os.path.join(folder_figures,
                                                             f"design_matrix_correlation_{fn_base}.png"))
    plt.close()  # Close the figure to free memory
    

def plot_contrast_matrix_to_file(contrast_vector, design_matrix, output_filepath):
    """Plots the contrast matrix and saves it to a file."""
    print(f"Plotting contrast matrix to {output_filepath}...")
    nilearn_plot_contrast_matrix(contrast_vector, design_matrix, output_file=output_filepath)
    plt.close()
    print("  Contrast matrix plot saved.")


def plot_diagnostic_images_to_file(exp_args, mean_func_img, anat_file, path2root):
    """Plots and saves mean functional and anatomical images for one or more subjects."""

    # Ensure subject_id is a list for consistent iteration
    subject_ids = exp_args['subject']
    if not isinstance(subject_ids, list):
        subject_ids = [subject_ids]

    # Handle single or multiple images
    if not isinstance(mean_func_img, list):
        mean_func_img = [mean_func_img]
    if not isinstance(anat_file, list):
        anat_file = [anat_file]

    if len(subject_ids) != len(mean_func_img) or len(subject_ids) != len(anat_file):
        raise ValueError("The number of subjects, mean functional images, and anatomical files must be the same.")

    print("Plotting diagnostic images...")

    for i, subject_id in enumerate(subject_ids):
        session, task = exp_args['session'], exp_args['task']
        
        # Build file and folder names based on experiment arguments
        fn_base = f"sub-{subject_id:02d}_ses-{session}_task-{task}"
        
        # Get the corresponding image files for the current subject
        current_mean_func = mean_func_img[i]
        current_anat_file = anat_file[i]
        
        folder_figures = os.path.join(path2root, "figures", f"sub-{subject_id:02d}_ses-{session}", "diagnostic_images")
        os.makedirs(folder_figures, exist_ok=True)
        print(f"  Saving diagnostic images for subject {subject_id} to {folder_figures}...")
        
        # Plot mean functional and anatomical images
        plot_img(current_mean_func, 
                 colorbar=True, 
                 cbar_tick_format="%i", 
                 cmap="gray",
                 output_file=os.path.join(folder_figures, f"{fn_base}_mean_func_img.png"))
        plot_anat(current_anat_file, 
                  colorbar=True, 
                  cbar_tick_format="%i",
                  output_file=os.path.join(folder_figures, f"{fn_base}_anat_img.png"))
        plt.close('all')
        print(f"  Diagnostic images for subject {subject_id} saved.")
  
    
def plot_diagnostic_images_to_file_temp(exp_args, mean_func_img, anat_file, path2root):
    """Plots and saves mean functional and anatomical images."""
    # Build file and folder names based on experiment arguments
    subject_id, session, task = exp_args['subject'], exp_args['session'], exp_args['task']
    fn_base = f"sub-{subject_id:02d}_ses-{session}_task-{task}"
    print("Plotting diagnostic images...")
    folder_figures = os.path.join(path2root, "figures", f"sub-{subject_id:02d}_ses-{session}", "diagnostic_images")
    os.makedirs(folder_figures, exist_ok=True)
    print(f"  Saving diagnostic images to {folder_figures}...")
    
    # Plot mean functional and anatomical images
    plot_img(mean_func_img, 
             colorbar=True, 
             cbar_tick_format="%i", 
             cmap="gray",
             output_file=os.path.join(folder_figures, f"{fn_base}_mean_func_img.png"))
    plot_anat(anat_file, 
              colorbar=True, 
              cbar_tick_format="%i",
              output_file=os.path.join(folder_figures, f"{fn_base}_anat_img.png"))
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
        z_map, 
        alpha=current_alpha, 
        height_control="fdr",
        cluster_threshold=cluster_threshold, 
        two_sided=True,
    )
    print(f"  Thresholded map generated. Threshold value: {threshold:.3f}")

    # Plot stat map
    stat_map_plotting_config = {"bg_img": mean_func_img, 
                                "display_mode": "z", 
                                "cut_coords": 3, 
                                "black_bg": True}
    title_stat_map = (f"{original_contrast_name} (p<{current_alpha:.3f} FDR; thresh: {threshold:.3f}; clusters > {cluster_threshold} voxels)")
    
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
    plot_glass_brain(clean_map, threshold=threshold, 
                     title=title_stat_map, 
                     figure=plt.figure(figsize=(10, 8)), 
                     output_file=glass_brain_filepath,
                     **glass_brain_plotting_config)
    print(f"  Glass brain plot saved to {glass_brain_filepath}")
    plt.close('all')