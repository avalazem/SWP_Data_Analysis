import matplotlib.pyplot as plt
from nilearn.plotting import plot_design_matrix as nilearn_plot_design_matrix
from nilearn.plotting import plot_contrast_matrix as nilearn_plot_contrast_matrix
from nilearn.plotting import plot_anat, plot_img, plot_stat_map
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_glass_brain


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