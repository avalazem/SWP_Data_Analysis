from nilearn.glm.first_level import FirstLevelModel


# Default GLM parameters
GLM_PARAMS = {
    "t_r": 1.81,
    "noise_model": "ar1",
    "standardize": False,
    "hrf_model": "spm",
    "drift_model": "cosine",
    "high_pass": 0.01,
}

def fit_glm_model(func_files, events_dfs, confound_dfs_list, glm_params):
    """Fits the GLM and returns the model and the design matrix. Handles multiple runs by concatenation."""
    print("Fitting GLM model...")
    fmri_glm = FirstLevelModel(**glm_params)
    fmri_glm.fit(func_files, events_dfs, confound_dfs_list)
    design_matrix = fmri_glm.design_matrices_[0]
    print("  GLM fitting complete. Design matrix extracted.")
    return fmri_glm, design_matrix

