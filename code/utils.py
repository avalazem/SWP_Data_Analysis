import pandas as pd
from nilearn.image import mean_img


def load_confound_data(subject_id, run_ids, project_root_dir):
    """ Loads confound data for a subject across multiple runs. """
    # Load paths
    bids_derivatives_dir = project_root_dir / "single-word-processing" / "data" / "derivatives"
    if not bids_derivatives_dir.exists():
        alt_bids_derivatives_dir = project_root_dir.parent / "data" / "derivatives"
        if alt_bids_derivatives_dir.exists():
            bids_derivatives_dir = alt_bids_derivatives_dir
        else:
            # Fallback to a common relative path if primary structures not found
            bids_derivatives_dir = project_root_dir / ".." / "data" / "derivatives"
            print(f"Warning: Primary derivatives directory not found. Trying relative path: {bids_derivatives_dir}")
    subject_id_padded = f"{subject_id:02d}"
    confound_files_list = []
    for run_id in run_ids:
        run_id_padded = run_id # No padding since run_ids are already formatted as strings with leading zeros
        current_confound_file = bids_derivatives_dir / f"sub-{subject_id_padded}/ses-1/func/sub-{subject_id_padded}_ses-1_task-swp_dir-pa_run-{run_id_padded}_desc-confounds_timeseries.tsv"
        
        if not current_confound_file.exists():
            print(f"ERROR: Confound file not found: {current_confound_file}")
            return None
        
        confound_files_list.append(current_confound_file)
        #print(f"  Loaded confound file for run {run_id_padded}: {current_confound_file}")
    
    # Define which confound columns to include
    motion_confound_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    compound_dataframes = []
    for run_id, confound_file in zip(run_ids, confound_files_list):
        # Load the confound data
        confound_df = pd.read_table(confound_file)
        
        # Check if required columns are present
        missing_columns = [col for col in motion_confound_columns if col not in confound_df.columns]
        if missing_columns:
            print(f"ERROR: Missing confound columns {missing_columns} in file {confound_file}.")
            return None
        
        # Select only the motion confounds
        confound_df = confound_df[motion_confound_columns]
        compound_dataframes.append(confound_df)
        
        # Save or process the confound data as needed
        #print(f"  Loaded confounds for run {run_id}: {confound_file}")

        # Return a list of confound DataFrames
    return compound_dataframes  # Return list of confound file paths



def setup_paths_and_load_data(subject_id, run_ids, project_root_dir):
    """ Defines paths, validates them, and loads event data for multiple runs. """ #
    print(f"Loading data for subject {subject_id}, runs {run_ids}...")
    subject_id_padded = f"{subject_id:02d}"

    # BIDS-like paths for derivatives
    bids_derivatives_dir = project_root_dir / "single-word-processing" / "data" / "derivatives"
    if not bids_derivatives_dir.exists():
        alt_bids_derivatives_dir = project_root_dir.parent / "data" / "derivatives"
        if alt_bids_derivatives_dir.exists():
            bids_derivatives_dir = alt_bids_derivatives_dir
        else:
            # Fallback to a common relative path if primary structures not found
            bids_derivatives_dir = project_root_dir / ".." / "data" / "derivatives"
            print(f"Warning: Primary derivatives directory not found. Trying relative path: {bids_derivatives_dir}")


    anat_file = bids_derivatives_dir / f"sub-{subject_id_padded}/ses-1/anat/sub-{subject_id_padded}_ses-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    
    # Initialize lists for multiple runs
    func_files_list = []
    events_files_list = []
    events_dfs_list = []

    events_dir = project_root_dir / "event_tsvs"
    if not events_dir.exists(): # Fallback
        events_dir = project_root_dir / ".." / "event_tsvs"
        print(f"Warning: Primary events directory not found. Trying relative path: {events_dir}")

    for run_id in run_ids: # Loop through each run_id
        run_id_padded = run_id
        current_func_file = bids_derivatives_dir / f"sub-{subject_id_padded}/ses-1/func/sub-{subject_id_padded}_ses-1_task-swp_dir-pa_run-{run_id_padded}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        current_events_file = events_dir / f"sub_{subject_id_padded}_run_{run_id_padded}_events.tsv"

        if not current_func_file.exists(): 
            print(f"ERROR: Func file not found: {current_func_file}"); return None
        if not current_events_file.exists(): 
            print(f"ERROR: Events file not found: {current_events_file}"); return None

        func_files_list.append(str(current_func_file)) # Nilearn expects strings or Niimg-like objects
        events_files_list.append(current_events_file)
        events_dfs_list.append(pd.read_table(current_events_file))
        
        print(f"  Loaded func for run {run_id_padded}: {current_func_file}")
        #print(f"  Loaded events for run {run_id_padded}: {current_events_file}")

    if not anat_file.exists(): print(f"ERROR: Anat file not found: {anat_file}"); return None
    if not func_files_list: print("ERROR: No functional files were loaded."); return None # Check if list is empty

    mean_func_img = mean_img(func_files_list[0], copy_header=True) if func_files_list else None
    if mean_func_img:
        print("  Mean functional image calculated (from first run).")
    else:
        print("Warning: Could not calculate mean functional image (no functional files loaded).")
    
    # Load confound data
    confound_dfs_list = load_confound_data(subject_id, run_ids, project_root_dir)
    return {
        "anat_file": anat_file,
        "func_files": func_files_list, # Return list of func files
        "events_files": events_files_list, # Return list of event file paths
        "events_dfs": events_dfs_list, # Return list of event dataframes
        "mean_func_img": mean_func_img,
        "confound_data": confound_dfs_list  # Return list of confound DataFrames
    }


