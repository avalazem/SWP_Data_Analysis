import os
import pandas as pd


def load_confound_data(subject_id,
                       session,
                       task,
                        run_ids,
                          path2subjectdata,
                           confound_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']):
    """ Loads confound data for a subject across multiple runs. """
    
    confound_files_list = []
    if run_ids:
        for run_id in run_ids:
            fn_confound = f"sub-{subject_id:02d}_ses-{session}_task-{task}_dir-pa_run-{run_id:02d}_desc-confounds_timeseries.tsv"
            current_confound_file = os.path.join(path2subjectdata, "func", fn_confound)
            
            if not os.path.exists(current_confound_file):
                print(f"ERROR: Confound file not found: {current_confound_file}")
                return None
            
            confound_files_list.append(current_confound_file)
            #print(f"  Loaded confound file for run {run_id_padded}: {current_confound_file}")
    else:
        fn_confound = f"sub-{subject_id:02d}_ses-{session}_task-{task}_dir-pa_desc-confounds_timeseries.tsv"
        current_confound_file = os.path.join(path2subjectdata, "func", fn_confound)
        
        if not os.path.exists(current_confound_file):
            print(f"ERROR: Confound file not found: {current_confound_file}")
            return None
        
        confound_files_list.append(current_confound_file)
        #print(f"  Loaded confound file: {current_confound_file}")
        
    
    compound_dataframes = []
    for confound_file in confound_files_list:
        # Load the confound data
        confound_df = pd.read_table(confound_file)
        
        # Check if required columns are present
        missing_columns = [col for col in confound_columns if col not in confound_df.columns]
        if missing_columns:
            print(f"ERROR: Missing confound columns {missing_columns} in file {confound_file}.")
            return None
        
        # Select only the motion confounds
        confound_df = confound_df[confound_columns]
        compound_dataframes.append(confound_df)
        
        # Save or process the confound data as needed
        #print(f"  Loaded confounds for run {run_id}: {confound_file}")

        # Return a list of confound DataFrames
    return compound_dataframes  # Return list of confound file paths

def load_BIDS_data(exp_args, run_ids=None, path2root="", load_confounds=False):
    """
    Loads BIDS-formatted data for one or more subjects.

    Args:
        exp_args (dict): Dictionary with subject(s), session, and task.
                         'subject' can be an int or a list of ints.
        run_ids (list, optional): List of run IDs. Defaults to None.
        path2root (str, optional): Path to the BIDS root directory. Defaults to "".
        load_confounds (bool, optional): If True, loads confound data. Defaults to False.

    Returns:
        dict: A dictionary containing anatomical file paths, a list of functional 
              file paths, and concatenated dataframes for events and confounds.
    """
    subject_ids, session, task = exp_args['subject'], exp_args['session'], exp_args['task']
    if not isinstance(subject_ids, list):
        subject_ids = [subject_ids]

    all_fn_anat = []
    all_fns_func, all_fns_events = [], []
    all_dfs_events, all_dfs_confounds = [], []

    for subject_id in subject_ids:
        fn_base = f"sub-{subject_id:02d}_ses-{session}_task-{task}"
        path2subject_data = os.path.join(path2root, "data", "derivatives", f"sub-{subject_id:02d}", "ses-1")
        
        # Anatomical data file
        fn_anat = f"{fn_base.split('_task-')[0]}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
        fn_anat = os.path.join(path2subject_data, "anat", fn_anat)
        all_fn_anat.append(fn_anat)
    
        # Functional and event data
        if run_ids:
            for run_id in run_ids:
                fn_func = f"{fn_base}_dir-pa_run-{run_id:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                fn_func = os.path.join(path2subject_data, "func", fn_func)                                              
                all_fns_func.append(fn_func)
                
                current_events_file = os.path.join(path2subject_data, "func", f"{fn_base}_run-{run_id:02d}_events.tsv")
                all_fns_events.append(current_events_file)
                df_events = pd.read_table(current_events_file)
                df_events['subject_id'] = subject_id # Add subject ID to dataframe
                all_dfs_events.append(df_events)
        else:
            fn_func = f"{fn_base}_dir-pa_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            fn_func = os.path.join(path2subject_data, "func", fn_func)                                              
            all_fns_func.append(fn_func)
            
            current_events_file = os.path.join(path2subject_data,"func", f"{fn_base}_events.tsv")
            all_fns_events.append(current_events_file)
            df_events = pd.read_table(current_events_file)
            df_events['subject_id'] = subject_id # Add subject ID to dataframe
            all_dfs_events.append(df_events)

        if load_confounds:
            confound_dfs_list = load_confound_data(subject_id, session, task, run_ids, path2subject_data)
            all_dfs_confounds.extend(confound_dfs_list)

    return {
        "fn_anat": all_fn_anat,
        "fns_func": all_fns_func,
        "fns_events": all_fns_events,
        "dfs_events": all_dfs_events,
        "dfs_confounds": all_dfs_confounds
    }

def load_BIDS_data_temp(exp_args, run_ids, path2root, load_confounds=False):
    subject_id, session, task = exp_args['subject'], exp_args['session'], exp_args['task']
    fn_base = f"sub-{subject_id:02d}_ses-{session}_task-{task}"
    # BIDS-like paths for derivatives
    path2subject_data = os.path.join(path2root, "data", "derivatives", f"sub-{subject_id:02d}", "ses-1")
    
    # Anatomical data file
    fn_anat = f"{fn_base.split('_task-')[0]}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    fn_anat = os.path.join(path2subject_data, "anat", fn_anat)
    
    # Initialize lists for multiple runs
    fns_func, fns_events, dfs_events = [], [], []
    if run_ids:
        for run_id in run_ids: # Loop through each run_id
            # Functional data file
            fn_func = f"{fn_base}_dir-pa_run-{run_id:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            fn_func = os.path.join(path2subject_data, "func", fn_func)                                              
            fns_func.append(fn_func) # Nilearn expects strings or Niimg-like objects
            
            # Event file
            current_events_file = os.path.join(path2subject_data,"func",
                                                f"{fn_base}_run-{run_id:02d}_events.tsv")
            fns_events.append(current_events_file)
            dfs_events.append(pd.read_table(current_events_file))
    else:
        fn_func = f"{fn_base}_dir-pa_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        fn_func = os.path.join(path2subject_data, "func", fn_func)                                              
        fns_func.append(fn_func)
        current_events_file = os.path.join(path2subject_data,"func",
                                                f"{fn_base}_events.tsv")
        fns_events.append(current_events_file)
        dfs_events.append(pd.read_table(current_events_file))
        # Event file

    
    if load_confounds:
        confound_dfs_list = load_confound_data(subject_id, session, task, run_ids, path2subject_data)

    return {
        "fn_anat": fn_anat,
        "fns_func": fns_func, # Return list of func files
        "fns_events": fns_events, # Return list of event file paths
        "dfs_events": dfs_events, # Return list of event dataframes
        "dfs_confounds": confound_dfs_list  # Return list of confound DataFrames
    }