import pandas as pd
import os
from pathlib import Path

# Function for condition mappings
# Extract the trial_type from Condition and Modalities

def build_trial_type_list(df):
    # Mappings for Condition column parts
    lexicality_map = {'R': 'real', 'P': 'pseudo'}
    length_map = {'L': 'long', 'S': 'short'}
    frequency_map = {'L': 'low', 'H': 'high'}
    complexity_map = {'S': 'simple', 'C': 'complex'}

    def trial_type_row(row):
        # Input and output modalities
        input_mod = row['Input Modality'].lower()
        output_mod = row['Output Modality'].lower()
        # Condition string
        cond = row['Condition']
        # Lexicality, Length, Frequency, Complexity
        lexicality = lexicality_map.get(cond[0], cond[0])
        length = length_map.get(cond[1], cond[1])
        # If pseudo, skip frequency
        if cond[0] == 'P':
            complexity = complexity_map.get(cond[2], cond[2])
            trial_parts = [input_mod, output_mod, lexicality, length, complexity]
        else:
            frequency = frequency_map.get(cond[2], cond[2])
            complexity = complexity_map.get(cond[3], cond[3])
            trial_parts = [input_mod, output_mod, lexicality, length, frequency, complexity]
        return '_'.join(trial_parts)

    return trial_type_row

    
# Function to create event tsv files from run csvs
def create_main_event_files(RUN_DIR, DATA_DIR, SUBJECT_ID, RUN_NUM):
    
    # Load the csv files from the data directory
    event_df = pd.read_csv(os.path.join(RUN_DIR, f'{SUBJECT_ID}_run_{RUN_NUM}.csv'))
    
    # Create the event_df DataFrame with the required columns
    # Calculate base onset (start time of each trial)
    # Initial 5s delay, then cumulative sum of previous trial durations
    # All times seconds
    # The term (event_df['Trial Duration'].cumsum() - event_df['Trial Duration'])
    # gives the sum of durations of all preceding trials
        
    # Determine if Input or Output Modalities changed compared to the previous trial
    input_modality_changed = (event_df['Input Modality'] != event_df['Input Modality'].shift(1)).fillna(False)
    output_modality_changed = (event_df['Output Modality'] != event_df['Output Modality'].shift(1)).fillna(False)
    
    # A modality change event occurs if either input or output modality has changed
    modality_change_event = input_modality_changed | output_modality_changed

    # Ensure the first trial never has a modality change penalty
    if not modality_change_event.empty:
        modality_change_event.iloc[0] = False
    
    # Calculate base onset times (initial 5s delay + sum of previous trial durations)
    # All times seconds.
    # The term (event_df['Trial Duration'].cumsum() - event_df['Trial Duration'])
    # gives the sum of durations of all preceding trials.
    initial_delay = 5
    sum_prev_durations = (event_df['Trial Duration'].cumsum() - event_df['Trial Duration'])
    base_onset = initial_delay + sum_prev_durations
    
    # Define the rest period for modality change
    modality_rest = 15
    
    # Calculate cumulative penalty from modality changes.
    # A penalty of modality_rest_ms is applied for each trial where a modality change occurs.
    # This penalty is cumulative, affecting the onset of the current trial and all subsequent trials.
    penalties_for_change = modality_change_event.astype(int) * modality_rest
    cumulative_penalties = penalties_for_change.cumsum()
    
    # The final 'onset' is the base_onset plus the cumulative penalty from modality changes
    onset_times = base_onset + cumulative_penalties
    event_df['onset'] = onset_times
    event_df['duration'] = event_df['Trial Duration']
    #event_df['duration'] = 7.5 # Set a fixed duration of 7.5 seconds for all trials
    
    
    # Apply trial_type based on conditions
    trial_type_func = build_trial_type_list(event_df)
    event_df['trial_type'] = event_df.apply(trial_type_func, axis=1)
    
    # Limit the DataFrame to the required columns
    event_df = event_df[['onset', 'duration', 'trial_type']]

    # Print the DataFrame to show the new 'trial_type' column (and others)
    #print(event_df)
    
    # Define the output path for the tsv file
    output_path = os.path.join(DATA_DIR, f'{SUBJECT_ID}_run_{RUN_NUM}_events.tsv')
    
    # Save the DataFrame to a tsv file
    event_df.to_csv(output_path, sep='\t', index=False)
    #print(f"Event file created: {output_path}")
    
    
    

def create_visual_localizer_event_files(CSV_PATH, DATA_DIR, SUBJECT_ID):
    visual_stim_df_raw = pd.read_csv(CSV_PATH)
    
    # Create a new DataFrame with the required columns 
    visual_stim_df = pd.DataFrame()
    visual_stim_df['trial_type'] = visual_stim_df_raw['cond']
    
    # Set a fixed duration of 6 seconds for all trials
    visual_stim_df['duration'] = 6.0
    
    # Set the initial delay for the first trial
    initial_delay = 6.0
    
    # Set rest time between trials
    rest_time = 6.0
    
    # Initialize onset times list
    onset_times = [initial_delay] 

    # Calculate base onset times (initial 6s delay + cumulative sum of previous trial durations)
    for i in range(1, len(visual_stim_df)):
        # The next onset is the previous onset + previous duration
        next_onset = onset_times[i-1] + visual_stim_df.at[i-1, 'duration'] + rest_time
        onset_times.append(next_onset)
    
    visual_stim_df['onset'] = onset_times

    # Reorder columns
    visual_stim_df = visual_stim_df[['onset', 'duration', 'trial_type']]

    # Define the output path for the tsv file
    output_path = os.path.join(DATA_DIR, f'{SUBJECT_ID}_visual_localizer_events.tsv')
    
    # Save the DataFrame to a tsv file
    visual_stim_df.to_csv(output_path, sep='\t', index=False)

    print(f"Visual localizer event file created: {output_path}")


def create_auditory_localizer_event_files(CSV_PATH, DATA_DIR, SUBJECT_ID):
    auditory_stim_df_raw = pd.read_csv(CSV_PATH)
    
    # Create a new DataFrame with the required columns 
    auditory_stim_df = pd.DataFrame()
    auditory_stim_df['onset'] = auditory_stim_df_raw['onset']/ 1000  # Convert ms to seconds
    
    auditory_stim_df['trial_type'] = auditory_stim_df_raw['stim'].str[:-6]
    
    # Assign duration based on trial_type
    auditory_stim_df['duration'] = auditory_stim_df['trial_type'].apply(lambda x: 2.0 if x == 'scrambled_words' else 1.0)
    
    # Ensure the trial_type column is properly assigned and reorder if necessary
    # If 'onset', 'duration', 'trial_type' are the only columns desired in a specific order:
    auditory_stim_df = auditory_stim_df[['onset', 'duration', 'trial_type']]

    
    output_path = os.path.join(DATA_DIR, f'{SUBJECT_ID}_auditory_localizer_events.tsv')
    
    # Save the DataFrame to a tsv file
    auditory_stim_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Auditory localizer event file created: {output_path}")
    

def create_hand_localizer_event_files(CSV_PATH, DATA_DIR, SUBJECT_ID):
    hand_stim_df_raw = pd.read_csv(CSV_PATH)
    # Create a new DataFrame with the required columns
    hand_stim_df = pd.DataFrame()
    
    hand_stim_df['onset'] = hand_stim_df_raw['onset'] / 1000  # Convert ms to seconds
    hand_stim_df['trial_type'] = hand_stim_df_raw['stim'].str[:-4]  # Remove the last 4 characters
    hand_stim_df['type'] = hand_stim_df_raw['type']  # Keep the type column for calculating duratuib
    
    # Remove irrelevant rows
    hand_stim_df = hand_stim_df[hand_stim_df['type'] != 'blank']
    hand_stim_df = hand_stim_df[hand_stim_df['trial_type'] != 'write']
    
    # Rename all trial_type values that are not 'finger' to 'write' (i.e., words)
    hand_stim_df.loc[hand_stim_df['trial_type'] != 'finger', 'trial_type'] = 'write'
    
    # Assign duration based on trial_type
    # ATTN !! Intentionally including cue + fixation due to ambiguity in task design given to subjects
    # In the future it needs to be clear when the subject is to perform the task and when they are to just look at the screen
    # first 1 second is included to account for cue duration
    # Assign duration by row based on trial_type
    def assign_duration(row):
        if row['trial_type'] == 'finger':
            return 14.0
        elif row['trial_type'] == 'write':
            return 11.0

    
    hand_stim_df['duration'] = hand_stim_df.apply(assign_duration, axis=1)
    
    # Reorder columns to match the desired output
    hand_stim_df = hand_stim_df[['onset', 'duration', 'trial_type']]

    # Drop rows with missing trial_type, if any
    hand_stim_df = hand_stim_df.dropna(subset=['trial_type'])
    
    
    output_path = os.path.join(DATA_DIR, f'{SUBJECT_ID}_hand_localizer_events.tsv')
    
    # Save the DataFrame to a tsv file
    hand_stim_df.to_csv(output_path, sep='\t', index=False)
    print(f"Hand localizer event file created: {output_path}")


def create_speech_localizer_event_files(CSV_PATH, DATA_DIR, SUBJECT_ID):
    speech_stim_df_raw = pd.read_csv(CSV_PATH)
    # Create a new DataFrame with the required columns
    speech_stim_df = pd.DataFrame()
    
    speech_stim_df['onset'] = speech_stim_df_raw['onset'] / 1000  # Convert ms to seconds
    speech_stim_df['trial_type'] = speech_stim_df_raw['stim'].str[:-4]  # Remove the last 4 characters
    speech_stim_df['type'] = speech_stim_df_raw['type']  # Keep the type column for calculating duratuib
    
    # Remove irrelevant rows
    speech_stim_df = speech_stim_df[speech_stim_df['type'] != 'blank']
    speech_stim_df = speech_stim_df[speech_stim_df['trial_type'] != 'speech']
    
    # Rename all trial_type values that are not 'hum' to 'speech' (i.e., words)
    speech_stim_df.loc[speech_stim_df['trial_type'] != 'hum', 'trial_type'] = 'speech'
    
    # Assign duration based on trial_type
    # ATTN !! Intentionally including cue + fixation due to ambiguity in task design given to subjects
    # In the future it needs to be clear when the subject is to perform the task and when they are to just look at the screen
    # first 1 second is included to account for cue duration
    # Assign duration by row based on trial_type
    def assign_duration(row):
        if row['trial_type'] == 'hum':
            return 14.0
        elif row['trial_type'] == 'speech':
            return 5.0

    
    speech_stim_df['duration'] = speech_stim_df.apply(assign_duration, axis=1)
    
    # Reorder columns to match the desired output
    speech_stim_df = speech_stim_df[['onset', 'duration', 'trial_type']]

    # Drop rows with missing trial_type, if any
    speech_stim_df = speech_stim_df.dropna(subset=['trial_type'])
    
    
    output_path = os.path.join(DATA_DIR, f'{SUBJECT_ID}_speech_localizer_events.tsv')
    
    # Save the DataFrame to a tsv file
    speech_stim_df.to_csv(output_path, sep='\t', index=False)
    print(f"Speech localizer event file created: {output_path}")

# Run functions as desired


# RUN_DIR = r"/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/fMRI_Data_Analysis/run_csvs/SWP_Pilot_1/SWP_Pilot_1/main-exp/"
# DATA_DIR = r"/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/fMRI_Data_Analysis/event_tsvs"

# for i in range(1, 7):
#     create_main_event_files(Path(RUN_DIR), Path(DATA_DIR), SUBJECT_ID = 'sub01', RUN_NUM = str(i))




# create_visual_localizer_event_files(
#     CSV_PATH = r"/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/SWP_Pilot_1/localizer/visual_categories/sub1_vis.csv",
#     DATA_DIR = r"/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/fMRI_Data_Analysis/event_tsvs",
#     SUBJECT_ID = 'sub1'
# )

# create_auditory_localizer_event_files(
#     CSV_PATH = r"/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/SWP_Pilot_1/localizer/audio_categories/sub1_audio.csv",
#     DATA_DIR = r"/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/fMRI_Data_Analysis/event_tsvs",
#     SUBJECT_ID = 'sub1'
# )

# create_hand_localizer_event_files(
#     CSV_PATH = "/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/SWP_Pilot_1/localizer/hand_categories/sub1_hand.csv",
#     DATA_DIR = "/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/fMRI_Data_Analysis/event_tsvs",
#     SUBJECT_ID = 'sub1'
# )


# create_speech_localizer_event_files(
#     CSV_PATH = "/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/SWP_Pilot_1/localizer/speech_categories/sub1_speech.csv",
#     DATA_DIR = "/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/fMRI_Data_Analysis/event_tsvs",
#     SUBJECT_ID = 'sub1'
# )

