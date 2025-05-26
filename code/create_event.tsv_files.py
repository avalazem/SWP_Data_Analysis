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
    
    
    
    
RUN_DIR = r"/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/fMRI_Data_Analysis/run_csvs/SWP_Pilot_1/SWP_Pilot_1/main-exp/"
DATA_DIR = r"/home/avalazem/Desktop/Work/Single_Word_Processing_Stage/fMRI_Data_Analysis/event_tsvs"

for i in range(1, 7):
    create_main_event_files(Path(RUN_DIR), Path(DATA_DIR), SUBJECT_ID = 'sub01', RUN_NUM = str(i))



def create_visual_localizer_event_files(CSV_PATH, DATA_DIR, SUBJECT_ID):
    visual_stim_df = pd.read_csv(CSV_PATH)
    
    visual_stim_df['cond']
    