#!/bin/zsh
set -x
# This script runs nilearn analysis for the SWP dataset with specified parameters.

# ==============================================================================
# Define the parameters to loop over
# ==============================================================================
# Define the alpha values
alpha_values=(0.05 0.1)

# Define the subjects to loop over
subjects=(1 2 3) # Add all subjects here

# Define the tasks and their corresponding parameters
tasks_info=(
    "swp,6,long_vs_short_vis"
    "swp,6,viz_gt_aud"
    "locvis,0,locvis:words_vs_rest"
    "locaudio,0,locaud:words_vs_rest" # Assumed num_runs is 0, adjust if needed
    "lochand,0,lochand:writing_vs_rest" # Assumed num_runs is 0, adjust if needed
    "locspeech,0,locspeech:speech_vs_rest" # Assumed num_runs is 0, adjust if needed
)
# Note: For tasks other than 'swp', the number of runs is set to 0.
# You can modify the tasks_info array to be more flexible if needed.
# For example: "task_name,num_runs,contrast_name"


# ==============================================================================
# Loop over each parameter
# ==============================================================================
# Loop over each subject
for subject in "${subjects[@]}"; do

    # Loop over each task and its parameters
    for task_info in "${tasks_info[@]}"; do
        # Use Zsh's built-in string splitting to parse the task_info string
        # This splits the string at the comma and puts the parts into an array
        IFS=',' read -r task_name num_runs contrast_name <<< "$task_info"

        # Loop over each alpha value
        for alpha_val in "${alpha_values[@]}"; do
            echo "----------------------------------------------------"
            echo "Running analysis for Subject: $subject, Task: $task_name, Alpha: $alpha_val"
            echo "----------------------------------------------------"

            # The command to run the python script with all the variables
            python3 swp_nilearn-analysis.py \
                --subject "$subject" \
                --task "$task_name" \
                --num-runs "$num_runs" \
                --contrast-name "$contrast_name" \
                --alpha "$alpha_val"

            echo "" # Add an empty line for better readability
        done
    done
done

echo "All analysis runs completed."
