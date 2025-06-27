#!/bin/bash
# This script runs nilearn analysis for the SWP dataset with specified parameters.

# Define the alpha values
alpha_values=(0.05 0.1 0.15 0.2)

# Loop over each alpha value
for alpha_val in "${alpha_values[@]}"; do
    echo "----------------------------------------------------"
    echo "Running all analyses for alpha = $alpha_val"
    echo "----------------------------------------------------"

    # Command 1: SWP task
    python3 swp_nilearn-analysis.py --subject 3 --task swp --num-runs 6 --contrast-name real_gt_pseudo --alpha "$alpha_val"

    # Command 2: locvis task
    #python3 swp_nilearn-analysis.py --subject 2 --task locvis --contrast-name locvis:language_vs_non_language --alpha "$alpha_val"

    # Command 3: locaudio task
    #python3 swp_nilearn-analysis.py --subject 2 --task locaudio --contrast-name locaud:words_vs_scrambledwords --alpha "$alpha_val"

    # Command 4: lochand task
    #python3 swp_nilearn-analysis.py --subject 3 --task lochand --contrast-name lochand:writing_vs_rest --alpha "$alpha_val"

    # Command 5: locspeech task
    #python3 swp_nilearn-analysis.py --subject 3 --task locspeech --contrast-name locspeech:speech_vs_rest --alpha "$alpha_val"

    echo "" # Add an empty line for better readability between alpha runs
done

echo "All analysis runs completed."