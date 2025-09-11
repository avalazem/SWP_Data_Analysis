#!/bin/bash

# Define the complete list of contrast rules
contrasts=(
    # audio > visual
    "audio > visual"
    "audio > visual | speech"
    "audio > visual | write"
    
    # speech > write
    "speech > write"
    "speech > write | audio"
    "speech > write | visual"
    
    # long > short
    "long > short"
    "long > short | audio"
    "long > short | visual"
    "long > short | speech"
    "long > short | write"
    "long > short | audio | speech"
    "long > short | audio | write"
    "long > short | visual | speech"
    "long > short | visual | write"
    
    # real > pseudo
    "real > pseudo"
    "real > pseudo | audio"
    "real > pseudo | visual"
    "real > pseudo | speech"
    "real > pseudo | write"
    "real > pseudo | audio | speech"
    "real > pseudo | audio | write"
    "real > pseudo | visual | speech"
    "real > pseudo | visual | write"
    
    # complex > simple
    "complex > simple"
    "complex > simple | audio"
    "complex > simple | visual"
    "complex > simple | speech"
    "complex > simple | write"
    "complex > simple | audio | speech"
    "complex > simple | audio | write"
    "complex > simple | visual | speech"
    "complex > simple | visual | write"
    
    # high > low | real
    "high > low | real | audio"
    "high > low | real | visual"
    "high > low | real | audio | speech"
    "high > low | real | audio | write"
    "high > low | real | visual | speech"
    "high > low | real | visual | write"
)

# Loop through the array and run the analysis for each contrast
for contrast_name in "${contrasts[@]}"; do
    echo "Running analysis for contrast: $contrast_name"
    
    # Execute the Python script with the --contrast-name argument
    python3 main_fMRI_analysis.py --contrast-name "$contrast_name"
    
    # Add a separator for clarity in the output
    echo "----------------------------------------"
done

echo "✅ All analyses complete."
