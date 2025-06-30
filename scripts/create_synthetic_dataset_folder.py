#!/usr/bin/env python3
"""
Script to create synthetic dataset folder structure from circuit QA dataset JSON.
Creates folders q1, q2, q3, ... with corresponding image, question, answer, and netlist files.
"""

import json
import os
import shutil
from pathlib import Path

def create_synthetic_dataset():
    # Load the JSON data
    with open('circuit_qa_dataset.json', 'r') as f:
        data = json.load(f)
    
    # Create the main synthetic_dataset folder
    dataset_dir = Path('../synthetic_dataset')
    dataset_dir.mkdir(exist_ok=True)
    
    # Process each question in the dataset
    questions = data['questions']
    
    for i, question_data in enumerate(questions, 1):
        # Create question folder (q1, q2, q3, ...)
        question_folder = dataset_dir / f'q{i}'
        question_folder.mkdir(exist_ok=True)
        
        # Extract data from JSON
        question_id = question_data['question_id']
        circuit_id = question_data['circuit_id']
        question_text = question_data['question']
        answer = question_data.get('answer_formatted', 'Simulation failed')
        
        # Create question file
        question_file = question_folder / f'q{i}_question.txt'
        with open(question_file, 'w') as f:
            f.write(question_text)
        
        # Create answer file
        answer_file = question_folder / f'q{i}_ta.txt'
        with open(answer_file, 'w') as f:
            f.write(str(answer))
        
        # Create netlist file
        original_netlist = question_data.get('original_netlist', '')
        if original_netlist:
            netlist_file = question_folder / f'q{i}_netlist.txt'
            with open(netlist_file, 'w') as f:
                f.write(original_netlist)
        else:
            print(f"WARNING: No original_netlist found for q{i}")
        
        # Find and copy the corresponding image
        # Extract circuit part from question_id (e.g., "1_2_q1" -> "1_2")
        circuit_part = '_'.join(question_id.split('_')[:-1])  # Remove the "_q{N}" part
        source_image = Path(f'../datasets/first_batch/{circuit_part}.jpg')
        target_image = question_folder / f'q{i}_image.png'
        
        if source_image.exists():
            # Copy and rename to .png
            shutil.copy2(source_image, target_image)
            print(f"Created q{i}: {question_id} ({circuit_part}.jpg) -> {question_text[:50]}...")
        else:
            print(f"WARNING: Image not found for q{i} (circuit_part: {circuit_part})")
    
    print(f"\nDataset creation complete!")
    print(f"Total questions processed: {len(questions)}")
    print(f"Synthetic dataset created in: {dataset_dir.absolute()}")

if __name__ == '__main__':
    create_synthetic_dataset()