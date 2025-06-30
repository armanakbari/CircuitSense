#!/usr/bin/env python3
"""
Script to add netlist files to existing synthetic dataset folders.
Extracts circuit_netlist from circuit_qa_dataset.json and creates q{N}_netlist.txt files.
"""

import json
import os
from pathlib import Path

def add_netlist_files():
    # Load the JSON data
    json_file = Path('circuit_qa_dataset.json')
    if not json_file.exists():
        print(f"ERROR: {json_file} not found in current directory")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Path to the synthetic dataset folder
    dataset_dir = Path('../synthetic_dataset_2')
    if not dataset_dir.exists():
        print(f"ERROR: {dataset_dir} not found")
        return
    
    # Process each question in the dataset
    questions = data['questions']
    
    for i, question_data in enumerate(questions, 1):
        # Check if the question folder exists
        question_folder = dataset_dir / f'q{i}'
        if not question_folder.exists():
            print(f"WARNING: Folder q{i} not found, skipping...")
            continue
        
        # Extract circuit netlist from JSON
        circuit_netlist = question_data.get('original_netlist', '')
        
        if not circuit_netlist:
            print(f"WARNING: No circuit_netlist found for q{i}, skipping...")
            continue
        
        # Create netlist file
        netlist_file = question_folder / f'q{i}_netlist.txt'
        with open(netlist_file, 'w') as f:
            f.write(circuit_netlist)
        
        print(f"Created netlist file for q{i}")
    
    print(f"\nNetlist file creation complete!")
    print(f"Total questions processed: {len(questions)}")

if __name__ == '__main__':
    add_netlist_files() 