#!/usr/bin/env python3
\
\
\
   

import json
import os
import shutil
from pathlib import Path

def create_synthetic_dataset():
                        
    with open('circuit_qa_dataset.json', 'r') as f:
        data = json.load(f)
    
                                              
    dataset_dir = Path('../synthetic_dataset')
    dataset_dir.mkdir(exist_ok=True)
    
                                          
    questions = data['questions']
    
    for i, question_data in enumerate(questions, 1):
                                                  
        question_folder = dataset_dir / f'q{i}'
        question_folder.mkdir(exist_ok=True)
        
                                
        question_id = question_data['question_id']
        circuit_id = question_data['circuit_id']
        question_text = question_data['question']
        answer = question_data.get('answer_formatted', 'Simulation failed')
        
                              
        question_file = question_folder / f'q{i}_question.txt'
        with open(question_file, 'w') as f:
            f.write(question_text)
        
                            
        answer_file = question_folder / f'q{i}_ta.txt'
        with open(answer_file, 'w') as f:
            f.write(str(answer))
        
                             
        original_netlist = question_data.get('original_netlist', '')
        if original_netlist:
            netlist_file = question_folder / f'q{i}_netlist.txt'
            with open(netlist_file, 'w') as f:
                f.write(original_netlist)
        else:
            print(f"WARNING: No original_netlist found for q{i}")
        
                                               
                                                                         
        circuit_part = '_'.join(question_id.split('_')[:-1])                           
        source_image = Path(f'../datasets/first_batch/{circuit_part}.jpg')
        target_image = question_folder / f'q{i}_image.png'
        
        if source_image.exists():
                                     
            shutil.copy2(source_image, target_image)
            print(f"Created q{i}: {question_id} ({circuit_part}.jpg) -> {question_text[:50]}...")
        else:
            print(f"WARNING: Image not found for q{i} (circuit_part: {circuit_part})")
    
    print(f"\nDataset creation complete!")
    print(f"Total questions processed: {len(questions)}")
    print(f"Synthetic dataset created in: {dataset_dir.absolute()}")

if __name__ == '__main__':
    create_synthetic_dataset()