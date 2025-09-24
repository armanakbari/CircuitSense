#!/usr/bin/env python3
\
\
\
   

import json
import os
import shutil
from pathlib import Path
import re

def clean_transfer_function_name(tf_name):
                                                                           
                                    
    return tf_name.replace('_to_', ' to ').replace('_', ' ')

def create_question_text(circuit_id, tf_name, tf_equation, netlist):
                                                                       
    clean_tf = clean_transfer_function_name(tf_name)
    
    question = f"""
Task: Derive the transfer function from {clean_tf}.

Please provide a complete step-by-step derivation showing:
1. The circuit analysis method used (nodal analysis, mesh analysis, etc.)
2. All intermediate equations
3. The algebraic manipulation steps
4. The final transfer function expression

The transfer function should be expressed in the s-domain (Laplace domain) showing the relationship between the output and input.
"""
    return question

def create_answer_text(circuit_id, tf_name, tf_equation, nodal_equations):
                                                                 
    clean_tf = clean_transfer_function_name(tf_name)
    
    answer = tf_equation

    return answer

def process_circuit_dataset(json_file_path, image_base_path, output_dir):
\
\
\
\
\
\
\
       
    
                        
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
                             
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    question_count = 1
    processed_count = 0
    skipped_count = 0
    
    print(f"Processing {data['summary']['total_circuits']} circuits...")
    print(f"Expected successful circuits: {data['summary']['successful']}")
    
    for result in data['results']:
        circuit_id = result['circuit_id']
        
                                                      
        if 'transfer_functions' not in result or not result['transfer_functions']:
            print(f"Skipping {circuit_id}: No transfer functions")
            skipped_count += 1
            continue
            
                                                                
        if 'nodal_equations' not in result:
            print(f"Skipping {circuit_id}: No nodal equations")
            skipped_count += 1
            continue
            
        nodal_eq = result['nodal_equations']
                                                                          
        if not nodal_eq.get('t_domain') or nodal_eq.get('t_domain') == 'SKIPPED':
            print(f"Skipping {circuit_id}: Time domain equations were skipped")
            skipped_count += 1
            continue
        
                                                         
        error_states = ['TIMEOUT_OR_ERROR', 'SKIPPED_TOO_COMPLEX', 'SKIPPED_NO_TRANSFER_FUNCTIONS']
        for tf_name, tf_equation in result['transfer_functions'].items():
                                                       
            if tf_equation in error_states:
                print(f"Skipping {circuit_id} - {tf_name}: {tf_equation}")
                skipped_count += 1
                continue
                                    
            question_folder = output_path / f"q{question_count}"
            question_folder.mkdir(exist_ok=True)
            
                                    
            question_text = create_question_text(
                circuit_id, tf_name, tf_equation, result['cleaned_netlist']
            )
            
                                  
            answer_text = create_answer_text(
                circuit_id, tf_name, tf_equation, result['nodal_equations']
            )
            
                                 
            question_file = question_folder / f"q{question_count}_question.txt"
            with open(question_file, 'w') as f:
                f.write(question_text)
            
                               
            answer_file = question_folder / f"q{question_count}_ta.txt"
            with open(answer_file, 'w') as f:
                f.write(answer_text)
            
                                           
            source_image = Path(image_base_path) / f"{circuit_id}.jpg"
            target_image = question_folder / f"q{question_count}_image.jpg"
            
            if source_image.exists():
                shutil.copy2(source_image, target_image)
                print(f"Created question {question_count}: {circuit_id} - {tf_name}")
            else:
                print(f"Warning: Image not found for {circuit_id} at {source_image}")
                                                                     
                with open(question_folder / f"q{question_count}_image_missing.txt", 'w') as f:
                    f.write(f"Image file {circuit_id}.jpg not found in {image_base_path}")
            
            processed_count += 1
            question_count += 1
    
    print(f"\nDataset creation completed!")
    print(f"Total questions created: {processed_count}")
    print(f"Circuits skipped: {skipped_count}")
    print(f"Output directory: {output_path.absolute()}")

def main():
                                                    
                                                  
    json_file = "datasets/equations_2/symbolic_equations_complete.json"
    image_base = "datasets/equations_2"                                             
    output_directory = "equation_derivation_full"
    
                                
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found: {json_file}")
        return
    
    if not os.path.exists(image_base):
        print(f"Error: Image directory not found: {image_base}")
        return
    
                         
    process_circuit_dataset(json_file, image_base, output_directory)

if __name__ == "__main__":
    main() 