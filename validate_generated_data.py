#!/usr/bin/env python3

import os
import json
import sys
from tqdm import tqdm
from ppm_construction.ft_vlm.evaluate_spice import process_single_simulation, run_process_with_timeout

def validate_circuit_dataset(labels_file, output_file=None, timeout=10):
    """
    Validate generated synthetic circuit dataset by running SPICE simulation
    
    Args:
        labels_file: Path to labels.json file
        output_file: Path to save validation results (optional)
        timeout: Simulation timeout in seconds
    """
    
    # Load the labels
    with open(labels_file, 'r') as f:
        labels_data = json.load(f)
    
    print(f"Loaded {len(labels_data)} circuits from {labels_file}")
    
    results = []
    valid_count = 0
    invalid_count = 0
    
    for qid, spice_code in tqdm(labels_data.items()):
        print(f"\n{'='*50}")
        print(f"Validating circuit: {qid}")
        print(f"{'='*50}")
        
        # Run simulation with timeout
        try:
            result = run_process_with_timeout(
                func=process_single_simulation, 
                args=(spice_code, qid), 
                timeout=timeout
            )
            
            is_valid = result.get('valid', False)
            has_zero_resistor = result.get('has_zero_resistor', False)
            sim_ret = result.get('sim_ret', {})
            
            # Use the same validation logic as evaluate_spice.py
            simulation_successful = (
                is_valid and 
                'error' not in sim_ret and 
                'raw_file' in sim_ret and 
                'Simulation interrupted due to error' not in sim_ret.get('raw_file', '')
            )
            
            if simulation_successful:
                valid_count += 1
                status = "✅ VALID"
            else:
                invalid_count += 1
                status = "❌ INVALID"
                
                # Print detailed error information
                if 'raw_file' in sim_ret and 'Simulation interrupted due to error' in sim_ret['raw_file']:
                    print(f"🚨 NgSpice Error Detected in raw_file")
                if 'error' in sim_ret:
                    print(f"🚨 Error in sim_ret: {sim_ret['error']}")
                
            print(f"Status: {status}")
            print(f"Has zero resistor: {has_zero_resistor}")
            print(f"Simulation results: {sim_ret}")
            
            results.append({
                'qid': qid,
                'spice_code': spice_code,
                'valid': is_valid,
                'has_zero_resistor': has_zero_resistor,
                'simulation_results': sim_ret,
                'status': status
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            invalid_count += 1
            results.append({
                'qid': qid,
                'spice_code': spice_code,
                'valid': False,
                'error': str(e),
                'status': "❌ ERROR"
            })
    
    # Summary
    total = len(labels_data)
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total circuits: {total}")
    print(f"Valid circuits: {valid_count} ({valid_count/total*100:.1f}%)")
    print(f"Invalid circuits: {invalid_count} ({invalid_count/total*100:.1f}%)")
    
    # Categorize issues
    zero_resistor_count = sum(1 for r in results if r.get('has_zero_resistor', False))
    error_count = sum(1 for r in results if 'error' in r)
    
    print(f"\nIssue breakdown:")
    print(f"- Circuits with zero resistors: {zero_resistor_count}")
    print(f"- Circuits with simulation errors: {error_count}")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nValidation results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    # Default paths
    labels_file = "datasets/grid_v11_240831/labels.json"
    output_file = "datasets/grid_v11_240831/validation_results.json"
    
    if len(sys.argv) > 1:
        labels_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("🔍 Starting validation of generated circuit dataset...")
    print(f"Input: {labels_file}")
    print(f"Output: {output_file}")
    
    results = validate_circuit_dataset(labels_file, output_file)
    
    print("\n✨ Validation complete!") 