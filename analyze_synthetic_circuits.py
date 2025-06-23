#!/usr/bin/env python3
"""
Analyze Synthetic Circuit Dataset with Symbolic Equation Extraction

This script applies lcapy symbolic analysis to the synthetically generated 
circuit dataset, extracting symbolic equations and analyzing the results.

Author: Assistant
Date: 2024
"""

import json
import re
from pathlib import Path
import lcapy
from lcapy import Circuit
import sympy as sp
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm

def clean_spice_for_lcapy(spice_netlist: str) -> str:
    """
    Clean SPICE netlist to make it compatible with lcapy.
    
    Args:
        spice_netlist: Raw SPICE netlist string
        
    Returns:
        Cleaned netlist string for lcapy
    """
    lines = spice_netlist.strip().split('\n')
    cleaned_lines = []
    
    def convert_spice_value(value_str):
        """Convert SPICE units to numeric values."""
        if not value_str or value_str == '0':
            return value_str
            
        # Handle unit multipliers
        multipliers = {
            'k': 1e3, 'K': 1e3,
            'm': 1e-3, 'M': 1e-3, 
            'u': 1e-6, 'μ': 1e-6,
            'n': 1e-9, 'N': 1e-9,
            'p': 1e-12, 'P': 1e-12,
            'f': 1e-15, 'F': 1e-15
        }
        
        for unit, mult in multipliers.items():
            if value_str.endswith(unit):
                try:
                    base_value = float(value_str[:-1])
                    return str(base_value * mult)
                except ValueError:
                    continue
        
        return value_str
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines, comments, and SPICE commands
        if (not line or line.startswith('*') or line.startswith('.') or 
            line.startswith('print') or ';' in line):
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
            
        component_name = parts[0]
        first_char = component_name[0].upper()
        
        # Convert voltage measurement sources (VI) to 0V sources
        if component_name.startswith('VI'):
            if len(parts) >= 4:
                # VI1 node1 node2 0 -> V_meas1 node1 node2 0
                name = f"V_meas{component_name[2:]}"
                node1, node2 = parts[1], parts[2]
                cleaned_lines.append(f"{name} {node1} {node2} 0")
        
        # Handle basic components (R, L, C, V, I)
        elif first_char in ['R', 'L', 'C', 'V', 'I']:
            if len(parts) >= 4:
                name = parts[0]
                node1, node2 = parts[1], parts[2]
                value = convert_spice_value(parts[3])
                
                # Clean node names - replace problematic characters
                node1 = re.sub(r'[^\w]', '_', node1)
                node2 = re.sub(r'[^\w]', '_', node2)
                
                cleaned_lines.append(f"{name} {node1} {node2} {value}")
        
        # Skip controlled sources for now (E, F, G, H) as they're complex
        elif first_char in ['E', 'F', 'G', 'H']:
            print(f"Skipping controlled source: {component_name}")
            continue
    
    return '\n'.join(cleaned_lines)

def extract_symbolic_equations_from_circuit(spice_netlist: str, circuit_id: str) -> Dict:
    """
    Extract symbolic equations from a single circuit using lcapy.
    
    Args:
        spice_netlist: SPICE netlist string
        circuit_id: Circuit identifier
        
    Returns:
        Dictionary containing symbolic analysis results
    """
    try:
        # Clean the netlist for lcapy
        cleaned_netlist = clean_spice_for_lcapy(spice_netlist)
        
        if not cleaned_netlist.strip():
            return {
                'circuit_id': circuit_id,
                'status': 'error',
                'error': 'Empty netlist after cleaning',
                'original_netlist': spice_netlist
            }
        
        # Create lcapy circuit
        cct = Circuit(cleaned_netlist)
        
        results = {
            'circuit_id': circuit_id,
            'status': 'success',
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist,
            'symbolic_equations': {},
            'circuit_properties': {
                'nodes': str(cct.node_list),
                'elements': str(list(cct.elements.keys())),
                'num_nodes': len(cct.node_list),
                'num_elements': len(cct.elements)
            }
        }
        
        # 1. NODAL ANALYSIS EQUATIONS
        try:
            na = cct.nodal_analysis()
            nodal_eqs = na.nodal_equations()
            
            results['symbolic_equations']['nodal'] = {
                'equations': {str(k): str(v) for k, v in nodal_eqs.items()},
                'A_matrix': str(na.A),
                'b_vector': str(na.b),
                'unknowns': str(na.unknowns),
                'description': 'Nodal analysis: A(s) * V(s) = B(s)'
            }
        except Exception as e:
            results['symbolic_equations']['nodal'] = {'error': str(e)}
        
        # 2. COMPONENT EQUATIONS IN S-DOMAIN
        try:
            component_eqs = {}
            for name, component in cct.elements.items():
                try:
                    V_s = component.V(lcapy.s)
                    I_s = component.I(lcapy.s)
                    component_eqs[name] = {
                        'voltage_s': str(V_s),
                        'current_s': str(I_s)
                    }
                except Exception as e:
                    component_eqs[name] = {'error': str(e)}
            
            results['symbolic_equations']['components'] = component_eqs
        except Exception as e:
            results['symbolic_equations']['components'] = {'error': str(e)}
        
        # 3. NODE VOLTAGE EXPRESSIONS
        try:
            node_voltages = {}
            for node_name in cct.node_list:
                if node_name != '0':  # Skip ground
                    try:
                        V_node_s = cct[node_name].V(lcapy.s)
                        node_voltages[f'V_{node_name}'] = str(V_node_s)
                    except Exception as e:
                        node_voltages[f'V_{node_name}'] = str(e)
            
            results['symbolic_equations']['node_voltages'] = node_voltages
        except Exception as e:
            results['symbolic_equations']['node_voltages'] = {'error': str(e)}
        
        # 4. TRANSFER FUNCTIONS
        try:
            transfer_functions = {}
            nodes = [n for n in cct.node_list if n != '0']
            
            # Get transfer functions between first few pairs of nodes
            for i, input_node in enumerate(nodes[:3]):  # Limit to avoid too many
                for output_node in nodes[i+1:i+3]:
                    try:
                        H_s = cct.transfer(input_node, 0, output_node, 0)
                        tf_name = f'H_{input_node}_to_{output_node}'
                        transfer_functions[tf_name] = str(H_s)
                    except Exception as e:
                        transfer_functions[f'H_{input_node}_to_{output_node}'] = f'Error: {str(e)}'
            
            results['symbolic_equations']['transfer_functions'] = transfer_functions
        except Exception as e:
            results['symbolic_equations']['transfer_functions'] = {'error': str(e)}
        
        # 5. INPUT IMPEDANCES
        try:
            impedances = {}
            nodes = [n for n in cct.node_list if n != '0']
            
            for node in nodes[:5]:  # Limit to first 5 nodes
                try:
                    Z_s = cct.impedance(node, 0)
                    impedances[f'Z_{node}'] = str(Z_s)
                except Exception as e:
                    impedances[f'Z_{node}'] = str(e)
            
            results['symbolic_equations']['impedances'] = impedances
        except Exception as e:
            results['symbolic_equations']['impedances'] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        return {
            'circuit_id': circuit_id,
            'status': 'error',
            'error': str(e),
            'original_netlist': spice_netlist
        }

def analyze_synthetic_dataset(labels_file: str, output_file: str = None, max_circuits: int = None) -> Dict:
    """
    Analyze the synthetic circuit dataset with symbolic equation extraction.
    
    Args:
        labels_file: Path to labels.json file
        output_file: Path to save results
        max_circuits: Maximum number of circuits to analyze
        
    Returns:
        Dictionary containing all analysis results
    """
    print(f"Loading synthetic circuit dataset from {labels_file}...")
    
    with open(labels_file, 'r') as f:
        dataset = json.load(f)
    
    circuit_ids = list(dataset.keys())
    if max_circuits:
        circuit_ids = circuit_ids[:max_circuits]
    
    print(f"Analyzing {len(circuit_ids)} circuits with symbolic equation extraction...")
    
    results = {
        'metadata': {
            'dataset_file': labels_file,
            'total_circuits_in_dataset': len(dataset),
            'circuits_analyzed': len(circuit_ids),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'analysis_summary': {}
        },
        'circuit_analyses': {}
    }
    
    successful_analyses = []
    failed_analyses = []
    
    # Analyze each circuit
    for i, circuit_id in enumerate(tqdm(circuit_ids, desc="Analyzing circuits")):
        spice_netlist = dataset[circuit_id]
        
        print(f"\n{'='*60}")
        print(f"Analyzing circuit {i+1}/{len(circuit_ids)}: {circuit_id}")
        print(f"{'='*60}")
        
        analysis = extract_symbolic_equations_from_circuit(spice_netlist, circuit_id)
        results['circuit_analyses'][circuit_id] = analysis
        
        if analysis['status'] == 'success':
            successful_analyses.append(circuit_id)
            print(f"✅ SUCCESS: Extracted symbolic equations")
            
            # Print some key results
            if 'nodal' in analysis['symbolic_equations']:
                nodal = analysis['symbolic_equations']['nodal']
                if 'equations' in nodal:
                    print(f"   📐 Nodal equations: {len(nodal['equations'])} nodes")
            
            if 'components' in analysis['symbolic_equations']:
                comp = analysis['symbolic_equations']['components']
                if isinstance(comp, dict) and 'error' not in comp:
                    print(f"   🔧 Component equations: {len(comp)} components")
            
            if 'transfer_functions' in analysis['symbolic_equations']:
                tf = analysis['symbolic_equations']['transfer_functions']
                if isinstance(tf, dict) and 'error' not in tf:
                    print(f"   📊 Transfer functions: {len(tf)} functions")
        else:
            failed_analyses.append(circuit_id)
            print(f"❌ FAILED: {analysis.get('error', 'Unknown error')}")
    
    # Update metadata
    results['metadata']['successful_analyses'] = len(successful_analyses)
    results['metadata']['failed_analyses'] = len(failed_analyses)
    results['metadata']['success_rate'] = len(successful_analyses) / len(circuit_ids) * 100
    
    # Create analysis summary
    results['metadata']['analysis_summary'] = {
        'successful_circuit_ids': successful_analyses,
        'failed_circuit_ids': failed_analyses,
        'common_failure_reasons': analyze_failure_patterns(results['circuit_analyses'])
    }
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SYMBOLIC ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total circuits analyzed: {len(circuit_ids)}")
    print(f"Successful analyses: {len(successful_analyses)} ({len(successful_analyses)/len(circuit_ids)*100:.1f}%)")
    print(f"Failed analyses: {len(failed_analyses)} ({len(failed_analyses)/len(circuit_ids)*100:.1f}%)")
    
    if successful_analyses:
        print(f"\n✅ Successful circuits: {successful_analyses[:10]}{'...' if len(successful_analyses) > 10 else ''}")
    
    if failed_analyses:
        print(f"\n❌ Failed circuits: {failed_analyses[:10]}{'...' if len(failed_analyses) > 10 else ''}")
    
    return results

def analyze_failure_patterns(circuit_analyses: Dict) -> List[str]:
    """Analyze common failure patterns in the analyses."""
    failure_reasons = []
    
    for circuit_id, analysis in circuit_analyses.items():
        if analysis['status'] == 'error':
            error_msg = analysis.get('error', '')
            failure_reasons.append(error_msg)
    
    # Count common errors
    from collections import Counter
    error_counts = Counter(failure_reasons)
    
    return [f"{error}: {count} occurrences" for error, count in error_counts.most_common(5)]

def demonstrate_successful_analysis(results: Dict):
    """Demonstrate a successful symbolic analysis."""
    successful_circuits = [
        cid for cid, analysis in results['circuit_analyses'].items()
        if analysis['status'] == 'success'
    ]
    
    if not successful_circuits:
        print("No successful analyses to demonstrate.")
        return
    
    # Pick the first successful circuit
    example_id = successful_circuits[0]
    example = results['circuit_analyses'][example_id]
    
    print(f"\n{'='*80}")
    print(f"EXAMPLE SYMBOLIC ANALYSIS: {example_id}")
    print(f"{'='*80}")
    
    print(f"\nOriginal SPICE netlist:")
    print(example['original_netlist'])
    
    print(f"\nCleaned netlist for lcapy:")
    print(example['cleaned_netlist'])
    
    print(f"\nCircuit properties:")
    props = example['circuit_properties']
    print(f"  Nodes: {props['nodes']}")
    print(f"  Elements: {props['elements']}")
    
    print(f"\nSymbolic equations extracted:")
    
    # Show nodal equations
    if 'nodal' in example['symbolic_equations']:
        nodal = example['symbolic_equations']['nodal']
        if 'equations' in nodal:
            print(f"\n📐 Nodal Analysis Equations:")
            for node, eq in nodal['equations'].items():
                print(f"  Node {node}: {eq}")
    
    # Show some component equations
    if 'components' in example['symbolic_equations']:
        comp = example['symbolic_equations']['components']
        if isinstance(comp, dict) and 'error' not in comp:
            print(f"\n🔧 Component Equations (first 3):")
            for i, (name, eqs) in enumerate(list(comp.items())[:3]):
                if 'voltage_s' in eqs:
                    print(f"  {name}: V(s) = {eqs['voltage_s']}")
                    print(f"        I(s) = {eqs['current_s']}")
    
    # Show transfer functions
    if 'transfer_functions' in example['symbolic_equations']:
        tf = example['symbolic_equations']['transfer_functions']
        if isinstance(tf, dict) and 'error' not in tf:
            print(f"\n📊 Transfer Functions:")
            for name, func in tf.items():
                if not func.startswith('Error'):
                    print(f"  {name}: {func}")

def main():
    """Main function to run symbolic analysis on synthetic dataset."""
    print("Symbolic Equation Extraction from Synthetic Circuit Dataset")
    print("="*80)
    
    # Dataset path
    labels_file = "datasets/grid_v11_240831/labels.json"
    output_file = "synthetic_circuits_symbolic_analysis.json"
    
    if not Path(labels_file).exists():
        print(f"❌ Dataset file not found: {labels_file}")
        print("Please ensure the dataset is in the correct location.")
        return
    
    # Start with a subset for testing, then expand
    print("Starting symbolic analysis on synthetic circuits...")
    print("(Starting with first 10 circuits for testing)\n")
    
    results = analyze_synthetic_dataset(
        labels_file=labels_file,
        output_file=output_file,
        max_circuits=10  # Start with 10 circuits
    )
    
    # Demonstrate a successful analysis
    demonstrate_successful_analysis(results)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main() 