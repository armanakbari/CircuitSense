#!/usr/bin/env python3
"""
Lcapy Circuit Analysis Script

This script uses the Lcapy Python library to perform symbolic analysis of circuit netlists.
It converts SPICE netlists to Lcapy-compatible format and extracts symbolic equations.

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

def clean_netlist_for_lcapy(netlist_str: str) -> str:
    """
    Convert SPICE netlist to Lcapy-compatible format.
    
    Lcapy requirements:
    1. Component names must be unique
    2. Node names should be alphanumeric or underscore
    3. Remove SPICE-specific commands (.control, .endc, .end)
    4. Convert measurement voltage sources (VI) to regular voltage sources
    5. Handle node naming conflicts
    
    Args:
        netlist_str: Original SPICE netlist string
        
    Returns:
        Cleaned netlist string compatible with Lcapy
    """
    lines = netlist_str.strip().split('\n')
    cleaned_lines = []
    node_counter = 1000  # Start high to avoid conflicts
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and SPICE control commands
        if not line or line.startswith('.control') or line.startswith('.endc') or line.startswith('.end'):
            continue
            
        # Skip title lines starting with .title
        if line.startswith('.title'):
            continue
            
        # Skip print commands (these are SPICE-specific)
        if line.startswith('print'):
            continue
            
        # Convert measurement voltage sources (VI) to regular voltage sources
        if line.startswith('VI'):
            # VI sources are used for current measurement in SPICE
            # Convert them to 0V voltage sources for Lcapy
            parts = line.split()
            if len(parts) >= 4:
                # VI1 N06 6 0 -> V_VI1 N06 6 0
                name = f"V_{parts[0]}"
                node1, node2 = parts[1], parts[2]
                cleaned_lines.append(f"{name} {node1} {node2} 0")
        else:
            # Handle node name cleaning
            # Replace problematic node names like N06, N43 etc.
            if re.search(r'\bN\d+\b', line):
                # Replace N followed by digits with node_ prefix
                line = re.sub(r'\bN(\d+)\b', r'node_\1', line)
            
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def extract_circuit_elements(netlist_str: str) -> Dict[str, List[str]]:
    """
    Extract different types of circuit elements from the netlist.
    
    Args:
        netlist_str: Cleaned netlist string
        
    Returns:
        Dictionary categorizing circuit elements by type
    """
    lines = netlist_str.strip().split('\n')
    elements = {
        'resistors': [],
        'capacitors': [],
        'inductors': [],
        'voltage_sources': [],
        'current_sources': [],
        'vcvs': [],  # Voltage Controlled Voltage Source
        'vccs': [],  # Voltage Controlled Current Source
        'ccvs': [],  # Current Controlled Voltage Source
        'cccs': [],  # Current Controlled Current Source
        'other': []
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        first_char = line[0].upper()
        
        if first_char == 'R':
            elements['resistors'].append(line)
        elif first_char == 'C':
            elements['capacitors'].append(line)
        elif first_char == 'L':
            elements['inductors'].append(line)
        elif first_char == 'V':
            elements['voltage_sources'].append(line)
        elif first_char == 'I':
            elements['current_sources'].append(line)
        elif first_char == 'E':
            elements['vcvs'].append(line)
        elif first_char == 'G':
            elements['vccs'].append(line)
        elif first_char == 'H':
            elements['ccvs'].append(line)
        elif first_char == 'F':
            elements['cccs'].append(line)
        else:
            elements['other'].append(line)
    
    return elements

def analyze_circuit_with_lcapy(netlist_str: str, circuit_id: str) -> Dict:
    """
    Analyze a circuit using Lcapy and extract symbolic equations.
    
    Args:
        netlist_str: Original SPICE netlist
        circuit_id: Identifier for the circuit
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Clean the netlist for Lcapy compatibility
        cleaned_netlist = clean_netlist_for_lcapy(netlist_str)
        
        if not cleaned_netlist.strip():
            return {
                'circuit_id': circuit_id,
                'status': 'error',
                'error': 'Empty netlist after cleaning',
                'original_netlist': netlist_str,
                'cleaned_netlist': cleaned_netlist
            }
        
        # Create Lcapy circuit
        try:
            circuit = Circuit(cleaned_netlist)
        except Exception as e:
            return {
                'circuit_id': circuit_id,
                'status': 'error',
                'error': f'Failed to create Lcapy circuit: {str(e)}',
                'original_netlist': netlist_str,
                'cleaned_netlist': cleaned_netlist
            }
        
        # Extract circuit elements
        elements = extract_circuit_elements(cleaned_netlist)
        
                 # Get circuit properties
        analysis_results = {
            'circuit_id': circuit_id,
            'status': 'success',
            'original_netlist': netlist_str,
            'cleaned_netlist': cleaned_netlist,
            'elements': elements,
            'circuit_properties': {
                'is_dc': circuit.is_dc,
                'is_ac': circuit.is_ac,
                'is_causal': circuit.is_causal,
                'nodes': list(circuit.nodes.keys()) if hasattr(circuit.nodes, 'keys') else str(circuit.nodes),
                'components': list(circuit.components.keys()) if hasattr(circuit.components, 'keys') else str(circuit.components)
            }
        }
        
        # Try to get nodal equations using modified nodal analysis (MNA)
        try:
            # Get the system of equations in matrix form
            equations = circuit.matrix_equations()
            analysis_results['matrix_equations'] = {
                'A_matrix': str(equations.A),
                'b_vector': str(equations.b),
                'description': 'Matrix equation A*x = b where x is the vector of unknown node voltages and branch currents'
            }
        except Exception as e:
            analysis_results['matrix_equations_error'] = str(e)
        
        # Try to get nodal voltages symbolically
        node_voltages = {}
        try:
            for node_name in circuit.nodes.keys():
                if node_name != '0':  # Skip ground node
                    try:
                        voltage = circuit[node_name].V
                        node_voltages[node_name] = str(voltage)
                    except Exception as e:
                        node_voltages[node_name] = f'Error: {str(e)}'
            analysis_results['node_voltages'] = node_voltages
        except Exception as e:
            analysis_results['node_voltages_error'] = str(e)
        
        # Try to get branch currents
        branch_currents = {}
        try:
            for comp_name in circuit.components.keys():
                try:
                    current = circuit[comp_name].I
                    branch_currents[comp_name] = str(current)
                except Exception as e:
                    branch_currents[comp_name] = f'Error: {str(e)}'
            analysis_results['branch_currents'] = branch_currents
        except Exception as e:
            analysis_results['branch_currents_error'] = str(e)
        
        # Try to get symbolic transfer function if applicable
        try:
            if len(circuit.nodes) > 2:  # Need at least input and output nodes
                nodes = list(circuit.nodes.keys())
                nodes.remove('0')  # Remove ground
                if len(nodes) >= 2:
                    # Try to compute transfer function between first two non-ground nodes
                    input_node = nodes[0]
                    output_node = nodes[1]
                    try:
                        tf = circuit.transfer(input_node, '0', output_node, '0')
                        analysis_results['transfer_function'] = {
                            'input_nodes': (input_node, '0'),
                            'output_nodes': (output_node, '0'),
                            'expression': str(tf)
                        }
                    except Exception as e:
                        analysis_results['transfer_function_error'] = str(e)
        except Exception as e:
            analysis_results['transfer_function_error'] = str(e)
        
        return analysis_results
        
    except Exception as e:
        return {
            'circuit_id': circuit_id,
            'status': 'error',
            'error': str(e),
            'original_netlist': netlist_str
        }

def analyze_dataset(labels_file: str, output_file: str = None, max_circuits: int = None) -> Dict:
    """
    Analyze all circuits in the dataset using Lcapy.
    
    Args:
        labels_file: Path to the labels.json file
        output_file: Optional path to save results
        max_circuits: Optional limit on number of circuits to analyze
        
    Returns:
        Dictionary containing all analysis results
    """
    print(f"Loading dataset from {labels_file}...")
    
    with open(labels_file, 'r') as f:
        dataset = json.load(f)
    
    results = {
        'metadata': {
            'total_circuits': len(dataset),
            'analyzed_circuits': 0,
            'successful_analyses': 0,
            'failed_analyses': 0
        },
        'circuit_analyses': {}
    }
    
    circuit_ids = list(dataset.keys())
    if max_circuits:
        circuit_ids = circuit_ids[:max_circuits]
    
    print(f"Analyzing {len(circuit_ids)} circuits...")
    
    for i, circuit_id in enumerate(circuit_ids):
        print(f"Analyzing circuit {i+1}/{len(circuit_ids)}: {circuit_id}")
        
        netlist = dataset[circuit_id]
        analysis = analyze_circuit_with_lcapy(netlist, circuit_id)
        
        results['circuit_analyses'][circuit_id] = analysis
        results['metadata']['analyzed_circuits'] += 1
        
        if analysis['status'] == 'success':
            results['metadata']['successful_analyses'] += 1
            print(f"  ✓ Success")
        else:
            results['metadata']['failed_analyses'] += 1
            print(f"  ✗ Failed: {analysis.get('error', 'Unknown error')}")
    
    print(f"\nAnalysis complete!")
    print(f"Total circuits: {results['metadata']['total_circuits']}")
    print(f"Analyzed: {results['metadata']['analyzed_circuits']}")
    print(f"Successful: {results['metadata']['successful_analyses']}")
    print(f"Failed: {results['metadata']['failed_analyses']}")
    
    if output_file:
        print(f"Saving results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def demo_single_circuit():
    """
    Demonstrate Lcapy analysis on a single simple circuit.
    """
    print("=== Lcapy Demo: Simple RC Circuit ===")
    
    # Create a simple RC circuit
    netlist = """
    V1 1 0 1
    R1 1 2 R
    C1 2 0 C
    """
    
    try:
        circuit = Circuit(netlist)
        print("Circuit created successfully!")
        print(f"Nodes: {list(circuit.nodes.keys()) if hasattr(circuit.nodes, 'keys') else circuit.nodes}")
        print(f"Components: {list(circuit.components.keys()) if hasattr(circuit.components, 'keys') else circuit.components}")
        
        # Get symbolic voltage across capacitor
        v_c = circuit.C1.V
        print(f"\nVoltage across C1: {v_c}")
        
        # Get symbolic current through resistor
        i_r = circuit.R1.I
        print(f"Current through R1: {i_r}")
        
        # Get transfer function
        H = circuit.transfer('1', '0', '2', '0')
        print(f"Transfer function V2/V1: {H}")
        
        # Get impedance looking into the circuit from V1
        Z_in = circuit.impedance('1', '0')
        print(f"Input impedance: {Z_in}")
        
        return True
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return False

def main():
    """
    Main function to run the circuit analysis.
    """
    print("Lcapy Circuit Analysis Script")
    print("============================")
    
    # First run a demo to verify Lcapy is working
    if not demo_single_circuit():
        print("Demo failed - check Lcapy installation")
        return
    
    print("\n" + "="*50)
    
    # Analyze the dataset
    labels_file = "datasets/grid_v11_240831/labels.json"
    
    if not Path(labels_file).exists():
        print(f"Dataset file not found: {labels_file}")
        print("Please ensure the dataset is in the correct location.")
        return
    
    # Analyze a small subset first for testing
    print("\nStarting analysis of dataset circuits...")
    results = analyze_dataset(
        labels_file=labels_file, 
        output_file="lcapy_analysis_results.json",
        max_circuits=5  # Start with just 5 circuits for testing
    )
    
    # Print summary of a successful analysis
    print("\n=== Example Analysis ===")
    successful_circuits = [
        cid for cid, analysis in results['circuit_analyses'].items() 
        if analysis['status'] == 'success'
    ]
    
    if successful_circuits:
        example_id = successful_circuits[0]
        example = results['circuit_analyses'][example_id]
        
        print(f"Circuit: {example_id}")
        print(f"Elements: {example['elements']}")
        print(f"Nodes: {example['circuit_properties']['nodes']}")
        print(f"Components: {example['circuit_properties']['components']}")
        
        if 'node_voltages' in example:
            print("Node voltages:")
            for node, voltage in example['node_voltages'].items():
                print(f"  V({node}): {voltage}")

if __name__ == "__main__":
    main() 