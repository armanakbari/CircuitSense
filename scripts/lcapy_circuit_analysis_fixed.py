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
    5. Handle current sources properly
    6. Convert units (k, m, u, n, p) to numeric values
    7. Handle controlled sources (E, F, G, H)
    
    Args:
        netlist_str: Raw SPICE netlist string
        
    Returns:
        Cleaned netlist string compatible with Lcapy
    """
    lines = netlist_str.strip().split('\n')
    cleaned_lines = []
    
    def convert_value(value_str):
        """Convert SPICE units to numeric values."""
        if not value_str:
            return value_str
            
        # Unit multipliers
        multipliers = {
            'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'K': 1e3, 'k': 1e3,
            'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
        }
        
        # Try to parse value with unit
        for unit, mult in multipliers.items():
            if value_str.upper().endswith(unit.upper()):
                try:
                    base_value = float(value_str[:-len(unit)])
                    return str(base_value * mult)
                except ValueError:
                    continue
        
        return value_str
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('*'):
            continue
            
        # Skip SPICE commands and all control/measurement statements
        if line.startswith('.') or line.startswith('print'):
            continue
            
        # Skip semicolon comments (SPICE style)
        if ';' in line:
            line = line.split(';')[0].strip()
            if not line:
                continue
        
        # Convert voltage measurement sources to 0V sources
        if line.startswith('VI'):
            parts = line.split()
            if len(parts) >= 4:
                # VI1 N06 6 0 -> V_VI1 N06 6 0
                name = f"V_{parts[0]}"
                node1 = parts[1]
                node2 = parts[2] 
                # Set to 0V for measurement
                new_line = f"{name} {node1} {node2} 0"
                cleaned_lines.append(new_line)
        # Handle controlled sources - skip them for now as they're complex
        elif line.startswith(('E', 'F', 'G', 'H')):
            # Skip controlled sources for initial implementation
            print(f"Skipping controlled source: {line}")
            continue
        else:
            # Process regular components
            parts = line.split()
            if len(parts) >= 4:
                # Check if this is a valid component line (starts with component letter)
                first_char = parts[0][0].upper()
                valid_components = ['R', 'C', 'L', 'V', 'I']
                
                if first_char in valid_components:
                    # Convert units in the value field
                    for i in range(3, len(parts)):
                        parts[i] = convert_value(parts[i])
                    
                    # Clean node names - replace problematic characters
                    cleaned_line = ' '.join(parts)
                    cleaned_line = re.sub(r'[^\w\s.-]', '_', cleaned_line)
                    cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)

def extract_circuit_elements(netlist_str: str) -> Dict[str, List[str]]:
    """
    Extract and categorize circuit elements from netlist.
    
    Args:
        netlist_str: Netlist string
        
    Returns:
        Dictionary with categorized elements
    """
    elements = {
        'resistors': [],
        'capacitors': [],
        'inductors': [],
        'voltage_sources': [],
        'current_sources': [],
        'other': []
    }
    
    lines = netlist_str.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('.') or line.startswith('*'):
            continue
            
        parts = line.split()
        if len(parts) < 3:
            continue
            
        element_name = parts[0]
        first_char = element_name[0].upper()
        
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
        else:
            elements['other'].append(line)
    
    return elements

def analyze_circuit_with_lcapy(netlist_str: str, circuit_id: str) -> Dict:
    """
    Analyze a single circuit using Lcapy.
    
    Args:
        netlist_str: SPICE netlist string
        circuit_id: Identifier for the circuit
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Clean the netlist for Lcapy
        cleaned_netlist = clean_netlist_for_lcapy(netlist_str)
        
        if not cleaned_netlist.strip():
            return {
                'circuit_id': circuit_id,
                'status': 'error',
                'error': 'Empty netlist after cleaning',
                'original_netlist': netlist_str
            }
        
        # Extract circuit elements
        elements = extract_circuit_elements(cleaned_netlist)
        
        # Create Lcapy circuit
        circuit = Circuit(cleaned_netlist)
        
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
                'nodes': str(circuit.nodes),
                'components': str(circuit.components)
            }
        }
        
        # Try to get node voltages
        node_voltages = {}
        try:
            for node in circuit.nodes:
                if str(node) != '0':  # Skip ground node
                    try:
                        voltage = circuit[node].V
                        node_voltages[str(node)] = str(voltage)
                    except Exception as e:
                        node_voltages[str(node)] = f'Error: {str(e)}'
            analysis_results['node_voltages'] = node_voltages
        except Exception as e:
            analysis_results['node_voltages_error'] = str(e)
        
        # Try to get branch currents
        branch_currents = {}
        try:
            for comp_name in circuit.components:
                try:
                    current = circuit[comp_name].I
                    branch_currents[str(comp_name)] = str(current)
                except Exception as e:
                    branch_currents[str(comp_name)] = f'Error: {str(e)}'
            analysis_results['branch_currents'] = branch_currents
        except Exception as e:
            analysis_results['branch_currents_error'] = str(e)
        
        # Try to get symbolic equations using Kirchhoff's laws
        try:
            # Get the system of equations
            equations = circuit.nodal_analysis()
            analysis_results['nodal_equations'] = str(equations)
        except Exception as e:
            analysis_results['nodal_equations_error'] = str(e)
        
        # Try to get impedance matrix if applicable
        try:
            Z = circuit.Z
            analysis_results['impedance_matrix'] = str(Z)
        except Exception as e:
            analysis_results['impedance_matrix_error'] = str(e)
        
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

def demo_simple_circuit():
    """
    Demonstrate Lcapy analysis on a simple circuit.
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
        print(f"Nodes: {circuit.nodes}")
        print(f"Components: {circuit.components}")
        
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

def demo_from_dataset():
    """
    Demonstrate analysis on a circuit from the actual dataset.
    """
    print("=== Demo: Circuit from Dataset ===")
    
    labels_file = "datasets/grid_v11_240831/labels.json"
    
    if not Path(labels_file).exists():
        print(f"Dataset file not found: {labels_file}")
        return False
    
    # Load one circuit from the dataset
    with open(labels_file, 'r') as f:
        dataset = json.load(f)
    
    # Get the first circuit
    circuit_id = list(dataset.keys())[0]
    netlist = dataset[circuit_id]
    
    print(f"Analyzing circuit: {circuit_id}")
    print(f"Original netlist:\n{netlist}\n")
    
    # Analyze the circuit
    analysis = analyze_circuit_with_lcapy(netlist, circuit_id)
    
    if analysis['status'] == 'success':
        print("✓ Analysis successful!")
        print(f"Cleaned netlist:\n{analysis['cleaned_netlist']}\n")
        print(f"Elements found:")
        for category, items in analysis['elements'].items():
            if items:
                print(f"  {category}: {len(items)} items")
        
        if 'node_voltages' in analysis:
            print("\nNode voltages:")
            for node, voltage in analysis['node_voltages'].items():
                print(f"  V({node}): {voltage}")
        
        if 'branch_currents' in analysis:
            print("\nBranch currents:")
            for comp, current in analysis['branch_currents'].items():
                print(f"  I({comp}): {current}")
                
        if 'nodal_equations' in analysis:
            print(f"\nNodal equations:\n{analysis['nodal_equations']}")
    else:
        print(f"✗ Analysis failed: {analysis['error']}")
    
    return analysis['status'] == 'success'

def main():
    """
    Main function to run the circuit analysis.
    """
    print("Lcapy Circuit Analysis Script")
    print("============================")
    
    # First run a simple demo to verify Lcapy is working
    if not demo_simple_circuit():
        print("Simple demo failed - check Lcapy installation")
        return
    
    print("\n" + "="*50)
    
    # Demo with dataset circuit
    if not demo_from_dataset():
        print("Dataset demo failed")
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
    print("\n=== Example Analysis Summary ===")
    successful_circuits = [
        cid for cid, analysis in results['circuit_analyses'].items() 
        if analysis['status'] == 'success'
    ]
    
    if successful_circuits:
        example_id = successful_circuits[0]
        example = results['circuit_analyses'][example_id]
        
        print(f"Circuit: {example_id}")
        print(f"Elements: {example['elements']}")
        
        if 'node_voltages' in example:
            print("Node voltages:")
            for node, voltage in example['node_voltages'].items():
                print(f"  V({node}): {voltage}")

if __name__ == "__main__":
    main() 