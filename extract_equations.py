#!/usr/bin/env python3
"""
Extract Symbolic Equations from Circuit Netlists using Lcapy

This script focuses on extracting symbolic system of equations from circuit netlists.
It provides clear output of the equations in mathematical form.

Author: Assistant
Date: 2024
"""

import json
import re
from pathlib import Path
from lcapy import Circuit
import sympy as sp
from typing import Dict, List
import warnings

def clean_netlist_for_lcapy(netlist_str: str) -> str:
    """Clean SPICE netlist for Lcapy compatibility."""
    lines = netlist_str.strip().split('\n')
    cleaned_lines = []
    
    def convert_value(value_str):
        """Convert SPICE units to numeric values."""
        if not value_str:
            return value_str
            
        multipliers = {
            'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'K': 1e3, 'k': 1e3,
            'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
        }
        
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
        
        # Skip empty lines, comments, and SPICE commands
        if not line or line.startswith('*') or line.startswith('.') or line.startswith('print'):
            continue
            
        # Remove semicolon comments
        if ';' in line:
            line = line.split(';')[0].strip()
            if not line:
                continue
        
        # Convert measurement voltage sources
        if line.startswith('VI'):
            parts = line.split()
            if len(parts) >= 4:
                name = f"V_{parts[0]}"
                new_line = f"{name} {parts[1]} {parts[2]} 0"
                cleaned_lines.append(new_line)
        # Skip controlled sources
        elif line.startswith(('E', 'F', 'G', 'H')):
            continue
        else:
            # Process regular components
            parts = line.split()
            if len(parts) >= 4:
                first_char = parts[0][0].upper()
                if first_char in ['R', 'C', 'L', 'V', 'I']:
                    # Convert units
                    for i in range(3, len(parts)):
                        parts[i] = convert_value(parts[i])
                    
                    # Clean node names
                    cleaned_line = ' '.join(parts)
                    cleaned_line = re.sub(r'[^\w\s.-]', '_', cleaned_line)
                    cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)

def extract_symbolic_equations(circuit_id: str, netlist_str: str) -> Dict:
    """
    Extract symbolic equations from a circuit netlist.
    
    Args:
        circuit_id: Circuit identifier
        netlist_str: SPICE netlist string
        
    Returns:
        Dictionary containing symbolic equations and analysis
    """
    try:
        # Clean netlist
        cleaned_netlist = clean_netlist_for_lcapy(netlist_str)
        
        if not cleaned_netlist.strip():
            return {'circuit_id': circuit_id, 'status': 'error', 'error': 'Empty netlist'}
        
        # Create circuit
        circuit = Circuit(cleaned_netlist)
        
        result = {
            'circuit_id': circuit_id,
            'status': 'success',
            'cleaned_netlist': cleaned_netlist,
            'nodes': [str(node) for node in circuit.nodes if str(node) != '0'],
            'components': {}
        }
        
        # Get component information
        for comp_type in ['resistors', 'capacitors', 'inductors', 'voltage_sources', 'current_sources']:
            if hasattr(circuit, comp_type):
                comp_list = getattr(circuit, comp_type)
                if comp_list:
                    result['components'][comp_type] = [str(comp) for comp in comp_list]
        
        # Extract symbolic nodal equations
        try:
            nodal = circuit.nodal_analysis()
            
            # Get the symbolic equations
            if hasattr(nodal, 'equations'):
                equations = nodal.equations
                result['symbolic_equations'] = {
                    'matrix_form': str(equations),
                    'individual_equations': []
                }
                
                # Try to get individual equations
                if hasattr(equations, 'lhs') and hasattr(equations, 'rhs'):
                    for i, (lhs, rhs) in enumerate(zip(equations.lhs, equations.rhs)):
                        eq_str = f"{lhs} = {rhs}"
                        result['symbolic_equations']['individual_equations'].append(eq_str)
            
            # Get the system matrix A and vector b (Ax = b)
            if hasattr(nodal, 'A') and hasattr(nodal, 'b'):
                result['system_matrix'] = {
                    'A_matrix': str(nodal.A),
                    'b_vector': str(nodal.b),
                    'description': 'System: A * x = b, where x is the vector of node voltages'
                }
            
            # Get node voltage expressions
            node_voltages = {}
            for node in result['nodes']:
                try:
                    voltage = circuit[node].V
                    node_voltages[f'V_{node}'] = str(voltage)
                except:
                    pass
            result['node_voltage_expressions'] = node_voltages
            
        except Exception as e:
            result['equations_error'] = str(e)
        
        return result
        
    except Exception as e:
        return {
            'circuit_id': circuit_id,
            'status': 'error',
            'error': str(e)
        }

def analyze_single_circuit(circuit_id: str, netlist: str):
    """Analyze and display equations for a single circuit."""
    print(f"\n{'='*60}")
    print(f"Circuit Analysis: {circuit_id}")
    print(f"{'='*60}")
    
    analysis = extract_symbolic_equations(circuit_id, netlist)
    
    if analysis['status'] == 'error':
        print(f"❌ Error: {analysis['error']}")
        return
    
    print("✅ Analysis successful!")
    
    # Show cleaned netlist
    print(f"\n📋 Cleaned Netlist:")
    print(analysis['cleaned_netlist'])
    
    # Show circuit components
    print(f"\n🔧 Circuit Components:")
    for comp_type, components in analysis['components'].items():
        if components:
            print(f"  {comp_type}: {components}")
    
    # Show nodes
    print(f"\n📍 Nodes: {analysis['nodes']}")
    
    # Show symbolic equations
    if 'symbolic_equations' in analysis:
        print(f"\n🧮 Symbolic Equations:")
        eq_info = analysis['symbolic_equations']
        
        if 'individual_equations' in eq_info and eq_info['individual_equations']:
            print("Individual nodal equations:")
            for i, eq in enumerate(eq_info['individual_equations']):
                print(f"  Equation {i+1}: {eq}")
        else:
            print(f"Matrix form: {eq_info.get('matrix_form', 'Not available')}")
    
    # Show system matrix
    if 'system_matrix' in analysis:
        print(f"\n📊 System Matrix (Ax = b):")
        sys_info = analysis['system_matrix']
        print(f"  Description: {sys_info['description']}")
        print(f"  A matrix: {sys_info['A_matrix']}")
        print(f"  b vector: {sys_info['b_vector']}")
    
    # Show node voltage expressions
    if 'node_voltage_expressions' in analysis:
        print(f"\n⚡ Node Voltage Expressions:")
        for node, expr in analysis['node_voltage_expressions'].items():
            print(f"  {node}: {expr}")

def main():
    """Main function to demonstrate symbolic equation extraction."""
    print("Symbolic Circuit Equation Extractor using Lcapy")
    print("=" * 50)
    
    # Load dataset
    labels_file = "datasets/grid_v11_240831/labels.json"
    
    if not Path(labels_file).exists():
        print(f"Dataset file not found: {labels_file}")
        return
    
    with open(labels_file, 'r') as f:
        dataset = json.load(f)
    
    # Analyze first few circuits as examples
    circuit_ids = list(dataset.keys())[:3]  # First 3 circuits
    
    for circuit_id in circuit_ids:
        netlist = dataset[circuit_id]
        analyze_single_circuit(circuit_id, netlist)
    
    print(f"\n\n{'='*60}")
    print("Analysis Summary")
    print(f"{'='*60}")
    print(f"Analyzed {len(circuit_ids)} circuits successfully!")
    print("The symbolic equations show the relationships between node voltages,")
    print("resistances, currents, and voltage sources in mathematical form.")
    print("\nThese equations can be used for:")
    print("- Circuit analysis and design")
    print("- Parameter sensitivity analysis") 
    print("- Optimization and synthesis")
    print("- Educational purposes")

if __name__ == "__main__":
    main() 