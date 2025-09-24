#!/usr/bin/env python3
"""
Convert Circuit Netlists: Remove N-nodes and Measurement Components

This script processes netlists to remove intermediate measurement nodes (N-prefixed)
and V_meas components, restoring the original circuit connectivity that matches
the visual circuit diagrams.

Pattern Recognition:
- Original: R1 2 1 100    (component from node 2 to node 1)
- With measurement: R1 2 N21 100 + V_meas1 1 N21 0
- Converted back: R1 2 1 100

Usage:
    python convert_netlist_remove_n_nodes.py input.json output.json
"""

import json
import re
import sys
from typing import Dict, List, Tuple

def parse_netlist_line(line: str) -> Tuple[str, str, str, str]:
    """
    Parse a SPICE netlist line into components.
    
    Args:
        line: SPICE netlist line (e.g., "R1 2 N21 100")
        
    Returns:
        Tuple of (component_name, node1, node2, value)
    """
    parts = line.strip().split()
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], ' '.join(parts[3:])
    elif len(parts) == 3:
        return parts[0], parts[1], parts[2], ""
    else:
        return line, "", "", ""

def find_measurement_pairs(netlist_lines: List[str]) -> Dict[str, Tuple[str, str, str, str]]:
    """
    Find pairs of components that form measurement circuits.
    
    Args:
        netlist_lines: List of netlist lines
        
    Returns:
        Dict mapping N-node names to (original_component_info, measurement_component_info)
    """
    components = {}
    measurement_sources = {}
    
    # Parse all components
    for line in netlist_lines:
        if not line.strip():
            continue
            
        comp_name, node1, node2, value = parse_netlist_line(line)
        
        # Check if this connects to an N-node
        n_node = None
        other_node = None
        
        if node1.startswith('N') and not node1.startswith('Ninv') and not node1.startswith('Nmeas'):
            n_node = node1
            other_node = node2
        elif node2.startswith('N') and not node2.startswith('Ninv') and not node2.startswith('Nmeas'):
            n_node = node2
            other_node = node1
            
        if n_node:
            if comp_name.startswith('V_meas') or comp_name.startswith('VI'):
                # This is a measurement voltage source
                measurement_sources[n_node] = (comp_name, node1, node2, value, line)
            else:
                # This is a regular component connecting to N-node
                components[n_node] = (comp_name, node1, node2, value, line, other_node)
    
    return components, measurement_sources

def convert_netlist_remove_n_nodes(netlist: str) -> str:
    """
    Convert a netlist to remove N-nodes and measurement components.
    
    Args:
        netlist: Original netlist string with N-nodes
        
    Returns:
        Converted netlist string without N-nodes
    """
    lines = netlist.split('\n')
    components, measurement_sources = find_measurement_pairs(lines)
    
    converted_lines = []
    processed_n_nodes = set()
    
    for line in lines:
        if not line.strip():
            continue
            
        comp_name, node1, node2, value = parse_netlist_line(line)
        
        # Skip measurement voltage sources (V_meas, VI)
        if comp_name.startswith('V_meas') or comp_name.startswith('VI'):
            continue
            
        # Check if this component connects to an N-node
        n_node = None
        if node1.startswith('N') and not node1.startswith('Ninv') and not node1.startswith('Nmeas'):
            n_node = node1
        elif node2.startswith('N') and not node2.startswith('Ninv') and not node2.startswith('Nmeas'):
            n_node = node2
            
        if n_node and n_node not in processed_n_nodes:
            # Find the measurement source that completes this connection
            if n_node in measurement_sources:
                meas_comp, meas_node1, meas_node2, meas_value, meas_line = measurement_sources[n_node]
                
                # Determine the final connection
                # The measurement source connects the N-node to the final destination
                if meas_node1 == n_node:
                    final_node = meas_node2
                else:
                    final_node = meas_node1
                
                # Reconstruct the original component connection
                if node1 == n_node:
                    converted_line = f"{comp_name} {final_node} {node2} {value}".strip()
                else:
                    converted_line = f"{comp_name} {node1} {final_node} {value}".strip()
                
                converted_lines.append(converted_line)
                processed_n_nodes.add(n_node)
            else:
                # No measurement source found, keep original (shouldn't happen in well-formed netlists)
                converted_lines.append(line)
        elif not n_node:
            # Regular component without N-nodes, keep as-is
            converted_lines.append(line)
        # Skip lines that connect to already processed N-nodes
    
    return '\n'.join(converted_lines)

def process_json_file(input_file: str, output_file: str):
    """
    Process a JSON file containing circuit data and convert all netlists.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    print(f"Loading {input_file}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if 'results' in data:
        circuits = data['results']
    elif isinstance(data, list):
        circuits = data
    else:
        raise ValueError("Unknown JSON structure - expected 'results' key or direct array")
    
    total_circuits = len(circuits)
    converted_count = 0
    
    print(f"Processing {total_circuits} circuits...")
    
    for i, circuit in enumerate(circuits):
        circuit_id = circuit.get('circuit_id', f'circuit_{i}')
        
        if 'cleaned_netlist' in circuit:
            original_netlist = circuit['cleaned_netlist']
            
            # Convert the netlist
            converted_netlist = convert_netlist_remove_n_nodes(original_netlist)
            
            # Update the circuit data
            circuit['cleaned_netlist'] = converted_netlist
            circuit['original_netlist_with_measurements'] = original_netlist  # Preserve original
            
            converted_count += 1
            
            # Print progress and examples
            if i < 3:  # Show first 3 conversions as examples
                print(f"\n=== Example {i+1}: {circuit_id} ===")
                print("Original:")
                for line in original_netlist.split('\n')[:5]:  # Show first 5 lines
                    print(f"  {line}")
                print("Converted:")
                for line in converted_netlist.split('\n')[:5]:
                    print(f"  {line}")
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_circuits} circuits...")
    
    print(f"\nSaving converted data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Conversion complete!")
    print(f"   - Total circuits: {total_circuits}")
    print(f"   - Converted: {converted_count}")
    print(f"   - Output file: {output_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_netlist_remove_n_nodes.py input.json output.json")
        print("\nExample:")
        print("  python convert_netlist_remove_n_nodes.py datasets/mllm_benchmark_v12/symbolic_equations.json datasets/mllm_benchmark_v12/symbolic_equations_no_n_nodes.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        process_json_file(input_file, output_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 