#!/usr/bin/env python3

import json
import re

def identify_voltages(circuit_data):
    """
    Identify v_in (voltage source) and v_out (voltage measurements) from circuit data
    
    Args:
        circuit_data: Dictionary containing 'spice' key with SPICE netlist
    
    Returns:
        Dictionary with v_in and v_out information
    """
    spice_code = circuit_data['spice']
    
    # Find voltage source (v_in)
    v_source_pattern = r'V(\d+)\s+(\w+)\s+(\w+)\s+(\d+)'
    v_source_match = re.search(v_source_pattern, spice_code)
    
    v_in = None
    if v_source_match:
        source_name = f"V{v_source_match.group(1)}"
        node1 = v_source_match.group(2)
        node2 = v_source_match.group(3)
        voltage = v_source_match.group(4)
        v_in = {
            'name': source_name,
            'voltage': f"{voltage}V",
            'nodes': [node1, node2],
            'description': f"{voltage}V between nodes {node1} and {node2}"
        }
    
    # Find voltage measurements (v_out)
    v_out_pattern = r'print\s+([-v()\w,\s]+)\s*;\s*measurement\s+of\s+U(\w*)'
    v_out_matches = re.findall(v_out_pattern, spice_code)
    
    v_out = []
    for match in v_out_matches:
        measurement_cmd = match[0].strip()
        label = f"U{match[1]}"
        
        # Parse different voltage measurement formats
        if measurement_cmd.startswith('-v(') and measurement_cmd.endswith(')'):
            # Format: -v(node) - voltage at node relative to ground
            node = measurement_cmd[3:-1]
            description = f"Voltage at node {node} relative to ground"
            nodes = [node, "0"]
        elif measurement_cmd.startswith('v(') and ', ' in measurement_cmd:
            # Format: v(node1, node2) - voltage between two nodes
            nodes_str = measurement_cmd[2:-1]
            nodes = [n.strip() for n in nodes_str.split(',')]
            description = f"Voltage between nodes {nodes[0]} and {nodes[1]}"
        elif measurement_cmd.startswith('v(') and measurement_cmd.endswith(')'):
            # Format: v(node) - voltage at node relative to ground
            node = measurement_cmd[2:-1]
            description = f"Voltage at node {node} relative to ground"
            nodes = [node, "0"]
        else:
            description = f"Unknown format: {measurement_cmd}"
            nodes = []
        
        v_out.append({
            'label': label,
            'command': measurement_cmd,
            'nodes': nodes,
            'description': description
        })
    
    return {
        'v_in': v_in,
        'v_out': v_out,
        'summary': {
            'input_voltage': v_in['description'] if v_in else "No voltage source found",
            'num_outputs': len(v_out),
            'output_labels': [vo['label'] for vo in v_out]
        }
    }

def analyze_circuit_file(json_file_path):
    """
    Analyze all circuits in a JSON file and identify their voltages
    """
    print(f"Analyzing circuits in: {json_file_path}")
    print("=" * 60)
    
    with open(json_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                circuit = json.loads(line.strip())
                circuit_id = circuit.get('id', f'line_{line_num}')
                
                voltage_info = identify_voltages(circuit)
                
                print(f"\n📋 Circuit {circuit_id}:")
                print(f"   🔋 v_in:  {voltage_info['summary']['input_voltage']}")
                print(f"   ⚡ v_out: {voltage_info['summary']['num_outputs']} measurements")
                
                for vo in voltage_info['v_out']:
                    print(f"      • {vo['label']}: {vo['description']}")
                    
            except json.JSONDecodeError:
                print(f"Error parsing line {line_num}")
                continue

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = "ppm_construction/data_syn/data/test_constraint_fixed.json"
    
    try:
        analyze_circuit_file(json_file)
    except FileNotFoundError:
        print(f"File not found: {json_file}")
        print("Usage: python identify_voltages.py <circuit_file.json>") 