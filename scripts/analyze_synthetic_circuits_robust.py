import json
import argparse
from pathlib import Path
from lcapy import Circuit, s, t
import re

def clean_netlist_for_lcapy(spice_netlist):
    lines = []
    skip_control_block = False
    
    for line in spice_netlist.strip().split('\n'):
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('*'):
            continue
            
        # Skip control blocks
        if line.startswith('.control'):
            skip_control_block = True
            continue
        if line.startswith('.endc'):
            skip_control_block = False
            continue
        if skip_control_block:
            continue
            
        # Skip SPICE directives
        if line.startswith('.') or line.startswith('print') or ';' in line:
            continue
            
        # Skip AC/DC analysis commands
        if line.startswith(('ac ', 'dc ', 'tran ', 'op')):
            continue
        
        parts = line.split()
        if len(parts) < 4:
            continue
            
        component = parts[0]
        
        # Must start with a valid component identifier
        if not component[0].upper() in ['R', 'L', 'C', 'V', 'I', 'E', 'F', 'G', 'H']:
            continue
            
        node1, node2 = parts[1], parts[2]
        value = parts[3]
        
        # Clean node names
        node1 = re.sub(r'[^\w]', '_', node1)
        node2 = re.sub(r'[^\w]', '_', node2)
        
        # Handle measurement voltage sources
        if component.startswith('VI'):
            component = f"V_meas{component[2:]}"
            value = "0"
        
        # Handle empty values
        if value == '<Empty>':
            value = component
        
        # Handle voltage sources with DC/AC keywords
        if component.startswith('V') and value in ['DC', 'AC']:
            if len(parts) > 4:
                value = parts[4]
            else:
                value = component
            
        lines.append(f"{component} {node1} {node2} {value}")
    
    return '\n'.join(lines)

def find_voltage_sources(circuit):
    voltage_sources = []
    for name, element in circuit.elements.items():
        if name.startswith('V') and not name.startswith('V_meas'):
            nodes = [str(n) for n in element.nodes]
            voltage_sources.append((name, tuple(nodes)))
    return voltage_sources

def find_components(circuit):
    components = []
    for name, element in circuit.elements.items():
        if not name.startswith('V'):
            components.append(name)
    return components

def analyze_circuit(netlist, circuit_id):
    try:
        cleaned = clean_netlist_for_lcapy(netlist)
        print(f"\nCircuit {circuit_id}:")
        print(f"Cleaned netlist:\n{cleaned}")
        
        if not cleaned:
            print("No components after cleaning")
            return None
            
        circuit = Circuit(cleaned)
        
        voltage_sources = find_voltage_sources(circuit)
        components = find_components(circuit)
        
        print(f"Voltage sources: {voltage_sources}")
        print(f"Components: {components}")
        
        if not voltage_sources or not components:
            print("Missing voltage sources or components")
            return None
        
        result = {
            'circuit_id': circuit_id,
            'cleaned_netlist': cleaned,
            'voltage_sources': voltage_sources,
            'components': components,
            'transfer_functions': {},
            'nodal_equations': {}
        }
        
        vs_name, vs_nodes = voltage_sources[0]
        
        for comp in components[:3]:
            try:
                H = circuit.transfer(vs_nodes, comp)
                result['transfer_functions'][f"{vs_name}_to_{comp}"] = str(H)
                print(f"Transfer function {vs_name} -> {comp}: {H}")
            except Exception as e:
                print(f"Transfer function failed for {comp}: {e}")
        
        try:
            n_t = circuit.nodal_analysis(node_prefix='n')
            result['nodal_equations']['t_domain'] = str(n_t.nodal_equations())
            print(f"T-domain equations: {n_t.nodal_equations()}")
        except Exception as e:
            print(f"T-domain equations failed: {e}")
        
        try:
            n_s = circuit.laplace().nodal_analysis(node_prefix='n')
            result['nodal_equations']['s_domain'] = str(n_s.nodal_equations())
            print(f"S-domain equations: {n_s.nodal_equations()}")
        except Exception as e:
            print(f"S-domain equations failed: {e}")
        
        return result
        
    except Exception as e:
        print(f"Circuit {circuit_id} failed: {e}")
        return {'circuit_id': circuit_id, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_file', default="datasets/grid_v11_240831/labels.json")
    parser.add_argument('--output_file', default="symbolic_equations.json")
    parser.add_argument('--max_circuits', type=int, default=None)
    args = parser.parse_args()
    
    if not Path(args.labels_file).exists():
        print(f"File not found: {args.labels_file}")
        return
    
    with open(args.labels_file, 'r') as f:
        circuit_data = json.load(f)
    
    circuits = list(circuit_data.items())
    if args.max_circuits:
        circuits = circuits[:args.max_circuits]
    
    results = []
    successful = 0
    
    for circuit_id, netlist in circuits:
        result = analyze_circuit(netlist, circuit_id)
        if result and 'error' not in result:
            successful += 1
            results.append(result)
            print(f"✓ {circuit_id}")
        else:
            print(f"✗ {circuit_id}")
    
    output = {
        'summary': {
            'total': len(circuits),
            'successful': successful,
            'success_rate': successful / len(circuits)
        },
        'results': results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nAnalyzed {successful}/{len(circuits)} circuits")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()