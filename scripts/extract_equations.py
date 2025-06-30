#!/usr/bin/env python3
"""
Comprehensive Symbolic Equation Extraction with Lcapy

This script demonstrates all the methods available in lcapy for extracting
symbolic equations from circuits, including:
1. Nodal analysis equations (KCL at each node)
2. Modified nodal analysis (MNA) matrix equations
3. Mesh/loop analysis equations
4. State-space equations
5. Individual component equations
6. System matrix equations

Author: Assistant
Date: 2024
"""

import json
from pathlib import Path
import lcapy
from lcapy import Circuit
import sympy as sp
from typing import Dict, List, Tuple, Optional

def extract_all_symbolic_equations(circuit_spice: str, circuit_name: str = "example") -> Dict:
    """
    Extract all possible symbolic equations from a SPICE netlist string.
    
    Parameters:
    -----------
    circuit_spice : str
        SPICE netlist as a string
    circuit_name : str
        Name for the circuit (for documentation)
        
    Returns:
    --------
    Dict containing all extracted equations and analysis results
    """
    
    print(f"\n{'='*80}")
    print(f"SYMBOLIC EQUATION EXTRACTION FOR: {circuit_name.upper()}")
    print(f"{'='*80}")
    
    # Create circuit from SPICE netlist
    cct = Circuit(circuit_spice)
    
    # Initialize results dictionary
    results = {
        'circuit_name': circuit_name,
        'netlist': circuit_spice,
        'equations': {},
        'matrices': {},
        'analysis_methods': []
    }
    
    print(f"\nCircuit Netlist:")
    print(circuit_spice)
    
    # 1. NODAL ANALYSIS EQUATIONS
    try:
        print(f"\n{'-'*60}")
        print("1. NODAL ANALYSIS EQUATIONS")
        print(f"{'-'*60}")
        
        na = cct.nodal_analysis()
        
        # Get nodal equations (KCL at each node)
        nodal_eqs = na.nodal_equations()
        print("Nodal equations (KCL at each node):")
        for node, eq in nodal_eqs.items():
            print(f"  Node {node}: {eq}")
            
        # Get matrix form of nodal equations
        matrix_eqs = na.matrix_equations()
        print(f"\nMatrix form: A*v = b")
        print(f"A matrix:\n{na.A}")
        print(f"b vector:\n{na.b}")
        
        results['equations']['nodal'] = {
            'node_equations': {str(k): str(v) for k, v in nodal_eqs.items()},
            'matrix_A': str(na.A),
            'vector_b': str(na.b),
            'unknowns': str(na.unknowns)
        }
        results['analysis_methods'].append('nodal_analysis')
        
    except Exception as e:
        print(f"Nodal analysis failed: {e}")
        results['equations']['nodal'] = {'error': str(e)}
    
    # 2. MODIFIED NODAL ANALYSIS (MNA)
    try:
        print(f"\n{'-'*60}")
        print("2. MODIFIED NODAL ANALYSIS (MNA)")
        print(f"{'-'*60}")
        
        mna = cct.modified_nodal_analysis()
        
        # Get MNA matrix equations
        mna_matrix_eqs = mna.matrix_equations()
        print("MNA Matrix equations:")
        print(f"G matrix (conductance):\n{mna.G}")
        print(f"B matrix:\n{mna.B}")
        print(f"C matrix:\n{mna.C}")
        print(f"D matrix:\n{mna.D}")
        
        results['equations']['mna'] = {
            'G_matrix': str(mna.G),
            'B_matrix': str(mna.B), 
            'C_matrix': str(mna.C),
            'D_matrix': str(mna.D),
            'matrix_equations': str(mna_matrix_eqs)
        }
        results['analysis_methods'].append('modified_nodal_analysis')
        
    except Exception as e:
        print(f"MNA failed: {e}")
        results['equations']['mna'] = {'error': str(e)}
    
    # 3. MESH/LOOP ANALYSIS
    try:
        print(f"\n{'-'*60}")
        print("3. MESH/LOOP ANALYSIS")
        print(f"{'-'*60}")
        
        la = cct.loop_analysis()
        
        # Get mesh equations
        mesh_eqs = la.mesh_equations()
        print("Mesh equations (KVL around each loop):")
        for mesh, eq in mesh_eqs.items():
            print(f"  Mesh {mesh}: {eq}")
            
        # Get matrix form
        mesh_matrix = la.matrix_equations()
        print(f"\nMesh matrix form:")
        print(f"A matrix:\n{la.A}")
        print(f"b vector:\n{la.b}")
        
        results['equations']['mesh'] = {
            'mesh_equations': {str(k): str(v) for k, v in mesh_eqs.items()},
            'matrix_A': str(la.A),
            'vector_b': str(la.b),
            'unknowns': str(la.unknowns)
        }
        results['analysis_methods'].append('loop_analysis')
        
    except Exception as e:
        print(f"Mesh analysis failed: {e}")
        results['equations']['mesh'] = {'error': str(e)}
    
    # 4. STATE-SPACE ANALYSIS
    try:
        print(f"\n{'-'*60}")
        print("4. STATE-SPACE ANALYSIS")
        print(f"{'-'*60}")
        
        ss = cct.ss
        
        # Get state equations
        state_eqs = ss.state_equations()
        print("State equations (dx/dt = Ax + Bu):")
        print(state_eqs)
        
        # Get output equations  
        output_eqs = ss.output_equations()
        print("\nOutput equations (y = Cx + Du):")
        print(output_eqs)
        
        print(f"\nState-space matrices:")
        print(f"A matrix:\n{ss.A}")
        print(f"B matrix:\n{ss.B}")
        print(f"C matrix:\n{ss.C}")
        print(f"D matrix:\n{ss.D}")
        
        results['equations']['state_space'] = {
            'state_equations': str(state_eqs),
            'output_equations': str(output_eqs),
            'A_matrix': str(ss.A),
            'B_matrix': str(ss.B),
            'C_matrix': str(ss.C),
            'D_matrix': str(ss.D),
            'state_variables': str(ss.x),
            'input_variables': str(ss.u),
            'output_variables': str(ss.y)
        }
        results['analysis_methods'].append('state_space')
        
    except Exception as e:
        print(f"State-space analysis failed: {e}")
        results['equations']['state_space'] = {'error': str(e)}
    
    # 5. INDIVIDUAL COMPONENT EQUATIONS
    try:
        print(f"\n{'-'*60}")
        print("5. INDIVIDUAL COMPONENT EQUATIONS")
        print(f"{'-'*60}")
        
        component_eqs = {}
        
        # Get equations for each component
        for name, cpt in cct.elements.items():
            try:
                # Voltage across component
                v_cpt = cpt.V(lcapy.s)
                i_cpt = cpt.I(lcapy.s)
                
                print(f"Component {name}:")
                print(f"  Voltage: {v_cpt}")
                print(f"  Current: {i_cpt}")
                
                component_eqs[name] = {
                    'voltage': str(v_cpt),
                    'current': str(i_cpt),
                    'type': str(type(cpt).__name__)
                }
                
            except Exception as e:
                component_eqs[name] = {'error': str(e)}
        
        results['equations']['components'] = component_eqs
        
        # Node voltages
        node_voltages = {}
        for node_name in cct.node_list:
            if node_name != '0':  # Skip ground node
                try:
                    v_node = cct[node_name].V(lcapy.s)
                    print(f"Node {node_name} voltage: {v_node}")
                    node_voltages[str(node_name)] = str(v_node)
                except Exception as e:
                    node_voltages[str(node_name)] = str(e)
        
        results['equations']['node_voltages'] = node_voltages
        
    except Exception as e:
        print(f"Component analysis failed: {e}")
        results['equations']['components'] = {'error': str(e)}
    
    # 6. TRANSFER FUNCTIONS AND FREQUENCY RESPONSE
    try:
        print(f"\n{'-'*60}")
        print("6. TRANSFER FUNCTIONS")
        print(f"{'-'*60}")
        
        # Try to find transfer functions between different nodes
        transfer_functions = {}
        nodes = [n for n in cct.node_list if n != '0']
        
        if len(nodes) >= 2:
            # Example: transfer function from first node to last node
            input_node = nodes[0]
            output_node = nodes[-1]
            
            try:
                # Transfer function H(s) = Vout(s)/Vin(s)
                H = cct.transfer(input_node, 0, output_node, 0)
                print(f"Transfer function H(s) = V{output_node}/V{input_node}: {H}")
                transfer_functions[f'H_{input_node}_to_{output_node}'] = str(H)
            except Exception as e:
                transfer_functions['error'] = str(e)
        
        results['equations']['transfer_functions'] = transfer_functions
        
    except Exception as e:
        print(f"Transfer function analysis failed: {e}")
        results['equations']['transfer_functions'] = {'error': str(e)}
    
    # 7. IMPEDANCE AND ADMITTANCE
    try:
        print(f"\n{'-'*60}")
        print("7. IMPEDANCE AND ADMITTANCE")
        print(f"{'-'*60}")
        
        impedances = {}
        nodes = [n for n in cct.node_list if n != '0']
        
        for i, node in enumerate(nodes):
            try:
                # Input impedance at each node
                Z_in = cct.impedance(node, 0)
                Y_in = cct.admittance(node, 0)
                print(f"Node {node} - Input impedance: {Z_in}")
                print(f"Node {node} - Input admittance: {Y_in}")
                
                impedances[f'node_{node}'] = {
                    'impedance': str(Z_in),
                    'admittance': str(Y_in)
                }
            except Exception as e:
                impedances[f'node_{node}'] = {'error': str(e)}
        
        results['equations']['impedances'] = impedances
        
    except Exception as e:
        print(f"Impedance analysis failed: {e}")
        results['equations']['impedances'] = {'error': str(e)}
    
    print(f"\n{'='*80}")
    print("EQUATION EXTRACTION COMPLETE")
    print(f"{'='*80}")
    
    return results

def demonstrate_simple_rc_circuit():
    """Demonstrate equation extraction on a simple RC circuit."""
    
    circuit_spice = """
    V1 1 0 step 5
    R1 1 2 1000
    C1 2 0 1e-6
    """
    
    return extract_all_symbolic_equations(circuit_spice, "Simple RC Circuit")

def demonstrate_rlc_circuit():
    """Demonstrate equation extraction on an RLC circuit."""
    
    circuit_spice = """
    V1 1 0 step 10
    R1 1 2 100
    L1 2 3 1e-3
    C1 3 0 1e-6
    """
    
    return extract_all_symbolic_equations(circuit_spice, "RLC Circuit")

def demonstrate_opamp_circuit():
    """Demonstrate equation extraction on an op-amp circuit."""
    
    circuit_spice = """
    V1 1 0 step 1
    R1 1 2 1000
    R2 2 3 10000
    O1 0 2 3
    """
    
    return extract_all_symbolic_equations(circuit_spice, "Op-Amp Circuit")

def save_results_to_file(results: Dict, filename: str):
    """Save the extracted equations to a JSON file."""
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main function to demonstrate equation extraction."""
    
    print("Lcapy Symbolic Equation Extraction Demonstration")
    print("="*80)
    
    # Demonstrate on different circuit types
    circuits_to_analyze = [
        demonstrate_simple_rc_circuit,
        demonstrate_rlc_circuit,
        # demonstrate_opamp_circuit,  # Uncomment if op-amp models are needed
    ]
    
    all_results = []
    
    for circuit_func in circuits_to_analyze:
        try:
            result = circuit_func()
            all_results.append(result)
        except Exception as e:
            print(f"Error analyzing circuit: {e}")
    
    # Save all results
    if all_results:
        save_results_to_file(all_results, 'symbolic_equations_results.json')
    
    print("\n" + "="*80)
    print("SUMMARY OF AVAILABLE EQUATION EXTRACTION METHODS:")
    print("="*80)
    print("1. Nodal Analysis: cct.nodal_analysis().nodal_equations()")
    print("2. Modified Nodal Analysis: cct.modified_nodal_analysis().matrix_equations()")
    print("3. Mesh Analysis: cct.loop_analysis().mesh_equations()")
    print("4. State-Space: cct.ss.state_equations() and cct.ss.output_equations()")
    print("5. Component Equations: cct.component_name.V(s), cct.component_name.I(s)")
    print("6. Node Voltages: cct[node].V(s)")
    print("7. Transfer Functions: cct.transfer(n1, n2, n3, n4)")
    print("8. Impedance/Admittance: cct.impedance(n1, n2), cct.admittance(n1, n2)")
    print("="*80)

if __name__ == "__main__":
    main() 