#!/usr/bin/env python3
"""
Symbolic Circuit Analysis in s-Domain with Lcapy

This script focuses on extracting symbolic equations in the s-domain (Laplace domain)
which is most useful for symbolic circuit analysis and control theory applications.

Key methods demonstrated:
1. Converting circuits to s-domain automatically 
2. Getting symbolic nodal equations in matrix form
3. Extracting system transfer functions
4. Getting impedance/admittance functions
5. State-space representation

Author: Assistant
Date: 2024
"""

import lcapy
from lcapy import Circuit
import sympy as sp
from sympy import symbols, simplify
import json
from typing import Dict, List, Tuple

def extract_symbolic_equations_s_domain(circuit_spice: str, circuit_name: str = "circuit") -> Dict:
    """
    Extract symbolic equations in s-domain from a SPICE circuit.
    
    Parameters:
    -----------
    circuit_spice : str
        SPICE netlist string
    circuit_name : str
        Circuit identifier
        
    Returns:
    --------
    Dict containing symbolic equations in s-domain
    """
    
    print(f"\n{'='*80}")
    print(f"SYMBOLIC s-DOMAIN ANALYSIS: {circuit_name.upper()}")
    print(f"{'='*80}")
    
    # Create circuit
    cct = Circuit(circuit_spice)
    print(f"Circuit netlist:\n{circuit_spice}")
    
    results = {
        'circuit_name': circuit_name,
        'netlist': circuit_spice,
        's_domain_equations': {}
    }
    
    # ========================================================================
    # 1. NODAL EQUATIONS IN s-DOMAIN 
    # ========================================================================
    try:
        print(f"\n{'-'*60}")
        print("1. NODAL EQUATIONS IN s-DOMAIN")
        print(f"{'-'*60}")
        
        # Get nodal analysis in s-domain (automatic conversion)
        na = cct.nodal_analysis()
        
        # Get symbolic equations
        nodal_eqs = na.nodal_equations()
        print("Symbolic nodal equations (KCL at each node):")
        for node, eq in nodal_eqs.items():
            print(f"  Node {node}: {eq}")
        
        # Get matrix form: A(s) * V(s) = B(s)
        A_matrix = na.A
        b_vector = na.b
        unknowns = na.unknowns
        
        print(f"\nMatrix form: A(s) * V(s) = B(s)")
        print(f"A(s) matrix:\n{A_matrix}")
        print(f"B(s) vector:\n{b_vector}")
        print(f"Unknown vector V(s):\n{unknowns}")
        
        results['s_domain_equations']['nodal'] = {
            'equations': {str(k): str(v) for k, v in nodal_eqs.items()},
            'A_matrix': str(A_matrix),
            'B_vector': str(b_vector),
            'unknowns': str(unknowns),
            'description': 'A(s) * V(s) = B(s) where V(s) are node voltages in s-domain'
        }
        
    except Exception as e:
        print(f"Nodal analysis failed: {e}")
        results['s_domain_equations']['nodal'] = {'error': str(e)}
    
    # ========================================================================
    # 2. INDIVIDUAL COMPONENT SYMBOLIC EXPRESSIONS
    # ========================================================================
    try:
        print(f"\n{'-'*60}")
        print("2. COMPONENT VOLTAGE/CURRENT EXPRESSIONS IN s-DOMAIN")
        print(f"{'-'*60}")
        
        component_expressions = {}
        
        for name, component in cct.elements.items():
            try:
                # Get voltage and current in s-domain
                V_s = component.V(lcapy.s)
                I_s = component.I(lcapy.s)
                
                print(f"Component {name}:")
                print(f"  V_{name}(s) = {V_s}")
                print(f"  I_{name}(s) = {I_s}")
                
                component_expressions[name] = {
                    'voltage_s': str(V_s),
                    'current_s': str(I_s),
                    'impedance': str(component.Z(lcapy.s)) if hasattr(component, 'Z') else 'N/A'
                }
                
            except Exception as e:
                component_expressions[name] = {'error': str(e)}
        
        results['s_domain_equations']['components'] = component_expressions
        
    except Exception as e:
        print(f"Component analysis failed: {e}")
        results['s_domain_equations']['components'] = {'error': str(e)}
    
    # ========================================================================
    # 3. NODE VOLTAGE EXPRESSIONS
    # ========================================================================
    try:
        print(f"\n{'-'*60}")
        print("3. NODE VOLTAGE EXPRESSIONS IN s-DOMAIN")
        print(f"{'-'*60}")
        
        node_voltages = {}
        for node_name in cct.node_list:
            if node_name != '0':  # Skip ground
                try:
                    V_node_s = cct[node_name].V(lcapy.s)
                    print(f"V_{node_name}(s) = {V_node_s}")
                    node_voltages[f'V_{node_name}'] = str(V_node_s)
                except Exception as e:
                    node_voltages[f'V_{node_name}'] = str(e)
        
        results['s_domain_equations']['node_voltages'] = node_voltages
        
    except Exception as e:
        print(f"Node voltage analysis failed: {e}")
        results['s_domain_equations']['node_voltages'] = {'error': str(e)}
    
    # ========================================================================
    # 4. TRANSFER FUNCTIONS
    # ========================================================================
    try:
        print(f"\n{'-'*60}")
        print("4. TRANSFER FUNCTIONS H(s)")
        print(f"{'-'*60}")
        
        transfer_functions = {}
        nodes = [n for n in cct.node_list if n != '0']
        
        # Get transfer functions between all pairs of nodes
        for i, input_node in enumerate(nodes):
            for output_node in nodes[i:]:  # Avoid duplicate pairs
                if input_node != output_node:
                    try:
                        # H(s) = Vout(s)/Vin(s)
                        H_s = cct.transfer(input_node, 0, output_node, 0)
                        tf_name = f'H_{input_node}_to_{output_node}'
                        print(f"{tf_name}(s) = V_{output_node}(s)/V_{input_node}(s) = {H_s}")
                        transfer_functions[tf_name] = str(H_s)
                    except Exception as e:
                        transfer_functions[f'H_{input_node}_to_{output_node}'] = f'Error: {str(e)}'
        
        results['s_domain_equations']['transfer_functions'] = transfer_functions
        
    except Exception as e:
        print(f"Transfer function analysis failed: {e}")
        results['s_domain_equations']['transfer_functions'] = {'error': str(e)}
    
    # ========================================================================
    # 5. IMPEDANCE AND ADMITTANCE FUNCTIONS  
    # ========================================================================
    try:
        print(f"\n{'-'*60}")
        print("5. IMPEDANCE AND ADMITTANCE FUNCTIONS Z(s), Y(s)")
        print(f"{'-'*60}")
        
        impedance_functions = {}
        nodes = [n for n in cct.node_list if n != '0']
        
        for node in nodes:
            try:
                # Input impedance/admittance at each node
                Z_s = cct.impedance(node, 0)
                Y_s = cct.admittance(node, 0)
                
                print(f"Node {node}:")
                print(f"  Z_{node}(s) = {Z_s}")
                print(f"  Y_{node}(s) = {Y_s}")
                
                impedance_functions[f'node_{node}'] = {
                    'impedance_s': str(Z_s),
                    'admittance_s': str(Y_s)
                }
                
            except Exception as e:
                impedance_functions[f'node_{node}'] = {'error': str(e)}
        
        results['s_domain_equations']['impedances'] = impedance_functions
        
    except Exception as e:
        print(f"Impedance analysis failed: {e}")
        results['s_domain_equations']['impedances'] = {'error': str(e)}
    
    # ========================================================================
    # 6. STATE-SPACE IN s-DOMAIN
    # ========================================================================
    try:
        print(f"\n{'-'*60}")
        print("6. STATE-SPACE REPRESENTATION IN s-DOMAIN")
        print(f"{'-'*60}")
        
        ss = cct.ss
        
        # State-space matrices
        A_ss = ss.A
        B_ss = ss.B  
        C_ss = ss.C
        D_ss = ss.D
        
        print(f"State-space representation:")
        print(f"A matrix:\n{A_ss}")
        print(f"B matrix:\n{B_ss}")
        print(f"C matrix:\n{C_ss}")
        print(f"D matrix:\n{D_ss}")
        
        # Transfer function from state-space: H(s) = C(sI-A)^(-1)B + D
        try:
            H_ss = ss.H
            print(f"\nTransfer function from state-space H(s):")
            print(f"{H_ss}")
        except Exception as e:
            H_ss = f"Error computing H(s): {e}"
        
        results['s_domain_equations']['state_space'] = {
            'A_matrix': str(A_ss),
            'B_matrix': str(B_ss),
            'C_matrix': str(C_ss),
            'D_matrix': str(D_ss),
            'transfer_function_H_s': str(H_ss),
            'state_variables': str(ss.x),
            'input_variables': str(ss.u),
            'output_variables': str(ss.y)
        }
        
    except Exception as e:
        print(f"State-space analysis failed: {e}")
        results['s_domain_equations']['state_space'] = {'error': str(e)}
    
    # ========================================================================
    # 7. SYMBOLIC SIMPLIFICATION
    # ========================================================================
    try:
        print(f"\n{'-'*60}")
        print("7. SIMPLIFIED SYMBOLIC EXPRESSIONS")
        print(f"{'-'*60}")
        
        simplified = {}
        
        # Simplify key expressions
        if 'transfer_functions' in results['s_domain_equations']:
            simplified_tf = {}
            for tf_name, tf_expr in results['s_domain_equations']['transfer_functions'].items():
                if not tf_expr.startswith('Error'):
                    try:
                        # Parse and simplify
                        expr = lcapy.expr(tf_expr)
                        simplified_expr = expr.simplify()
                        simplified_tf[tf_name] = str(simplified_expr)
                        print(f"{tf_name}_simplified = {simplified_expr}")
                    except:
                        simplified_tf[tf_name] = tf_expr
            simplified['transfer_functions'] = simplified_tf
        
        results['s_domain_equations']['simplified'] = simplified
        
    except Exception as e:
        print(f"Simplification failed: {e}")
        results['s_domain_equations']['simplified'] = {'error': str(e)}
    
    print(f"\n{'='*80}")
    print("s-DOMAIN SYMBOLIC ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return results

def demonstrate_symbolic_rc_circuit():
    """RC circuit with symbolic component values."""
    
    print("\n" + "="*80)
    print("DEMONSTRATION: RC CIRCUIT WITH SYMBOLIC VALUES")
    print("="*80)
    
    # Use symbolic values
    circuit_spice = """
    V1 1 0 {V_s}
    R1 1 2 {R}
    C1 2 0 {C}
    """
    
    return extract_symbolic_equations_s_domain(circuit_spice, "Symbolic RC Circuit")

def demonstrate_symbolic_rlc_circuit():
    """RLC circuit with symbolic values."""
    
    print("\n" + "="*80)
    print("DEMONSTRATION: RLC CIRCUIT WITH SYMBOLIC VALUES") 
    print("="*80)
    
    circuit_spice = """
    V1 1 0 {V_s}
    R1 1 2 {R1}
    L1 2 3 {L}
    C1 3 0 {C}
    R2 3 0 {R2}
    """
    
    return extract_symbolic_equations_s_domain(circuit_spice, "Symbolic RLC Circuit")

def save_equations_to_latex(results: Dict, filename: str):
    """Convert symbolic equations to LaTeX format."""
    
    latex_output = []
    latex_output.append("\\\\documentclass{article}")
    latex_output.append("\\\\usepackage{amsmath}")
    latex_output.append("\\\\begin{document}")
    latex_output.append(f"\\\\title{{Symbolic Circuit Analysis: {results['circuit_name']}}}")
    latex_output.append("\\\\maketitle")
    
    # Add equations sections
    if 'nodal' in results['s_domain_equations']:
        latex_output.append("\\\\section{Nodal Equations}")
        nodal = results['s_domain_equations']['nodal']
        if 'equations' in nodal:
            for node, eq in nodal['equations'].items():
                latex_output.append(f"\\\\subsection{{Node {node}}}")
                latex_output.append("\\\\begin{equation}")
                # Convert to LaTeX format (simplified)
                latex_eq = str(eq).replace('*', ' \\\\cdot ').replace('s', 's')
                latex_output.append(latex_eq)
                latex_output.append("\\\\end{equation}")
    
    latex_output.append("\\\\end{document}")
    
    try:
        with open(filename, 'w') as f:
            f.write('\n'.join(latex_output))
        print(f"LaTeX equations saved to: {filename}")
    except Exception as e:
        print(f"Error saving LaTeX: {e}")

def main():
    """Main demonstration function."""
    
    print("Lcapy Symbolic s-Domain Circuit Analysis")
    print("="*80)
    
    # Demonstrate with symbolic component values
    results = []
    
    # RC Circuit
    try:
        rc_result = demonstrate_symbolic_rc_circuit()
        results.append(rc_result)
    except Exception as e:
        print(f"RC circuit analysis failed: {e}")
    
    # RLC Circuit  
    try:
        rlc_result = demonstrate_symbolic_rlc_circuit()
        results.append(rlc_result)
    except Exception as e:
        print(f"RLC circuit analysis failed: {e}")
    
    # Save results
    if results:
        with open('symbolic_s_domain_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to: symbolic_s_domain_analysis.json")
        
        # Save LaTeX version for the first circuit
        if len(results) > 0:
            save_equations_to_latex(results[0], 'symbolic_equations.tex')
    
    print("\n" + "="*80)
    print("KEY METHODS FOR SYMBOLIC s-DOMAIN EXTRACTION:")
    print("="*80)
    print("• cct.nodal_analysis().nodal_equations() - Gets KCL equations")
    print("• cct.nodal_analysis().A, .b - Matrix form A(s)*V(s) = B(s)")
    print("• cct[node].V(lcapy.s) - Node voltage in s-domain")
    print("• component.V(lcapy.s), .I(lcapy.s) - Component V,I in s-domain")
    print("• cct.transfer(n1, n2, n3, n4) - Transfer function H(s)")
    print("• cct.impedance(n1, n2) - Impedance function Z(s)")
    print("• cct.ss.A, .B, .C, .D - State-space matrices")
    print("• expr.simplify() - Simplify symbolic expressions")
    print("="*80)

if __name__ == "__main__":
    main() 