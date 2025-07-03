#!/usr/bin/env python3
"""
Symbolic Equation Extraction for Circuit Analysis

This script extracts symbolic mathematical expressions from SPICE netlists 
using lcapy with a Modified Nodal Analysis (MNA) compatible approach.

Key Features:
- Extracts symbolic nodal equations (KCL at each node)
- Computes symbolic node voltage expressions V_n(s)
- Derives component voltage-current relationships
- Calculates symbolic transfer functions H(s)
- Outputs human-readable mathematical expressions suitable for MLLM training

The script uses MNA methodology for systematic analysis while focusing on
extracting actual symbolic equations rather than just matrix structures.
"""

import json
import re
import signal
from pathlib import Path
import lcapy
from lcapy import Circuit
import sympy as sp
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
import argparse

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Analysis timed out")

def estimate_mna_matrix_size(cleaned_netlist: str) -> Dict:
    """
    Estimate the Modified Nodal Analysis matrix size based on circuit components.
    
    MNA matrix includes:
    - Node voltages (except ground)
    - Currents through voltage sources (independent + controlled voltage sources)
    - Currents through inductors
    
    Args:
        cleaned_netlist: Cleaned SPICE netlist
        
    Returns:
        Dictionary with matrix size estimation
    """
    lines = [l.strip() for l in cleaned_netlist.split('\n') if l.strip()]
    
    # Count unique nodes (excluding ground '0')
    nodes = set()
    voltage_sources = 0  # V, E (VCVS), H (CCVS)
    inductors = 0        # L
    current_sources = 0  # I, F (CCCS), G (VCCS)
    resistors = 0        # R
    capacitors = 0       # C
    
    for line in lines:
        if line.startswith('.title'):
            continue
            
        parts = line.split()
        if len(parts) < 3:
            continue
            
        component_name = parts[0]
        node1, node2 = parts[1], parts[2]
        
        # Add nodes (excluding ground)
        if node1 != '0':
            nodes.add(node1)
        if node2 != '0':
            nodes.add(node2)
        
        first_char = component_name[0].upper()
        
        if first_char == 'V' or component_name.startswith('V_meas'):
            voltage_sources += 1
        elif first_char == 'E':  # VCVS
            voltage_sources += 1
        elif first_char == 'H':  # CCVS  
            voltage_sources += 1
        elif first_char == 'I':
            current_sources += 1
        elif first_char == 'F':  # CCCS
            current_sources += 1
        elif first_char == 'G':  # VCCS
            current_sources += 1
        elif first_char == 'L':
            inductors += 1
        elif first_char == 'R':
            resistors += 1
        elif first_char == 'C':
            capacitors += 1
    
    num_nodes = len(nodes)
    
    # MNA matrix size = num_nodes + voltage_source_currents + inductor_currents
    mna_size = num_nodes + voltage_sources + inductors
    
    return {
        'num_nodes': num_nodes,
        'voltage_sources': voltage_sources,
        'inductors': inductors,
        'current_sources': current_sources,
        'resistors': resistors,
        'capacitors': capacitors,
        'mna_matrix_size': mna_size,
        'total_components': voltage_sources + inductors + current_sources + resistors + capacitors,
        'reactive_components': inductors + capacitors,
        'complexity_score': mna_size * (1 + 0.5 * (inductors + capacitors))  # Higher for reactive circuits
    }

def categorize_circuit_analysis_mode(cleaned_netlist: str) -> Dict:
    """
    Determine what analysis modes lcapy will likely use for this circuit.
    
    Args:
        cleaned_netlist: Cleaned SPICE netlist
        
    Returns:
        Dictionary with analysis mode predictions
    """
    lines = [l.strip() for l in cleaned_netlist.split('\n') if l.strip()]
    
    has_capacitors = False
    has_inductors = False
    has_ac_sources = False
    has_dc_sources = False
    has_controlled_sources = False
    
    for line in lines:
        if line.startswith('.title'):
            continue
            
        parts = line.split()
        if len(parts) < 3:
            continue
            
        component_name = parts[0]
        first_char = component_name[0].upper()
        
        if first_char == 'C':
            has_capacitors = True
        elif first_char == 'L':
            has_inductors = True
        elif first_char in ['E', 'F', 'G', 'H']:
            has_controlled_sources = True
        elif first_char == 'V':
            # Simple heuristic: if voltage is 0, it's likely a measurement probe
            if len(parts) >= 4 and parts[3] == '0':
                continue  # Measurement voltage source
            else:
                has_dc_sources = True
    
    analysis_modes = {
        'is_reactive': has_capacitors or has_inductors,
        'has_controlled_sources': has_controlled_sources,
        'likely_analysis_modes': [],
        'complexity_level': 'low'
    }
    
    # Determine likely analysis modes
    if not has_capacitors and not has_inductors:
        analysis_modes['likely_analysis_modes'].append('dc_resistive')
        analysis_modes['complexity_level'] = 'low'
    else:
        if has_dc_sources:
            analysis_modes['likely_analysis_modes'].append('dc_analysis')
        analysis_modes['likely_analysis_modes'].append('laplace_transient')
        if has_capacitors and has_inductors:
            analysis_modes['complexity_level'] = 'high'
        else:
            analysis_modes['complexity_level'] = 'medium'
    
    if has_controlled_sources:
        analysis_modes['complexity_level'] = 'very_high'
        analysis_modes['likely_analysis_modes'].append('modified_nodal_analysis')
    
    return analysis_modes

def clean_spice_for_lcapy_with_controlled_sources(spice_netlist: str) -> str:
    """
    Clean SPICE netlist for lcapy while preserving controlled sources.
    
    Args:
        spice_netlist: Raw SPICE netlist string
        
    Returns:
        Cleaned netlist string for lcapy with controlled sources
    """
    lines = spice_netlist.strip().split('\n')
    cleaned_lines = []
    
    # Track measurement voltage sources to properly reference them
    measurement_sources = {}  # VI1 -> V_VI1 mapping
    
    # Extract and preserve title directive
    title_line = None
    for line in lines:
        if line.strip().startswith('.title'):
            title_line = line.strip()
            break
    
    # Add title if found (critical for correct ngspice simulation)
    if title_line:
        cleaned_lines.append(title_line)
    
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
    
    # Single pass: process all components in order
    for line in lines:
        line = line.strip()
        
        # Skip empty lines, comments, and SPICE commands (but preserve .title)
        if (not line or line.startswith('*') or 
            (line.startswith('.') and not line.startswith('.title')) or 
            line.startswith('print') or ';' in line):
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
            
        component_name = parts[0]
        first_char = component_name[0].upper()
        
        # Convert voltage measurement sources (VI) to regular voltage sources
        if component_name.startswith('VI'):
            if len(parts) >= 4:
                # VI1 node1 node2 0 -> V_meas1 node1 node2 0
                new_name = f"V_meas{component_name[2:]}"
                measurement_sources[component_name] = new_name
                node1, node2 = parts[1], parts[2]
                
                # Clean node names
                node1 = re.sub(r'[^\w]', '_', node1)
                node2 = re.sub(r'[^\w]', '_', node2)
                
                cleaned_lines.append(f"{new_name} {node1} {node2} 0")
        
        # Handle basic components (R, L, C, V, I)
        elif first_char in ['R', 'L', 'C', 'V', 'I']:
            if len(parts) >= 4:
                name = parts[0]
                node1, node2 = parts[1], parts[2]
                value = convert_spice_value(parts[3])
                
                # Clean node names
                node1 = re.sub(r'[^\w]', '_', node1)
                node2 = re.sub(r'[^\w]', '_', node2)
                
                cleaned_lines.append(f"{name} {node1} {node2} {value}")
        
        # Handle Voltage Controlled Voltage Source (VCVS) - E
        elif first_char == 'E':
            if len(parts) >= 6:
                name = parts[0]
                out_pos, out_neg = parts[1], parts[2]  # Output nodes
                in_pos, in_neg = parts[3], parts[4]   # Control nodes  
                gain = convert_spice_value(parts[5])
                
                # Clean node names
                out_pos = re.sub(r'[^\w]', '_', out_pos)
                out_neg = re.sub(r'[^\w]', '_', out_neg)
                in_pos = re.sub(r'[^\w]', '_', in_pos)
                in_neg = re.sub(r'[^\w]', '_', in_neg)
                
                # Lcapy syntax for VCVS: E<name> <n+> <n-> <nc+> <nc-> <gain>
                cleaned_lines.append(f"{name} {out_pos} {out_neg} {in_pos} {in_neg} {gain}")
        
        # Handle Voltage Controlled Current Source (VCCS) - G  
        elif first_char == 'G':
            if len(parts) >= 6:
                name = parts[0]
                out_pos, out_neg = parts[1], parts[2]  # Output nodes
                in_pos, in_neg = parts[3], parts[4]   # Control nodes
                gain = convert_spice_value(parts[5])
                
                # Clean node names
                out_pos = re.sub(r'[^\w]', '_', out_pos)
                out_neg = re.sub(r'[^\w]', '_', out_neg)
                in_pos = re.sub(r'[^\w]', '_', in_pos)
                in_neg = re.sub(r'[^\w]', '_', in_neg)
                
                # Lcapy syntax for VCCS: G<name> <n+> <n-> <nc+> <nc-> <gain>
                cleaned_lines.append(f"{name} {out_pos} {out_neg} {in_pos} {in_neg} {gain}")
        
        # Handle Current Controlled Current Source (CCCS) - F
        elif first_char == 'F':
            if len(parts) >= 5:
                name = parts[0]
                out_pos, out_neg = parts[1], parts[2]  # Output nodes
                control_source = parts[3]              # Control current source
                gain = convert_spice_value(parts[4])
                
                # Clean node names
                out_pos = re.sub(r'[^\w]', '_', out_pos)
                out_neg = re.sub(r'[^\w]', '_', out_neg)
                
                # Convert control source reference
                if control_source in measurement_sources:
                    control_source = measurement_sources[control_source]
                
                # Lcapy syntax for CCCS: F<name> <n+> <n-> <vcontrol> <gain>
                cleaned_lines.append(f"{name} {out_pos} {out_neg} {control_source} {gain}")
        
        # Handle Current Controlled Voltage Source (CCVS) - H
        elif first_char == 'H':
            if len(parts) >= 5:
                name = parts[0]
                out_pos, out_neg = parts[1], parts[2]  # Output nodes
                control_source = parts[3]              # Control current source
                gain = convert_spice_value(parts[4])
                
                # Clean node names
                out_pos = re.sub(r'[^\w]', '_', out_pos)
                out_neg = re.sub(r'[^\w]', '_', out_neg)
                
                # Convert control source reference
                if control_source in measurement_sources:
                    control_source = measurement_sources[control_source]
                
                # Lcapy syntax for CCVS: H<name> <n+> <n-> <vcontrol> <gain>
                cleaned_lines.append(f"{name} {out_pos} {out_neg} {control_source} {gain}")
    
    return '\n'.join(cleaned_lines)

def clean_spice_for_lcapy_simple(spice_netlist: str) -> str:
    """
    Clean SPICE netlist for lcapy - simplified version that handles only basic components.
    
    Args:
        spice_netlist: Raw SPICE netlist string
        
    Returns:
        Cleaned netlist string for lcapy
    """
    lines = spice_netlist.strip().split('\n')
    cleaned_lines = []
    
    # Extract and preserve title directive
    title_line = None
    for line in lines:
        if line.strip().startswith('.title'):
            title_line = line.strip()
            break
    
    # Add title if found (critical for correct ngspice simulation)
    if title_line:
        cleaned_lines.append(title_line)
    
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
        
        # Skip empty lines, comments, and SPICE commands (but preserve .title)
        if (not line or line.startswith('*') or 
            (line.startswith('.') and not line.startswith('.title')) or 
            line.startswith('print') or ';' in line):
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
            
        component_name = parts[0]
        first_char = component_name[0].upper()
        
        # Convert voltage measurement sources (VI) to 0V voltage sources
        if component_name.startswith('VI'):
            if len(parts) >= 4:
                # VI1 node1 node2 0 -> V_meas1 node1 node2 0
                new_name = f"V_meas{component_name[2:]}"
                node1, node2 = parts[1], parts[2]
                
                # Clean node names but preserve them
                node1 = re.sub(r'[^\w]', '_', node1)
                node2 = re.sub(r'[^\w]', '_', node2)
                
                cleaned_lines.append(f"{new_name} {node1} {node2} 0")
        
        # Skip controlled sources completely - they're too complex for basic analysis
        elif first_char in ['E', 'F', 'G', 'H']:
            continue
        
        # Handle only basic components (R, L, C, V, I)
        elif first_char in ['R', 'L', 'C', 'V', 'I']:
            if len(parts) >= 4:
                name = parts[0]
                node1, node2 = parts[1], parts[2]
                value = convert_spice_value(parts[3])
                
                # Clean node names - replace problematic characters but PRESERVE ALL NODES
                node1 = re.sub(r'[^\w]', '_', node1)
                node2 = re.sub(r'[^\w]', '_', node2)
                
                # KEEP ALL COMPONENTS - don't filter based on node names
                cleaned_lines.append(f"{name} {node1} {node2} {value}")
    
    return '\n'.join(cleaned_lines)

def analyze_circuit_with_timeout(spice_netlist: str, circuit_id: str, timeout_seconds: int = 30, 
                                include_controlled_sources: bool = True) -> Dict:
    """
    Analyze circuit with timeout protection using MNA complexity estimation.
    
    Args:
        spice_netlist: SPICE netlist string
        circuit_id: Circuit identifier
        timeout_seconds: Timeout in seconds
        include_controlled_sources: Whether to include controlled sources in analysis
        
    Returns:
        Dictionary containing analysis results
    """
    
    # Clean netlist first to get accurate complexity estimate
    if include_controlled_sources:
        cleaned_netlist = clean_spice_for_lcapy_with_controlled_sources(spice_netlist)
    else:
        cleaned_netlist = clean_spice_for_lcapy_simple(spice_netlist)
    
    if not cleaned_netlist.strip():
        return {
            'circuit_id': circuit_id,
            'status': 'error',
            'error': 'No valid components found after cleaning',
            'original_netlist': spice_netlist
        }
    
    # Estimate MNA matrix size and circuit complexity
    mna_info = estimate_mna_matrix_size(cleaned_netlist)
    analysis_mode_info = categorize_circuit_analysis_mode(cleaned_netlist)
    
    # More sophisticated complexity check based on MNA matrix size
    matrix_size = mna_info['mna_matrix_size']
    complexity_score = mna_info['complexity_score']
    
    # Skip circuits that will definitely cause timeout
    if matrix_size > 12:
        return {
            'circuit_id': circuit_id,
            'status': 'skipped',
            'error': f'MNA matrix too large ({matrix_size}x{matrix_size}) - would cause timeout',
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist,
            'mna_analysis': mna_info,
            'predicted_analysis_mode': analysis_mode_info
        }
    
    if complexity_score > 50:  # High complexity score threshold
        return {
            'circuit_id': circuit_id,
            'status': 'skipped',
            'error': f'Circuit complexity too high (score: {complexity_score:.1f}) - would cause timeout',
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist,
            'mna_analysis': mna_info,
            'predicted_analysis_mode': analysis_mode_info
        }
    
    # Set up timeout with shorter time for complex circuits
    timeout = min(timeout_seconds, 10 if matrix_size > 8 else timeout_seconds)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        if include_controlled_sources:
            result = extract_symbolic_equations_with_controlled_sources(spice_netlist, circuit_id, 
                                                                      mna_info, analysis_mode_info)
        else:
            result = extract_symbolic_equations_simple(spice_netlist, circuit_id, 
                                                     mna_info, analysis_mode_info)
            
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutException:
        signal.alarm(0)  # Cancel the alarm
        return {
            'circuit_id': circuit_id,
            'status': 'timeout',
            'error': f'Analysis timed out after {timeout} seconds',
            'original_netlist': spice_netlist
        }
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        return {
            'circuit_id': circuit_id,
            'status': 'error',
            'error': str(e),
            'original_netlist': spice_netlist
        }

def extract_symbolic_equations_with_controlled_sources(spice_netlist: str, circuit_id: str, 
                                                    mna_info: Dict, analysis_mode_info: Dict) -> Dict:
    """
    Extract symbolic equations from a circuit including controlled sources.
    
    Args:
        spice_netlist: SPICE netlist string
        circuit_id: Circuit identifier
        mna_info: Pre-computed MNA matrix size information
        analysis_mode_info: Pre-computed analysis mode information
        
    Returns:
        Dictionary containing symbolic analysis results
    """
    
    # Clean the netlist for lcapy (with controlled sources)
    cleaned_netlist = clean_spice_for_lcapy_with_controlled_sources(spice_netlist)
    
    if not cleaned_netlist.strip():
        return {
            'circuit_id': circuit_id,
            'status': 'error',
            'error': 'No valid components found after cleaning',
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist,
            'mna_analysis': mna_info,
            'predicted_analysis_mode': analysis_mode_info
        }
    
    # Check if circuit has enough components
    lines = [l for l in cleaned_netlist.split('\n') if l.strip()]
    if len(lines) < 2:
        return {
            'circuit_id': circuit_id,
            'status': 'error', 
            'error': f'Circuit too simple - only {len(lines)} components',
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist,
            'mna_analysis': mna_info,
            'predicted_analysis_mode': analysis_mode_info
        }
    
    try:
        # Create lcapy circuit
        cct = Circuit(cleaned_netlist)
        
        # Check node count - skip if too many nodes (causes large matrices)
        if len(cct.node_list) > 8:
            return {
                'circuit_id': circuit_id,
                'status': 'skipped',
                'error': f'Too many nodes ({len(cct.node_list)}) - would cause large matrix inversion',
                'original_netlist': spice_netlist,
                'cleaned_netlist': cleaned_netlist,
                'mna_analysis': mna_info,
                'predicted_analysis_mode': analysis_mode_info
            }
        
        results = {
            'circuit_id': circuit_id,
            'status': 'success',
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist,
            'mna_analysis': mna_info,
            'predicted_analysis_mode': analysis_mode_info,
            'symbolic_equations': {},
            'circuit_properties': {
                'nodes': str(cct.node_list),
                'elements': str(list(cct.elements.keys())),
                'num_nodes': len(cct.node_list),
                'num_elements': len(cct.elements),
                'mna_matrix_size': mna_info['mna_matrix_size'],
                'complexity_score': mna_info['complexity_score']
            }
        }
        
        # Categorize circuit elements
        elements = {
            'basic': [],
            'controlled_sources': [],
            'independent_sources': []
        }
        
        for element_name, element in cct.elements.items():
            element_type = element_name[0].upper()
            if element_type in ['E', 'F', 'G', 'H']:
                elements['controlled_sources'].append(element_name)
            elif element_type in ['V', 'I']:
                elements['independent_sources'].append(element_name)
            else:
                elements['basic'].append(element_name)
        
        results['circuit_properties']['element_categories'] = elements
        
        # Try to get basic circuit properties first
        try:
            results['circuit_properties']['is_dc'] = cct.is_dc
        except:
            results['circuit_properties']['is_dc'] = 'unknown'
        
        # SKIP HEAVY SYMBOLIC COMPUTATIONS for complex circuits
        if len(cct.node_list) > 6 or len(elements['controlled_sources']) > 2:
            results['symbolic_equations']['note'] = 'Symbolic analysis skipped - circuit too complex'
            return results
        
        # 1. TRY SYMBOLIC EQUATION EXTRACTION (MNA-compatible approach)
        try:
            # For MLLM training, we need actual symbolic equations, not just matrix structures
            if len(elements['controlled_sources']) > 0:
                results['analysis_method'] = 'symbolic_mna_with_controlled_sources'
                
                # Extract symbolic equations even with controlled sources
                try:
                    # Get symbolic nodal equations first (most important for MLLMs)
                    try:
                        na = cct.nodal_analysis()  # Still need symbolic equations
                        nodal_eqs = na.nodal_equations()
                        
                        if nodal_eqs:
                            symbolic_nodal = {}
                            for node, equation in nodal_eqs.items():
                                eq_str = str(equation)
                                symbolic_nodal[f'Node_{node}'] = {
                                    'equation': eq_str,
                                    'description': f'KCL equation at node {node}',
                                    'method': 'MNA-compatible symbolic analysis'
                                }
                            
                            results['symbolic_equations']['nodal_equations'] = {
                                'equations': symbolic_nodal,
                                'count': len(symbolic_nodal),
                                'description': 'Symbolic KCL equations (suitable for MLLM training)',
                                'formulation': 'Modified Nodal Analysis approach'
                            }
                        
                    except Exception as e:
                        results['symbolic_equations']['nodal_equations'] = {'error': str(e)}
                    
                    # Get node voltage expressions (symbolic)
                    node_voltages = {}
                    nodes_to_check = [node for node in cct.node_list if node != '0'][:3]
                    
                    for node in nodes_to_check:
                        try:
                            V_node = cct[node].V(lcapy.s)
                            node_voltages[f'V_{node}'] = {
                                's_domain': str(V_node),
                                'description': f'Symbolic voltage at node {node}'
                            }
                        except Exception as e:
                            node_voltages[f'V_{node}'] = {'error': str(e)}
                    
                    results['symbolic_equations']['node_voltages'] = node_voltages
                    
                    # Try to get component equations (symbolic)
                    component_eqs = {}
                    for element_name, element in list(cct.elements.items())[:3]:
                        try:
                            V_comp = element.V(lcapy.s)
                            I_comp = element.I(lcapy.s)
                            component_eqs[element_name] = {
                                'voltage_s': str(V_comp),
                                'current_s': str(I_comp),
                                'type': type(element).__name__
                            }
                        except Exception as e:
                            component_eqs[element_name] = {'error': str(e)}
                    
                    results['symbolic_equations']['component_equations'] = component_eqs
                    
                except Exception as e:
                    results['symbolic_equations']['controlled_source_analysis_error'] = str(e)
            
            else:
                # For basic circuits, extract full symbolic analysis
                results['analysis_method'] = 'symbolic_mna_basic'
                
                if len(cct.node_list) <= 5:  # Only for small circuits
                    try:
                        # 1. Symbolic nodal equations (KCL)
                        na = cct.nodal_analysis()
                        nodal_eqs = na.nodal_equations()
                        
                        if nodal_eqs:
                            symbolic_nodal = {}
                            for node, equation in nodal_eqs.items():
                                eq_str = str(equation)
                                symbolic_nodal[f'Node_{node}'] = {
                                    'equation': eq_str,
                                    'description': f'KCL: sum of currents at node {node} = 0'
                                }
                            
                            results['symbolic_equations']['nodal_equations'] = {
                                'equations': symbolic_nodal,
                                'count': len(symbolic_nodal),
                                'description': 'Symbolic nodal equations for MLLM training'
                            }
                        
                        # 2. Node voltage expressions
                        node_voltages = {}
                        nodes = [n for n in cct.node_list if n != '0'][:3]
                        for node in nodes:
                            try:
                                V_node = cct[node].V(lcapy.s)
                                node_voltages[f'V_{node}'] = {
                                    's_domain': str(V_node),
                                    'description': f'Symbolic voltage expression at node {node}'
                                }
                            except Exception as e:
                                node_voltages[f'V_{node}'] = {'error': str(e)}
                        
                        results['symbolic_equations']['node_voltages'] = node_voltages
                        
                        # 3. Component equations  
                        component_eqs = {}
                        for element_name, element in list(cct.elements.items())[:3]:
                            try:
                                V_comp = element.V(lcapy.s)
                                I_comp = element.I(lcapy.s)
                                component_eqs[element_name] = {
                                    'voltage_s': str(V_comp),
                                    'current_s': str(I_comp),
                                    'type': type(element).__name__,
                                    'expressions': {
                                        'V': f'V_{element_name}(s) = {str(V_comp)}',
                                        'I': f'I_{element_name}(s) = {str(I_comp)}'
                                    }
                                }
                            except Exception as e:
                                component_eqs[element_name] = {'error': str(e)}
                        
                        results['symbolic_equations']['component_equations'] = component_eqs
                        
                        # 4. Transfer function (if possible)
                        nodes = [n for n in cct.node_list if n != '0']
                        if len(nodes) >= 2:
                            try:
                                H_s = cct.transfer(nodes[0], 0, nodes[-1], 0)
                                results['symbolic_equations']['transfer_function'] = {
                                    f'H_{nodes[0]}_to_{nodes[-1]}': {
                                        's_domain': str(H_s),
                                        'description': f'Transfer function from {nodes[0]} to {nodes[-1]}'
                                    }
                                }
                            except Exception as e:
                                results['symbolic_equations']['transfer_function'] = {'error': str(e)}
                        
                    except Exception as e:
                        results['symbolic_equations']['symbolic_analysis_error'] = str(e)
                        
        except Exception as e:
            results['symbolic_equations']['symbolic_error'] = str(e)
        
        # 2. SKIP TRANSFER FUNCTIONS (too complex for circuits with C/L)
        results['symbolic_equations']['transfer_function'] = 'Skipped - too complex for reactive circuits'
        
        # 3. BASIC COMPONENT ANALYSIS - LIMITED
        try:
            component_info = {}
            # Only analyze first 5 components
            for i, (element_name, element) in enumerate(list(cct.elements.items())[:5]):
                try:
                    component_info[element_name] = {
                        'type': type(element).__name__,
                        'nodes': str(element.nodes),
                    }
                    # Skip voltage/current analysis to avoid symbolic computation
                        
                except:
                    component_info[element_name] = {'type': 'unknown'}
            
            results['symbolic_equations']['components'] = component_info
            
        except Exception as e:
            results['symbolic_equations']['component_error'] = str(e)
        
        return results
        
    except Exception as e:
        return {
            'circuit_id': circuit_id,
            'status': 'error',
            'error': str(e),
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist,
            'mna_analysis': mna_info,
            'predicted_analysis_mode': analysis_mode_info
        }

def extract_symbolic_equations_simple(spice_netlist: str, circuit_id: str, 
                                     mna_info: Dict, analysis_mode_info: Dict) -> Dict:
    """
    Extract symbolic equations from a circuit - simplified version without controlled sources.
    
    Args:
        spice_netlist: SPICE netlist string
        circuit_id: Circuit identifier
        mna_info: Pre-computed MNA matrix size information
        analysis_mode_info: Pre-computed analysis mode information
        
    Returns:
        Dictionary containing symbolic analysis results
    """
    
    # Clean the netlist for lcapy
    cleaned_netlist = clean_spice_for_lcapy_simple(spice_netlist)
    
    if not cleaned_netlist.strip():
        return {
            'circuit_id': circuit_id,
            'status': 'error',
            'error': 'No basic components found after cleaning (all controlled sources)',
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist,
            'mna_analysis': mna_info,
            'predicted_analysis_mode': analysis_mode_info
        }
    
    # Check if circuit has enough components
    lines = [l for l in cleaned_netlist.split('\n') if l.strip()]
    if len(lines) < 2:
        return {
            'circuit_id': circuit_id,
            'status': 'error', 
            'error': f'Circuit too simple - only {len(lines)} basic components',
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist,
            'mna_analysis': mna_info,
            'predicted_analysis_mode': analysis_mode_info
        }
    
    try:
        # Create lcapy circuit
        cct = Circuit(cleaned_netlist)
        
        results = {
            'circuit_id': circuit_id,
            'status': 'success',
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist,
            'mna_analysis': mna_info,
            'predicted_analysis_mode': analysis_mode_info,
            'symbolic_equations': {},
            'circuit_properties': {
                'nodes': str(cct.node_list),
                'elements': str(list(cct.elements.keys())),
                'num_nodes': len(cct.node_list),
                'num_elements': len(cct.elements),
                'mna_matrix_size': mna_info['mna_matrix_size'],
                'complexity_score': mna_info['complexity_score']
            }
        }
        
        # Try to get basic circuit properties first
        try:
            results['circuit_properties']['is_dc'] = cct.is_dc
        except:
            results['circuit_properties']['is_dc'] = 'unknown'
        
        # 1. TRY SYMBOLIC EQUATION EXTRACTION (using MNA methodology)
        try:
            # Extract actual symbolic equations for MLLM training
            results['symbolic_equations']['analysis_method'] = 'MNA-compatible symbolic analysis'
            
            # 1. Symbolic nodal equations (KCL at each node)
            try:
                na = cct.nodal_analysis()
                nodal_eqs = na.nodal_equations()
                
                if nodal_eqs:
                    symbolic_nodal = {}
                    for node, equation in nodal_eqs.items():
                        eq_str = str(equation)
                        symbolic_nodal[f'Node_{node}'] = {
                            'equation': eq_str,
                            'description': f'KCL equation at node {node}: sum of currents = 0',
                            'interpretation': 'Kirchhoff Current Law'
                        }
                    
                    results['symbolic_equations']['nodal_equations'] = {
                        'equations': symbolic_nodal,
                        'count': len(symbolic_nodal),
                        'description': 'Symbolic KCL equations suitable for MLLM training',
                        'formulation': 'Modified Nodal Analysis compatible'
                    }
                else:
                    results['symbolic_equations']['nodal_equations'] = {'error': 'No nodal equations found'}
                    
            except Exception as e:
                results['symbolic_equations']['nodal_equations'] = {'error': str(e)}
            
            # 2. Try to get MNA matrices for comparison
            try:
                mna = cct.modified_nodal_analysis()
                
                results['symbolic_equations']['mna_info'] = {
                    'description': 'MNA provides systematic matrix formulation for these symbolic equations',
                    'matrix_form': '[G B; C D] * [v; j] = [i; e]',
                    'advantages': ['Handles voltage sources directly', 'Includes branch currents', 'Used by SPICE']
                }
                
                # Try to get matrix equation
                try:
                    mna_eqs = mna.matrix_equations()
                    results['symbolic_equations']['mna_info']['matrix_equation'] = str(mna_eqs)
                except Exception as e:
                    results['symbolic_equations']['mna_info']['matrix_equation'] = f'Matrix error: {str(e)}'
                    
            except Exception as e:
                results['symbolic_equations']['mna_info'] = {'error': str(e)}
                
        except Exception as e:
            results['symbolic_equations']['symbolic_extraction_error'] = str(e)
        
        # 2. SYMBOLIC NODE VOLTAGE EXPRESSIONS 
        try:
            node_voltages = {}
            nodes = [n for n in cct.node_list if n != '0'][:3]  # Limit to first 3 nodes
            
            for node_name in nodes:
                try:
                    V_node = cct[node_name].V(lcapy.s)
                    voltage_str = str(V_node)
                    node_voltages[f'V_{node_name}'] = {
                        's_domain': voltage_str,
                        'description': f'Symbolic voltage expression at node {node_name}',
                        'equation': f'V_{node_name}(s) = {voltage_str}'
                    }
                except Exception as e:
                    node_voltages[f'V_{node_name}'] = {'error': str(e)}
            
            results['symbolic_equations']['node_voltages'] = {
                'voltages': node_voltages,
                'count': len([v for v in node_voltages.values() if 'error' not in v]),
                'description': 'Symbolic node voltage expressions for MLLM training'
            }
                     
        except Exception as e:
            results['symbolic_equations']['node_voltages'] = {'error': str(e)}
        
        # 3. SYMBOLIC COMPONENT EQUATIONS (V-I relationships)
        try:
            component_equations = {}
            # Limit to first 3 components to avoid timeout
            for i, (name, component) in enumerate(list(cct.elements.items())[:3]):
                try:
                    V_comp = component.V(lcapy.s)
                    I_comp = component.I(lcapy.s)
                    
                    voltage_str = str(V_comp)
                    current_str = str(I_comp)
                    
                    component_equations[name] = {
                        'voltage_s': voltage_str,
                        'current_s': current_str,
                        'type': type(component).__name__,
                        'expressions': {
                            'voltage_equation': f'V_{name}(s) = {voltage_str}',
                            'current_equation': f'I_{name}(s) = {current_str}'
                        },
                        'description': f'Symbolic V-I relationship for {name}'
                    }
                except Exception as e:
                    component_equations[name] = {'error': str(e), 'type': 'unknown'}
            
            results['symbolic_equations']['component_equations'] = {
                'components': component_equations,
                'count': len([c for c in component_equations.values() if 'error' not in c]),
                'description': 'Symbolic voltage-current relationships for each component'
            }
             
        except Exception as e:
            results['symbolic_equations']['component_equations'] = {'error': str(e)}
        
        # 4. SYMBOLIC TRANSFER FUNCTION
        try:
            nodes = [n for n in cct.node_list if n != '0']
            if len(nodes) >= 2:
                input_node = nodes[0]
                output_node = nodes[-1]  # Use last node as output
                try:
                    H_s = cct.transfer(input_node, 0, output_node, 0)
                    transfer_str = str(H_s)
                    
                    results['symbolic_equations']['transfer_function'] = {
                        'function': {
                            's_domain': transfer_str,
                            'equation': f'H(s) = V_{output_node}(s) / V_{input_node}(s) = {transfer_str}',
                            'input_node': input_node,
                            'output_node': output_node,
                            'description': f'Transfer function from node {input_node} to node {output_node}'
                        },
                        'interpretation': 'Symbolic transfer function - excellent for MLLM frequency domain analysis'
                    }
                except Exception as e:
                    results['symbolic_equations']['transfer_function'] = {'error': str(e)}
            else:
                results['symbolic_equations']['transfer_function'] = {'error': 'Not enough nodes for transfer function'}
        except Exception as e:
            results['symbolic_equations']['transfer_function'] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        return {
            'circuit_id': circuit_id,
            'status': 'error',
            'error': str(e),
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist,
            'mna_analysis': mna_info,
            'predicted_analysis_mode': analysis_mode_info
        }

def analyze_synthetic_dataset_robust(labels_file: str, output_file: str = None, max_circuits: int = None, 
                                   include_controlled_sources: bool = True) -> Dict:
    """
    Analyze synthetic circuit dataset with robust error handling.
    
    Args:
        labels_file: Path to labels.json file
        output_file: Output file path (optional)
        max_circuits: Maximum number of circuits to analyze (optional)
        include_controlled_sources: Whether to include controlled sources in analysis
        
    Returns:
        Dictionary containing analysis results
    """
    
    results = {
        'analysis_settings': {
            'labels_file': labels_file,
            'max_circuits': max_circuits,
            'include_controlled_sources': include_controlled_sources,
            'analysis_method': 'with_controlled_sources' if include_controlled_sources else 'basic_only'
        },
        'summary': {
            'total_circuits': 0,
            'successful': 0,
            'errors': 0,
            'timeouts': 0,
            'skipped': 0,
            'controlled_sources_found': 0,
            'basic_only_circuits': 0,
            'mna_matrix_stats': {
                'avg_matrix_size': 0,
                'max_matrix_size': 0,
                'high_complexity_circuits': 0,
                'reactive_circuits': 0
            }
        },
        'circuit_results': [],
        'error_types': {},
        'controlled_source_stats': {
            'vcvs_count': 0,  # E - Voltage Controlled Voltage Source
            'vccs_count': 0,  # G - Voltage Controlled Current Source  
            'cccs_count': 0,  # F - Current Controlled Current Source
            'ccvs_count': 0   # H - Current Controlled Voltage Source
        }
    }
    
    # Load circuit data
    with open(labels_file, 'r') as f:
        circuit_data = json.load(f)
    
    circuits_to_analyze = list(circuit_data.items())
    if max_circuits:
        circuits_to_analyze = circuits_to_analyze[:max_circuits]
    
    results['summary']['total_circuits'] = len(circuits_to_analyze)
    
    print(f"🔬 Starting robust analysis of {len(circuits_to_analyze)} circuits...")
    print(f"📊 Analysis mode: {'With controlled sources' if include_controlled_sources else 'Basic components only'}")
    
    for circuit_id, netlist in tqdm(circuits_to_analyze, desc="Analyzing circuits"):
        
        # Count controlled sources in original netlist
        controlled_sources = {
            'E': netlist.count('\nE') + (1 if netlist.startswith('E') else 0),
            'F': netlist.count('\nF') + (1 if netlist.startswith('F') else 0), 
            'G': netlist.count('\nG') + (1 if netlist.startswith('G') else 0),
            'H': netlist.count('\nH') + (1 if netlist.startswith('H') else 0)
        }
        
        total_controlled = sum(controlled_sources.values())
        if total_controlled > 0:
            results['summary']['controlled_sources_found'] += 1
            results['controlled_source_stats']['vcvs_count'] += controlled_sources['E']
            results['controlled_source_stats']['vccs_count'] += controlled_sources['G'] 
            results['controlled_source_stats']['cccs_count'] += controlled_sources['F']
            results['controlled_source_stats']['ccvs_count'] += controlled_sources['H']
        else:
            results['summary']['basic_only_circuits'] += 1
        
        # Analyze circuit
        circuit_result = analyze_circuit_with_timeout(
            netlist, 
            circuit_id, 
            timeout_seconds=30,
            include_controlled_sources=include_controlled_sources
        )
        
        # Add controlled source info to result
        circuit_result['controlled_sources'] = controlled_sources
        circuit_result['has_controlled_sources'] = total_controlled > 0
        
        # Update summary statistics
        if circuit_result['status'] == 'success':
            results['summary']['successful'] += 1
        elif circuit_result['status'] == 'timeout':
            results['summary']['timeouts'] += 1
        elif circuit_result['status'] == 'skipped':
            results['summary']['skipped'] += 1
        else:
            results['summary']['errors'] += 1
            
            # Track error types
            error_type = circuit_result.get('error', 'unknown')
            if error_type in results['error_types']:
                results['error_types'][error_type] += 1
            else:
                results['error_types'][error_type] = 1
        
        results['circuit_results'].append(circuit_result)
    
    # Calculate success rate
    total = results['summary']['total_circuits']
    successful = results['summary']['successful']
    results['summary']['success_rate'] = successful / total if total > 0 else 0
    
    # Calculate MNA matrix statistics
    matrix_sizes = []
    complexity_scores = []
    reactive_count = 0
    high_complexity_count = 0
    
    for circuit_result in results['circuit_results']:
        if 'mna_analysis' in circuit_result:
            mna_info = circuit_result['mna_analysis']
            matrix_sizes.append(mna_info['mna_matrix_size'])
            complexity_scores.append(mna_info['complexity_score'])
            
            if mna_info['reactive_components'] > 0:
                reactive_count += 1
            if mna_info['complexity_score'] > 20:  # Threshold for high complexity
                high_complexity_count += 1
    
    if matrix_sizes:
        results['summary']['mna_matrix_stats']['avg_matrix_size'] = sum(matrix_sizes) / len(matrix_sizes)
        results['summary']['mna_matrix_stats']['max_matrix_size'] = max(matrix_sizes)
        results['summary']['mna_matrix_stats']['reactive_circuits'] = reactive_count
        results['summary']['mna_matrix_stats']['high_complexity_circuits'] = high_complexity_count
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_file}")
    
    return results

def demonstrate_successful_analysis_robust(results: Dict):
    """
    Demonstrate successful analysis with detailed output.
    
    Args:
        results: Results dictionary from analysis
    """
    
    print(f"\n🎯 ANALYSIS RESULTS SUMMARY")
    print("="*50)
    
    summary = results['summary']
    print(f"Total circuits: {summary['total_circuits']}")
    print(f"Successful: {summary['successful']} ({summary['success_rate']:.1%})")
    print(f"Errors: {summary['errors']}")
    print(f"Timeouts: {summary['timeouts']}")
    print(f"Skipped (too complex): {summary['skipped']}")
    
    if summary['successful'] > 0:
        print(f"\n✅ SUCCESSFUL ANALYSIS EXAMPLES")
        print("-" * 40)
        
        # Show first successful analysis
        successful_results = [r for r in results['circuit_results'] if r['status'] == 'success']
        
        for i, result in enumerate(successful_results[:2]):  # Show first 2 successful
            print(f"\n📋 Circuit {i+1}: {result['circuit_id']}")
            
            if 'circuit_properties' in result:
                props = result['circuit_properties']
                print(f"  Nodes: {props.get('num_nodes', 'unknown')}")
                print(f"  Elements: {props.get('num_elements', 'unknown')}")
                print(f"  MNA Matrix Size: {props.get('mna_matrix_size', 'unknown')}x{props.get('mna_matrix_size', 'unknown')}")
                print(f"  Complexity Score: {props.get('complexity_score', 'unknown'):.1f}")
                
                if 'element_categories' in props:
                    cats = props['element_categories']
                    print(f"  Basic components: {len(cats.get('basic', []))}")
                    print(f"  Independent sources: {len(cats.get('independent_sources', []))}")
                    print(f"  Controlled sources: {len(cats.get('controlled_sources', []))}")
            
            # Show MNA analysis if available
            if 'mna_analysis' in result:
                mna = result['mna_analysis']
                print(f"  🔢 MNA Analysis:")
                print(f"    Voltage sources: {mna.get('voltage_sources', 0)}")
                print(f"    Inductors: {mna.get('inductors', 0)}")
                print(f"    Reactive components: {mna.get('reactive_components', 0)}")
            
            # Show predicted analysis mode
            if 'predicted_analysis_mode' in result:
                mode = result['predicted_analysis_mode']
                print(f"  🎯 Predicted Analysis: {mode.get('complexity_level', 'unknown')}")
                if 'likely_analysis_modes' in mode:
                    print(f"    Modes: {', '.join(mode['likely_analysis_modes'])}")
            
            # Show symbolic equations if available
            if 'symbolic_equations' in result:
                eqs = result['symbolic_equations']
                
                # Show symbolic nodal equations
                if 'nodal_equations' in eqs and 'equations' in eqs['nodal_equations']:
                    nodal = eqs['nodal_equations']
                    print(f"  📐 Symbolic nodal equations: {len(nodal['equations'])} nodes")
                    for node, eq_info in list(nodal['equations'].items())[:2]:  # Show first 2
                        print(f"    {node}: {eq_info['equation']}")
                    if len(nodal['equations']) > 2:
                        print(f"    ... and {len(nodal['equations'])-2} more")
                
                # Show symbolic node voltages
                if 'node_voltages' in eqs and 'voltages' in eqs['node_voltages']:
                    node_v = eqs['node_voltages']
                    print(f"  ⚡ Symbolic node voltages: {node_v['count']} expressions")
                    for node, info in list(node_v['voltages'].items())[:2]:  # Show first 2
                        if 'error' not in info:
                            print(f"    {info['equation']}")
                
                # Show symbolic component equations
                if 'component_equations' in eqs and 'components' in eqs['component_equations']:
                    comp = eqs['component_equations']
                    print(f"  🔧 Symbolic component equations: {comp['count']} components")
                    for name, info in list(comp['components'].items())[:2]:  # Show first 2
                        if 'error' not in info and 'expressions' in info:
                            print(f"    {info['expressions']['voltage_equation']}")
                            print(f"    {info['expressions']['current_equation']}")
                
                # Show transfer function
                if 'transfer_function' in eqs and 'function' in eqs['transfer_function']:
                    tf = eqs['transfer_function']['function']
                    print(f"  📈 Symbolic transfer function: {tf['equation']}")

    # Show error summary if there are errors
    if 'error_types' in results and results['error_types']:
        print(f"\n❌ ERROR SUMMARY")
        print("-" * 40)
        for error_type, count in results['error_types'].items():
            print(f"  {error_type}: {count} circuits")

def main():

    parser = argparse.ArgumentParser(description='Symbolic Equation Extraction for Circuit Analysis (MNA-compatible)')
    parser.add_argument('--max-circuits', '--max_circuits', type=int, default=None, help='Maximum number of circuits to analyze (default: all)')
    parser.add_argument('--labels-file', '--labels_file', default="datasets/grid_v11_240831/labels.json", help='Path to labels.json file')
    parser.add_argument('--output-file', '--output_file', type=str, default=None, help='Output file path (optional)')
    parser.add_argument('--show-samples', '--show_samples', action='store_true', help='Display sample equations in terminal')
    args = parser.parse_args()
    
    # Path to dataset
    labels_file = args.labels_file
    max_circuits = args.max_circuits
    output_file = args.output_file
    show_samples = args.show_samples
    
    if not Path(labels_file).exists():
        print(f"Dataset file not found: {labels_file}")
        return

    if max_circuits:
        print(f"Analyzing first {max_circuits} circuits")
    else:
        print(f"Analyzing ALL circuits in dataset")
    
    # When called from main.py, focus on single analysis with controlled sources
    if output_file:
        print("\n🔬 SYMBOLIC EQUATION EXTRACTION")
        print("-" * 60)
        print("Extracting symbolic mathematical expressions suitable for MLLM training:")
        print("• Nodal equations (KCL): (V1-V2)/R1 + V1*s*C1 = I_source")
        print("• Node voltages: V_1(s) = I_s*R1/(R1*C1*s + 1)")
        print("• Component V-I: I_R1(s) = (V1-V2)/R1")
        print("• Transfer functions: H(s) = Vout(s)/Vin(s)")
        
        results = analyze_synthetic_dataset_robust(
            labels_file=labels_file,
            output_file=output_file,
            max_circuits=max_circuits,
            include_controlled_sources=True
        )
        
        if show_samples:
            demonstrate_successful_analysis_robust(results)
        
        return
    
    # When called standalone, run both analyses for comparison
    print("\n🔬 Analysis #1: SYMBOLIC EQUATIONS - Including Controlled Sources")
    print("-" * 60)
    print("Extracting symbolic mathematical expressions suitable for MLLM training:")
    print("• Nodal equations (KCL): (V1-V2)/R1 + V1*s*C1 = I_source")
    print("• Node voltages: V_1(s) = I_s*R1/(R1*C1*s + 1)")
    print("• Component V-I: I_R1(s) = (V1-V2)/R1")
    print("• Transfer functions: H(s) = Vout(s)/Vin(s)")
    
    results_with_controlled = analyze_synthetic_dataset_robust(
        labels_file=labels_file,
        output_file="symbolic_equations_with_controlled_sources.json",
        max_circuits=max_circuits,  # Use command line arg or None for all
        include_controlled_sources=True
    )
    
    # Run symbolic equation extraction without controlled sources (basic only)
    print("\n🔬 Analysis #2: SYMBOLIC EQUATIONS - Basic Components Only")
    print("-" * 60)
    print("Extracting symbolic equations from circuits with basic components only:")
    print("• Focus on R, L, C, independent V and I sources")
    print("• Cleaner symbolic expressions for foundational learning")
    
    results_basic_only = analyze_synthetic_dataset_robust(
        labels_file=labels_file,
        output_file="symbolic_equations_basic_components.json", 
        max_circuits=max_circuits,  # Use command line arg or None for all
        include_controlled_sources=False
    )
    
    # Compare results
    print("\n📊 COMPARISON SUMMARY")
    print("="*50)
    
    print(f"{'Metric':<30} {'With Controlled':<15} {'Basic Only':<15}")
    print("-" * 60)
    print(f"{'Total Circuits':<30} {results_with_controlled['summary']['total_circuits']:<15} {results_basic_only['summary']['total_circuits']:<15}")
    print(f"{'Successful':<30} {results_with_controlled['summary']['successful']:<15} {results_basic_only['summary']['successful']:<15}")
    success_rate_with = f"{results_with_controlled['summary']['success_rate']:.1%}"
    success_rate_basic = f"{results_basic_only['summary']['success_rate']:.1%}"
    print(f"{'Success Rate':<30} {success_rate_with:<15} {success_rate_basic:<15}")
    
    # Show controlled source statistics
    cs_stats = results_with_controlled['controlled_source_stats']
    print(f"\n🎛️ CONTROLLED SOURCE STATISTICS")
    print("-" * 40)
    print(f"VCVS (E): {cs_stats['vcvs_count']}")
    print(f"VCCS (G): {cs_stats['vccs_count']}")
    print(f"CCCS (F): {cs_stats['cccs_count']}")
    print(f"CCVS (H): {cs_stats['ccvs_count']}")
    print(f"Circuits with controlled sources: {results_with_controlled['summary']['controlled_sources_found']}")
    print(f"Basic-only circuits: {results_with_controlled['summary']['basic_only_circuits']}")
    
    # Show MNA matrix statistics
    mna_stats_with = results_with_controlled['summary']['mna_matrix_stats']
    mna_stats_basic = results_basic_only['summary']['mna_matrix_stats']
    print(f"\n🔢 MNA MATRIX STATISTICS")
    print("-" * 40)
    print(f"{'Metric':<30} {'With Controlled':<15} {'Basic Only':<15}")
    print("-" * 60)
    avg_size_with = f"{mna_stats_with['avg_matrix_size']:.1f}"
    avg_size_basic = f"{mna_stats_basic['avg_matrix_size']:.1f}"
    print(f"{'Average Matrix Size':<30} {avg_size_with:<15} {avg_size_basic:<15}")
    print(f"{'Max Matrix Size':<30} {mna_stats_with['max_matrix_size']:<15} {mna_stats_basic['max_matrix_size']:<15}")
    print(f"{'Reactive Circuits':<30} {mna_stats_with['reactive_circuits']:<15} {mna_stats_basic['reactive_circuits']:<15}")
    print(f"{'High Complexity':<30} {mna_stats_with['high_complexity_circuits']:<15} {mna_stats_basic['high_complexity_circuits']:<15}")
    
    # Demonstrate differences between controlled vs basic analysis
    print(f"\n🔬 ANALYSIS METHOD COMPARISON")
    print("-" * 50)
    
    # Find a circuit with controlled sources
    for result_with, result_basic in zip(results_with_controlled['circuit_results'], 
                                       results_basic_only['circuit_results']):
        if (result_with['has_controlled_sources'] and 
            result_with['status'] == 'success' and 
            result_basic['status'] == 'success'):
            
            print(f"\nCircuit: {result_with['circuit_id']}")
            print(f"Controlled sources: {result_with['controlled_sources']}")
            
            print(f"\n📜 Original SPICE netlist:")
            original_lines = [l for l in result_with['original_netlist'].split('\n') 
                            if l.strip() and not l.startswith('.') and not l.startswith('print')][:5]
            for line in original_lines:
                print(f"  {line}")
            if len(original_lines) >= 5:
                print("  ...")
            
            print(f"\n🔧 With controlled sources (cleaned):")
            cleaned_with = result_with['cleaned_netlist'].split('\n')[:5]
            for line in cleaned_with:
                if line.strip():
                    print(f"  {line}")
            if len(cleaned_with) >= 5:
                print("  ...")
            
            print(f"\n🔧 Basic components only (cleaned):")
            cleaned_basic = result_basic['cleaned_netlist'].split('\n')[:5]
            for line in cleaned_basic:
                if line.strip():
                    print(f"  {line}")
            if len(cleaned_basic) >= 5:
                print("  ...")
            
            break
    
    print(f"\n✅ Symbolic Equation Extraction Complete!")
    print("="*60)
    print("🎯 SYMBOLIC EQUATIONS EXTRACTED FOR MLLM TRAINING:")
    print("• Nodal equations: Mathematical KCL expressions")
    print("• Node voltages: V_n(s) symbolic expressions")  
    print("• Component V-I: Voltage-current relationships")
    print("• Transfer functions: H(s) frequency domain expressions")
    print()
    print("📁 Results saved to:")
    print(f"  • symbolic_equations_with_controlled_sources.json")
    print(f"  • symbolic_equations_basic_components.json")
    print()
    print("🧮 These symbolic mathematical expressions are perfect for training")
    print("   multimodal large language models on circuit reasoning tasks!")

if __name__ == "__main__":
    main() 