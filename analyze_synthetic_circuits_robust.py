#!/usr/bin/env python3
"""
Robust Symbolic Analysis of Synthetic Circuit Dataset

This script applies lcapy symbolic analysis to synthetic circuits with 
improved error handling and timeout protection.

Author: Assistant
Date: 2024
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
    Analyze circuit with timeout protection.
    
    Args:
        spice_netlist: SPICE netlist string
        circuit_id: Circuit identifier
        timeout_seconds: Timeout in seconds
        include_controlled_sources: Whether to include controlled sources in analysis
        
    Returns:
        Dictionary containing analysis results
    """
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        if include_controlled_sources:
            result = extract_symbolic_equations_with_controlled_sources(spice_netlist, circuit_id)
        else:
            result = extract_symbolic_equations_simple(spice_netlist, circuit_id)
            
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutException:
        signal.alarm(0)  # Cancel the alarm
        return {
            'circuit_id': circuit_id,
            'status': 'timeout',
            'error': f'Analysis timed out after {timeout_seconds} seconds',
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

def extract_symbolic_equations_with_controlled_sources(spice_netlist: str, circuit_id: str) -> Dict:
    """
    Extract symbolic equations from a circuit including controlled sources.
    
    Args:
        spice_netlist: SPICE netlist string
        circuit_id: Circuit identifier
        
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
            'cleaned_netlist': cleaned_netlist
        }
    
    # Check if circuit has enough components
    lines = [l for l in cleaned_netlist.split('\n') if l.strip()]
    if len(lines) < 2:
        return {
            'circuit_id': circuit_id,
            'status': 'error', 
            'error': f'Circuit too simple - only {len(lines)} components',
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist
        }
    
    try:
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
        
        # 1. TRY NODAL ANALYSIS (handles controlled sources better)
        try:
            # For circuits with controlled sources, try modified nodal analysis
            if len(elements['controlled_sources']) > 0:
                results['analysis_method'] = 'modified_nodal_analysis'
                
                # Try to get the circuit equations in matrix form
                try:
                    # Get node voltages symbolically
                    node_voltages = {}
                    for node in cct.node_list:
                        if node != '0':  # Skip ground node
                            try:
                                v_node = cct[node].V
                                node_voltages[f'V_{node}'] = str(v_node)
                            except:
                                node_voltages[f'V_{node}'] = f'V_{node}'
                    
                    results['symbolic_equations']['node_voltages'] = node_voltages
                    
                except Exception as e:
                    results['symbolic_equations']['node_analysis_error'] = str(e)
            
            else:
                # Standard nodal analysis for circuits without controlled sources
                na = cct.nodal_analysis()
                nodal_eqs = na.nodal_equations()
                
                results['symbolic_equations']['nodal'] = {
                    'equations': {str(k): str(v) for k, v in nodal_eqs.items()},
                    'num_equations': len(nodal_eqs),
                    'description': 'Nodal analysis equations (KCL at each node)'
                }
                
        except Exception as e:
            results['symbolic_equations']['nodal_error'] = str(e)
        
        # 2. TRY TRANSFER FUNCTIONS (if possible)
        try:
            if len(elements['independent_sources']) > 0 and len(elements['controlled_sources']) > 0:
                # Try to compute transfer functions for controlled source circuits
                source_name = elements['independent_sources'][0]
                
                # Try to get transfer function from first source to first node
                if len(cct.node_list) > 1:
                    target_node = [n for n in cct.node_list if n != '0'][0]
                    
                    try:
                        tf = cct[target_node].V / cct[source_name].V
                        results['symbolic_equations']['transfer_function'] = {
                            'expression': str(tf),
                            'from': source_name,
                            'to': f'V_{target_node}',
                            'description': f'Transfer function from {source_name} to node {target_node}'
                        }
                    except:
                        pass
                        
        except Exception as e:
            results['symbolic_equations']['transfer_function_error'] = str(e)
        
        # 3. COMPONENT ANALYSIS
        try:
            component_info = {}
            for element_name, element in cct.elements.items():
                try:
                    component_info[element_name] = {
                        'type': type(element).__name__,
                        'nodes': str(element.nodes),
                    }
                    
                    # Try to get voltage and current
                    try:
                        component_info[element_name]['voltage'] = str(element.V)
                    except:
                        pass
                    try:
                        component_info[element_name]['current'] = str(element.I)
                    except:
                        pass
                        
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
            'cleaned_netlist': cleaned_netlist
        }

def extract_symbolic_equations_simple(spice_netlist: str, circuit_id: str) -> Dict:
    """
    Extract symbolic equations from a circuit - simplified version without controlled sources.
    
    Args:
        spice_netlist: SPICE netlist string
        circuit_id: Circuit identifier
        
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
            'cleaned_netlist': cleaned_netlist
        }
    
    # Check if circuit has enough components
    lines = [l for l in cleaned_netlist.split('\n') if l.strip()]
    if len(lines) < 2:
        return {
            'circuit_id': circuit_id,
            'status': 'error', 
            'error': f'Circuit too simple - only {len(lines)} basic components',
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist
        }
    
    try:
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
        
        # Try to get basic circuit properties first
        try:
            results['circuit_properties']['is_dc'] = cct.is_dc
        except:
            results['circuit_properties']['is_dc'] = 'unknown'
        
        # 1. TRY NODAL ANALYSIS (most basic)
        try:
            na = cct.nodal_analysis()
            nodal_eqs = na.nodal_equations()
            
            results['symbolic_equations']['nodal'] = {
                'equations': {str(k): str(v) for k, v in nodal_eqs.items()},
                'num_equations': len(nodal_eqs),
                'description': 'Nodal analysis equations (KCL at each node)'
            }
            
            # Try to get matrix form if possible
            try:
                results['symbolic_equations']['nodal']['A_matrix'] = str(na.A)
                results['symbolic_equations']['nodal']['b_vector'] = str(na.b)
            except:
                pass
                
        except Exception as e:
            results['symbolic_equations']['nodal'] = {'error': str(e)}
        
        # 2. TRY SIMPLE NODE VOLTAGES
        try:
            node_voltages = {}
            nodes = [n for n in cct.node_list if n != '0'][:3]  # Limit to first 3 nodes
            
            for node_name in nodes:
                try:
                    V_node = cct[node_name].V(lcapy.s)
                    node_voltages[f'V_{node_name}'] = str(V_node)
                except Exception as e:
                    node_voltages[f'V_{node_name}'] = f'Error: {str(e)}'
            
            results['symbolic_equations']['node_voltages'] = node_voltages
                     
        except Exception as e:
            results['symbolic_equations']['node_voltages'] = {'error': str(e)}
        
        # 3. TRY SIMPLE COMPONENT ANALYSIS
        try:
            component_eqs = {}
            # Limit to first 3 components to avoid timeout
            for i, (name, component) in enumerate(list(cct.elements.items())[:3]):
                try:
                    V_s = component.V(lcapy.s)
                    component_eqs[name] = {
                        'voltage_s': str(V_s),
                        'type': str(type(component).__name__)
                    }
                except Exception as e:
                    component_eqs[name] = {'error': str(e)}
            
            results['symbolic_equations']['components'] = component_eqs
             
        except Exception as e:
            results['symbolic_equations']['components'] = {'error': str(e)}
        
        # 4. TRY ONE SIMPLE TRANSFER FUNCTION
        try:
            nodes = [n for n in cct.node_list if n != '0']
            if len(nodes) >= 2:
                input_node = nodes[0]
                output_node = nodes[1]
                try:
                    H_s = cct.transfer(input_node, 0, output_node, 0)
                    results['symbolic_equations']['transfer_function'] = {
                        f'H_{input_node}_to_{output_node}': str(H_s)
                    }
                except Exception as e:
                    results['symbolic_equations']['transfer_function'] = {'error': str(e)}
        except Exception as e:
            results['symbolic_equations']['transfer_function'] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        return {
            'circuit_id': circuit_id,
            'status': 'error',
            'error': str(e),
            'original_netlist': spice_netlist,
            'cleaned_netlist': cleaned_netlist
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
            'controlled_sources_found': 0,
            'basic_only_circuits': 0
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
                
                if 'element_categories' in props:
                    cats = props['element_categories']
                    print(f"  Basic components: {len(cats.get('basic', []))}")
                    print(f"  Independent sources: {len(cats.get('independent_sources', []))}")
                    print(f"  Controlled sources: {len(cats.get('controlled_sources', []))}")
            
            # Show symbolic equations if available
            if 'symbolic_equations' in result:
                eqs = result['symbolic_equations']
                
                if 'nodal' in eqs and 'equations' in eqs['nodal']:
                    nodal = eqs['nodal']
                    print(f"  📐 Nodal equations: {len(nodal['equations'])} nodes")
                
                if 'node_voltages' in eqs:
                    node_v = eqs['node_voltages']
                    if isinstance(node_v, dict):
                        print(f"  ⚡ Node voltages: {len(node_v)} computed")
                
                if 'components' in eqs:
                    comp = eqs['components']
                    if isinstance(comp, dict) and 'error' not in comp:
                        print(f"  🔧 Component analysis: {len(comp)} components")
                        for name, info in list(comp.items())[:3]:  # Show first 3
                            if 'voltage' in info:
                                print(f"    {name}: V = {info['voltage']}")

    # Show error summary if there are errors
    if 'error_types' in results and results['error_types']:
        print(f"\n❌ ERROR SUMMARY")
        print("-" * 40)
        for error_type, count in results['error_types'].items():
            print(f"  {error_type}: {count} circuits")

def main():

    parser = argparse.ArgumentParser(description='Robust Synthetic Circuit Analysis')
    parser.add_argument('--max-circuits', type=int, default=None, help='Maximum number of circuits to analyze (default: all)')
    parser.add_argument('--labels-file', default="datasets/grid_v11_240831/labels.json", help='Path to labels.json file')
    args = parser.parse_args()
    
    # Path to dataset
    labels_file = args.labels_file
    max_circuits = args.max_circuits
    
    if not Path(labels_file).exists():
        print(f"Dataset file not found: {labels_file}")
        return
    
    print("ROBUST SYNTHETIC CIRCUIT ANALYSIS")
    print("="*50)
    if max_circuits:
        print(f"Analyzing first {max_circuits} circuits")
    else:
        print(f"Analyzing ALL circuits in dataset")
    
    # Run analysis with controlled sources
    print("\n🔬 Analysis #1: INCLUDING Controlled Sources")
    print("-" * 40)
    
    results_with_controlled = analyze_synthetic_dataset_robust(
        labels_file=labels_file,
        output_file="synthetic_circuits_robust_analysis_with_controlled.json",
        max_circuits=max_circuits,  # Use command line arg or None for all
        include_controlled_sources=True
    )
    
    # Run analysis without controlled sources (basic only)
    print("\n🔬 Analysis #2: BASIC Components Only (No Controlled Sources)")
    print("-" * 40)
    
    results_basic_only = analyze_synthetic_dataset_robust(
        labels_file=labels_file,
        output_file="synthetic_circuits_robust_analysis_basic_only.json", 
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
    
    print(f"\n✅ Analysis complete! Check the generated JSON files for detailed results.")

if __name__ == "__main__":
    main() 