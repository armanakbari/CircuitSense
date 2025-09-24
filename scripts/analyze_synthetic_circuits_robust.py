import json
import argparse
from pathlib import Path
from lcapy import Circuit, s, t
from lcapy import mna
import re
import multiprocessing
import time
from functools import partial
import sympy as sp


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convert_netlist_remove_n_nodes import convert_netlist_remove_n_nodes

def _multiprocessing_target(queue, func_data):
    """Top-level target function for multiprocessing (must be pickleable)"""
    try:
        func, args = func_data
        result = func(*args)
        queue.put(('success', result))
    except Exception as e:
        queue.put(('error', str(e)))

def run_with_timeout(func, args, timeout_seconds):
    """Run function with multiprocessing timeout - can kill stuck processes"""
    queue = multiprocessing.Queue()
    func_data = (func, args)
    process = multiprocessing.Process(target=_multiprocessing_target, args=(queue, func_data))
    process.start()
    process.join(timeout=timeout_seconds)
    
    if process.is_alive():
        # Force kill the stuck process
        process.terminate()
        process.join()
        return None, f"Timeout after {timeout_seconds}s"
    
    if queue.empty():
        return None, "Process ended without result"
    
    result_type, result = queue.get()
    if result_type == 'success':
        return result, None
    else:
        return None, result

def safe_computation_mp(func, args, timeout_seconds=30, description="computation"):
    """Safely execute a function with multiprocessing timeout"""
    print(f"🔧 Starting {description} (timeout: {timeout_seconds}s)...")
    start_time = time.time()
    
    result, error = run_with_timeout(func, args, timeout_seconds)
    
    elapsed = time.time() - start_time
    if error:
        if "Timeout" in error:
            print(f"⏰ {description} timed out after {timeout_seconds}s")
        else:
            print(f"❌ {description} failed: {error}")
        return None
    else:
        print(f"✅ {description} completed in {elapsed:.1f}s")
        return result

def limit_ad_to_infinity_str(expr_str):
    """If the string expression contains 'Ad', take symbolic limit as Ad -> oo.

    Returns the simplified string if successful; otherwise returns the original string.
    """
    try:
        if expr_str is None or ('Ad' not in str(expr_str)):
            return expr_str

        # Define symbols for parser; unknown names will become Symbols automatically
        Ad = sp.symbols('Ad', positive=True)
        s_sym = sp.symbols('s')  # Laplace variable

        # Sympify; provide 's' mapping to avoid confusion with lcapy.s object
        expr = sp.sympify(str(expr_str), locals={'s': s_sym, 'Ad': Ad})
        limited = sp.limit(expr, Ad, sp.oo)
        simplified = sp.simplify(limited)
        return str(simplified)
    except Exception:
        # Fall back to original if parsing/limit fails
        return expr_str

def _compute_transfer_function(circuit, vs_nodes, comp):
    """Top-level function for transfer function computation (must be pickleable)"""
    tf = str(circuit.transfer(vs_nodes, comp))
    return limit_ad_to_infinity_str(tf)

def _compute_mna_analysis(circuit, domain='t'):
    """Top-level function for MNA computation (must be pickleable)"""
    try:
        # Use laplace domain for s-domain analysis, time domain for t-domain
        print(f"Creating {domain}-domain circuit for MNA...")
        if domain == 's':
            circuit_domain = circuit.laplace()
        else:
            circuit_domain = circuit
            
        print(f"Creating MNA object...")
        # Create MNA object with scipy solver (as shown in working example)
        try:
            mna_obj = mna.MNA(circuit_domain, solver_method='scipy')
        except Exception as mna_creation_error:
            # Try alternative solver methods if scipy fails
            print(f"scipy solver failed, trying alternative methods...")
            try:
                mna_obj = mna.MNA(circuit_domain, solver_method='numpy')
            except Exception:
                try:
                    mna_obj = mna.MNA(circuit_domain, solver_method='sympy')
                except Exception as final_error:
                    return f"MNA Creation Error: Failed with all solver methods. Original: {str(mna_creation_error)}, Final: {str(final_error)}"
        
        print(f"Getting matrix equations...")
        # Get matrix equations
        matrix_eqs = mna_obj.matrix_equations()
        
        if matrix_eqs is None:
            return f"MNA Error: matrix_equations() returned None"
            
        print(f"Converting to readable form...")
        
        # First, let's try to just return the basic matrix representation to debug
        # This should work similar to the user's example: print(na.matrix_equations())
        try:
            basic_repr = str(matrix_eqs)
        except Exception as basic_error:
            basic_repr = f"<ERROR_GETTING_BASIC_REPR: {str(basic_error)}>"

        return limit_ad_to_infinity_str(basic_repr)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"MNA Error: {str(e)}\nDetails:\n{error_details}"

def debug_mna_object(netlist_string):
    """Debug function to inspect MNA object structure"""
    try:
        print("=== DEBUG MNA OBJECT ===")
        circuit = Circuit(netlist_string)
        print(f"Circuit created successfully")
        
        circuit_s = circuit.laplace()
        print(f"Laplace transform applied")
        
        mna_obj = mna.MNA(circuit_s, solver_method='scipy')
        print(f"MNA object created with scipy solver")
        
        matrix_eqs = mna_obj.matrix_equations()
        print(f"Matrix equations obtained: {type(matrix_eqs)}")
        
        print(f"\nMNA object attributes:")
        for attr in sorted(dir(mna_obj)):
            if not attr.startswith('_'):
                try:
                    value = getattr(mna_obj, attr)
                    print(f"  {attr}: {type(value)} = {str(value)[:100]}...")
                except:
                    print(f"  {attr}: <access_error>")
        
        print(f"\nMatrix equations attributes:")
        for attr in sorted(dir(matrix_eqs)):
            if not attr.startswith('_'):
                try:
                    value = getattr(matrix_eqs, attr)
                    print(f"  {attr}: {type(value)}")
                except:
                    print(f"  {attr}: <access_error>")
        
        print(f"\nBasic matrix representation:")
        print(str(matrix_eqs))
        
        return matrix_eqs, mna_obj
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def _convert_matrix_to_readable(matrix_eqs, mna_obj):
    """Convert matrix equations to human-readable form"""
    try:
        # Try to get unknowns from different possible attributes/methods
        unknowns = None
        
        # Try various ways to get the unknowns
        if hasattr(mna_obj, 'unknowns'):
            unknowns = mna_obj.unknowns
        elif hasattr(mna_obj, 'x'):
            unknowns = mna_obj.x
        elif hasattr(mna_obj, 'variables'):
            unknowns = mna_obj.variables
        elif hasattr(mna_obj, '_unknowns'):
            unknowns = mna_obj._unknowns
        else:
            # Try to extract from matrix_equations itself
            if hasattr(matrix_eqs, 'lhs') and hasattr(matrix_eqs.lhs, 'free_symbols'):
                unknowns = list(matrix_eqs.lhs.free_symbols)
            elif hasattr(matrix_eqs, 'args') and len(matrix_eqs.args) >= 2:
                # matrix_eqs might be in form Eq(unknowns_vector, solution)
                unknowns_vec = matrix_eqs.args[0]
                if hasattr(unknowns_vec, '__iter__'):
                    unknowns = list(unknowns_vec)
        
        # If we still don't have unknowns, create generic ones
        if unknowns is None:
            # Get dimensions from the matrix
            try:
                n_vars = matrix_eqs.A.cols if hasattr(matrix_eqs, 'A') else len(str(matrix_eqs).split(','))
                unknowns = [f"x{i}" for i in range(n_vars)]
            except:
                return f"Could not determine unknowns and matrix structure.\nMatrix form:\n{str(matrix_eqs)}"
        
        # Get A matrix and b vector from Ax = b form
        if hasattr(matrix_eqs, 'A') and hasattr(matrix_eqs, 'b'):
            A_matrix = matrix_eqs.A
            b_vector = matrix_eqs.b
        else:
            # Try to parse the equation structure
            return f"Matrix structure not in expected A*x = b format.\nMatrix form:\n{str(matrix_eqs)}"
        
        equations = []
        
        # For each row, create a human-readable equation
        for i in range(A_matrix.rows):
            lhs_terms = []
            
            # Build left-hand side terms
            for j in range(min(A_matrix.cols, len(unknowns))):
                try:
                    coeff = A_matrix[i, j]
                    
                    # Handle different coefficient types
                    if hasattr(coeff, 'is_zero') and coeff.is_zero:
                        continue
                    if str(coeff) == '0' or coeff == 0:
                        continue
                        
                    unknown = str(unknowns[j])
                    coeff_str = str(coeff)
                    
                    # Simplify coefficient representation
                    if coeff_str == '1':
                        lhs_terms.append(f"{unknown}")
                    elif coeff_str == '-1':
                        lhs_terms.append(f"-{unknown}")
                    else:
                        # Wrap complex coefficients in parentheses
                        if any(op in coeff_str for op in ['+', '-', '*', '/', '^', 's']):
                            lhs_terms.append(f"({coeff_str})*{unknown}")
                        else:
                            lhs_terms.append(f"{coeff_str}*{unknown}")
                            
                except Exception as coeff_error:
                    # Skip problematic coefficients
                    print(f"Warning: Skipping coefficient at [{i},{j}]: {coeff_error}")
                    continue
            
            # Build right-hand side
            try:
                rhs = str(b_vector[i])
                # Simplify RHS representation
                if rhs == '0':
                    rhs = '0'
            except Exception as rhs_error:
                rhs = f"<RHS_ERROR: {rhs_error}>"
            
            # Combine into equation
            if lhs_terms:
                lhs = " + ".join(lhs_terms).replace("+ -", "- ")
                # Clean up any double operators
                lhs = lhs.replace("- -", "+ ").replace("+ +", "+ ")
                equation = f"{lhs} = {rhs}"
                equations.append(equation)
            elif rhs != '0':  # Include equations with only RHS
                equations.append(f"0 = {rhs}")
        
        if equations:
            return "\n".join(equations)
        else:
            return f"Matrix form (no readable equations generated):\n{str(matrix_eqs)}"
        
    except Exception as e:
        # Enhanced fallback with more information
        try:
            # Debug info about the MNA object
            mna_attrs = [attr for attr in dir(mna_obj) if not attr.startswith('_')]
            matrix_attrs = [attr for attr in dir(matrix_eqs) if not attr.startswith('_')]
            
            debug_info = f"MNA object attributes: {mna_attrs[:10]}...\n"
            debug_info += f"Matrix object attributes: {matrix_attrs[:10]}...\n"
            
            if hasattr(matrix_eqs, 'A'):
                debug_info += f"Matrix dimensions: {matrix_eqs.A.rows}x{matrix_eqs.A.cols}\n"
            
            return f"Conversion Error: {str(e)}\n{debug_info}Matrix form:\n{str(matrix_eqs)}"
        except:
            return f"Conversion Error: {str(e)}\nMatrix form:\n{str(matrix_eqs)}"

def clean_netlist_for_lcapy(spice_netlist):
    """
    Clean SPICE netlist for lcapy compatibility.
    
    This function is enhanced to work better with ordinary netlists
    (without N-nodes) that have been converted by our converter.
    """
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
        
        # Preserve dependent source symbolic gains (x_k) if present
        def preserve_symbol(value):
            if value == '<Empty>':
                return '1'
            # keep symbolic x_<n> tokens
            if re.match(r'^x_\d+$', value):
                return value
            return value
        
        # Handle dependent sources and op-amps (E, F, G, H components)
        if component.upper().startswith('E'):
            # VCVS (E): Ename Np Nm Ncp Ncm gain [ac_gain]
            if len(parts) >= 6:  # At least 6 parameters needed
                output_p, output_n = parts[1], parts[2]
                input_p, input_n = parts[3], parts[4]
                gain = parts[5]
                ac_gain = parts[6] if len(parts) > 6 else "0"
                
                # Clean node names (but they should already be clean in converted netlists)
                output_p = re.sub(r'[^\w]', '_', output_p)
                output_n = re.sub(r'[^\w]', '_', output_n)
                input_p = re.sub(r'[^\w]', '_', input_p)
                input_n = re.sub(r'[^\w]', '_', input_n)
                
                # Preserve symbolic gains
                gain = preserve_symbol(gain)
                if ac_gain == '<Empty>':
                    ac_gain = "0"
                
                lines.append(f"{component} {output_p} {output_n} {input_p} {input_n} {gain} {ac_gain}")
            else:
                continue
        elif component.upper().startswith('G'):
            # VCCS (G): Gname Np Nm Ncp Ncm [value]
            if len(parts) >= 5:  # At least 5 parameters needed
                output_p, output_n = parts[1], parts[2]
                input_p, input_n = parts[3], parts[4]
                value = parts[5] if len(parts) > 5 else "1"
                
                # Clean node names
                output_p = re.sub(r'[^\w]', '_', output_p)
                output_n = re.sub(r'[^\w]', '_', output_n)
                input_p = re.sub(r'[^\w]', '_', input_p)
                input_n = re.sub(r'[^\w]', '_', input_n)
                
                value = preserve_symbol(value)
                
                lines.append(f"{component} {output_p} {output_n} {input_p} {input_n} {value}")
            else:
                continue
        elif component.upper().startswith('F'):
            # CCCS (F): Fname Np Nm Vcontrol [value]
            if len(parts) >= 4:
                output_p, output_n = parts[1], parts[2]
                vcontrol = parts[3]
                value = parts[4] if len(parts) > 4 else "1"
                
                # Clean node names
                output_p = re.sub(r'[^\w]', '_', output_p)
                output_n = re.sub(r'[^\w]', '_', output_n)
                vcontrol = re.sub(r'[^\w]', '_', vcontrol)
                
                value = preserve_symbol(value)
                
                lines.append(f"{component} {output_p} {output_n} {vcontrol} {value}")
            else:
                continue
        elif component.upper().startswith('H'):
            # CCVS (H): Hname Np Nm Vcontrol [value]
            if len(parts) >= 4:
                output_p, output_n = parts[1], parts[2]
                vcontrol = parts[3]
                value = parts[4] if len(parts) > 4 else "1"
                
                # Clean node names
                output_p = re.sub(r'[^\w]', '_', output_p)
                output_n = re.sub(r'[^\w]', '_', output_n)
                vcontrol = re.sub(r'[^\w]', '_', vcontrol)
                
                value = preserve_symbol(value)
                
                lines.append(f"{component} {output_p} {output_n} {vcontrol} {value}")
            else:
                continue
        else:
            # Handle regular components (R, L, C, V, I)
            node1, node2 = parts[1], parts[2]
            value = parts[3]
            
            # Clean node names (should already be clean in converted netlists)
            node1 = re.sub(r'[^\w]', '_', node1)
            node2 = re.sub(r'[^\w]', '_', node2)
            
            # Skip measurement voltage sources that shouldn't exist in converted netlists
            if component.startswith('V_meas') or component.startswith('VI'):
                print(f"⚠️ Warning: Found measurement component {component} in supposedly converted netlist")
                continue
            
            # Handle empty values
            if value == '<Empty>':
                value = component
            
            # Preserve DC/AC parameter lists for voltage sources (case-insensitive)
            if component.upper().startswith('V') and value.lower() in ['dc', 'ac']:
                tail = ' '.join(parts[3:])
                lines.append(f"{component} {node1} {node2} {tail}")
            else:
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

def load_circuit_data(data_source, use_converted_netlists=True):
    """
    Load circuit data from various sources and optionally convert N-nodes.
    
    Args:
        data_source: Path to JSON file or data structure
        use_converted_netlists: If True, convert N-nodes to ordinary netlists
        
    Returns:
        Dictionary of {circuit_id: netlist}
    """
    # Load the data
    if isinstance(data_source, (str, Path)):
        data_path = Path(data_source)
        if not data_path.exists():
            raise FileNotFoundError(f"File not found: {data_path}")
            
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        data = data_source
    
    circuits = {}
    
    # Handle different JSON structures
    if 'results' in data:
        # Structure from symbolic_equations.json
        results = data['results']
        for result in results:
            circuit_id = result.get('circuit_id')
            
            if use_converted_netlists and 'cleaned_netlist' in result:
                # Use the already converted netlist
                netlist = result['cleaned_netlist']
                print(f"✅ Using converted netlist for {circuit_id}")
            elif 'original_netlist_with_measurements' in result:
                # Convert on-the-fly
                original = result['original_netlist_with_measurements']
                netlist = convert_netlist_remove_n_nodes(original)
                print(f"🔄 Converting netlist for {circuit_id}")
            elif 'cleaned_netlist' in result:
                # Fallback to cleaned netlist and try conversion
                original = result['cleaned_netlist']
                if 'N' in original and ('V_meas' in original or 'VI' in original):
                    netlist = convert_netlist_remove_n_nodes(original)
                    print(f"🔄 Converting fallback netlist for {circuit_id}")
                else:
                    netlist = original
                    print(f"✅ Using existing clean netlist for {circuit_id}")
            else:
                print(f"⚠️ No suitable netlist found for {circuit_id}")
                continue
                
            if circuit_id and netlist:
                circuits[circuit_id] = netlist
                
    elif isinstance(data, dict):
        # Structure from labels.json (circuit_id: netlist)
        for circuit_id, netlist in data.items():
            if use_converted_netlists:
                # Check if conversion is needed
                if 'N' in netlist and ('V_meas' in netlist or 'VI' in netlist):
                    netlist = convert_netlist_remove_n_nodes(netlist)
                    print(f"🔄 Converting netlist for {circuit_id}")
                else:
                    print(f"✅ Using existing clean netlist for {circuit_id}")
            circuits[circuit_id] = netlist
    else:
        raise ValueError("Unknown data format")
    
    print(f"📊 Loaded {len(circuits)} circuits (converted N-nodes: {use_converted_netlists})")
    return circuits

def analyze_circuit(netlist, circuit_id):
    try:
        # The netlist should already be converted, but clean it for lcapy
        cleaned = clean_netlist_for_lcapy(netlist)
        print(f"\nCircuit {circuit_id}:")
        print(f"Cleaned netlist:\n{cleaned}")
        
        if not cleaned:
            print("No components after cleaning")
            return {'circuit_id': circuit_id, 'error': 'No components after cleaning netlist'}
            
        # Enhanced circuit complexity checking
        lines = [line for line in cleaned.split('\n') if line.strip()]
        num_components = len(lines)
        
        # Count different component types for complexity assessment
        num_capacitors = len([line for line in lines if line.startswith('C')])
        num_inductors = len([line for line in lines if line.startswith('L')])
        num_opamps = len([line for line in lines if line.startswith('E')])
        num_nodes = len(set(node for line in lines for node in line.split()[1:3] if node != '0'))
        
        # Calculate complexity score
        complexity_score = num_components + num_capacitors * 2 + num_inductors * 2 + num_opamps * 3
        matrix_size_estimate = num_nodes
        
        print(f"Circuit complexity: {num_components} components, {num_nodes} nodes, score: {complexity_score}")
        print(f"Estimated matrix size: {matrix_size_estimate}×{matrix_size_estimate}")
        
        # More permissive thresholds - only skip extremely complex circuits
        if num_components > 20 or complexity_score > 40 or matrix_size_estimate > 12:
            print(f"Circuit too complex for symbolic analysis, skipping")
            return {
                'circuit_id': circuit_id,
                'skipped': True,
                'reason': f'High complexity (score: {complexity_score}, matrix: {matrix_size_estimate}×{matrix_size_estimate})',
                'complexity_metrics': {
                    'num_components': num_components,
                    'num_capacitors': num_capacitors,
                    'num_inductors': num_inductors,
                    'num_opamps': num_opamps,
                    'num_nodes': num_nodes,
                    'complexity_score': complexity_score
                }
            }
        
        print(f"🔧 Creating lcapy circuit...")
        try:
            circuit = Circuit(cleaned)
            print(f"Circuit created successfully")
        except Exception as e:
            print(f"Failed to create lcapy circuit: {e}")
            return {'circuit_id': circuit_id, 'error': f'Circuit creation failed: {str(e)}'}
        
        print(f"Finding voltage sources and components...")
        try:
            voltage_sources = find_voltage_sources(circuit)
            components = find_components(circuit)
            print(f"Voltage sources: {voltage_sources}")
            print(f"Components: {components}")
        except Exception as e:
            print(f"Failed to analyze circuit elements: {e}")
            return {'circuit_id': circuit_id, 'error': f'Element analysis failed: {str(e)}'}
        
        if not voltage_sources:
            print("No voltage sources found")
            return {'circuit_id': circuit_id, 'error': 'No voltage sources found'}
            
        if not components:
            print("No components found")
            return {'circuit_id': circuit_id, 'error': 'No components found'}
        
        result = {
            'circuit_id': circuit_id,
            'cleaned_netlist': cleaned,
            'voltage_sources': voltage_sources,
            'components': components,
            'transfer_functions': {},
            'nodal_equations': {},
            'complexity_metrics': {
                'num_components': num_components,
                'num_capacitors': num_capacitors,
                'num_inductors': num_inductors,
                'num_opamps': num_opamps,
                'num_nodes': num_nodes,
                'complexity_score': complexity_score
            }
        }
        
        vs_name, vs_nodes = voltage_sources[0]
        
        # More generous timeouts based on circuit complexity
        if complexity_score <= 15:
            timeout_tf = 30
            timeout_nodal = 40
        elif complexity_score <= 25:
            timeout_tf = 20
            timeout_nodal = 25
        elif complexity_score <= 35:
            timeout_tf = 15
            timeout_nodal = 20
        else:
            timeout_tf = 10
            timeout_nodal = 15
        
        print(f"Using generous timeouts: TF={timeout_tf}s, MNA={timeout_nodal}s")
        
        # Try at least one transfer function for testing
        max_transfer_functions = min(1, len(components))  # Start with just 1 to test
        for i, comp in enumerate(components[:max_transfer_functions]):
            print(f"Analyzing transfer function {i+1}/{max_transfer_functions}: {vs_name} -> {comp}")
            tf_result = safe_computation_mp(
                _compute_transfer_function,
                (circuit, vs_nodes, comp),
                timeout_seconds=timeout_tf,
                description=f"transfer function {vs_name} -> {comp}"
            )
            if tf_result is not None:
                result['transfer_functions'][f"{vs_name}_to_{comp}"] = tf_result
                print(f"Transfer function success: {tf_result[:100]}...")
            else:
                result['transfer_functions'][f"{vs_name}_to_{comp}"] = "TIMEOUT_OR_ERROR"
                print(f"Transfer function timed out or failed")
        
        # Only try MNA analysis if we got at least one transfer function
        if any(v != "TIMEOUT_OR_ERROR" for v in result['transfer_functions'].values()):
            print(f"Attempting T-domain MNA equations...")
            mna_t = safe_computation_mp(
                _compute_mna_analysis,
                (circuit, 't'),
                timeout_seconds=timeout_nodal,
                description="T-domain MNA equations"
            )
            if mna_t is not None:
                result['nodal_equations']['t_domain'] = mna_t
                print(f"T-domain MNA equations success")
            else:
                result['nodal_equations']['t_domain'] = "TIMEOUT_OR_ERROR"
                print(f"T-domain MNA equations timed out or failed")
                
            # Also try S-domain MNA analysis
            print(f"Attempting S-domain MNA equations...")
            mna_s = safe_computation_mp(
                _compute_mna_analysis,
                (circuit, 's'),
                timeout_seconds=timeout_nodal,
                description="S-domain MNA equations"
            )
            if mna_s is not None:
                result['nodal_equations']['s_domain'] = mna_s
                print(f"S-domain MNA equations success")
            else:
                result['nodal_equations']['s_domain'] = "TIMEOUT_OR_ERROR"
                print(f"S-domain MNA equations timed out or failed")
        else:
            print(f"Skipping MNA analysis (no successful transfer functions)")
            result['nodal_equations']['t_domain'] = "SKIPPED_NO_TRANSFER_FUNCTIONS"
            result['nodal_equations']['s_domain'] = "SKIPPED_NO_TRANSFER_FUNCTIONS"
        
        return result
        
    except Exception as e:
        print(f"Circuit {circuit_id} failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {'circuit_id': circuit_id, 'error': f'Exception: {str(e)}'}

def main():
    # Use default multiprocessing method (fork on Linux, spawn on Windows)
    # This avoids pickling issues with complex lcapy objects
    pass
    
    parser = argparse.ArgumentParser(description="Analyze synthetic circuits using MNA with robust timeout handling")
    parser.add_argument('--labels_file', default="datasets/equations_2/labels.json", 
                       help="Path to labels.json file or symbolic_equations.json with converted netlists")
    parser.add_argument('--output_file', default="symbolic_equations.json")
    parser.add_argument('--max_circuits', type=int, default=1000)
    parser.add_argument('--show_samples', action='store_true', help='Show sample equations during analysis')
    parser.add_argument('--max_components', type=int, default=21, help='Skip circuits with more than this many components (increased default)')
    parser.add_argument('--fast_mode', action='store_true', help='Use shorter timeouts for faster processing')
    parser.add_argument('--use_converted_netlists', action='store_true', default=True,
                       help='Use converted netlists without N-nodes (default: True)')
    parser.add_argument('--converted_file', 
                       help='Path to JSON file with converted netlists (e.g., symbolic_equations_no_n_nodes.json)')
    args = parser.parse_args()
    
    # Determine which file to use
    if args.converted_file:
        data_file = args.converted_file
        print(f"Using converted netlists from: {data_file}")
    else:
        data_file = args.labels_file
        print(f"Using data from: {data_file}")
    
    if not Path(data_file).exists():
        print(f"File not found: {data_file}")
        return
    
    # Load circuit data with optional conversion
    try:
        circuits = load_circuit_data(data_file, use_converted_netlists=args.use_converted_netlists)
    except Exception as e:
        print(f"Failed to load circuit data: {e}")
        return
    
    circuit_items = list(circuits.items())
    
    
    results = []
    successful = 0
    skipped = 0
    failed = 0
    
    print(f"Analyzing {len(circuit_items)} circuits with CONVERTED NETLISTS...")
    print(f"Using N-node converted netlists: {args.use_converted_netlists}")
    print(f"Max components limit: {args.max_components} (increased for better success rate)")
    print(f"Fast mode: {'ON' if args.fast_mode else 'OFF'}")
    print(f"Using multiprocessing timeouts for robust analysis")
    print(f"Focus: Clean netlists should improve lcapy compatibility")
    
    # Track detailed error types
    error_types = {}
    
    for i, (circuit_id, netlist) in enumerate(circuit_items, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(circuit_items)}] Processing {circuit_id}...")
        
        result = analyze_circuit(netlist, circuit_id)
        
        if result is None:
            failed += 1
            print(f"{circuit_id} - Analysis returned None")
        elif result.get('skipped', False):
            skipped += 1
            results.append(result)
            reason = result.get('reason', 'Unknown')
            print(f"{circuit_id} - Skipped: {reason}")
        elif 'error' in result:
            failed += 1
            results.append(result)
            error_msg = result['error']
            
            # Track error types for debugging
            error_key = error_msg.split(':')[0] if ':' in error_msg else error_msg[:50]
            error_types[error_key] = error_types.get(error_key, 0) + 1
            
            print(f"{circuit_id} - Error: {error_msg}")
        else:
            successful += 1
            results.append(result)
            
            # Show complexity metrics and success details
            metrics = result.get('complexity_metrics', {})
            score = metrics.get('complexity_score', 0)
            nodes = metrics.get('num_nodes', 0)
            
            # Count successful analyses
            tf_success = sum(1 for v in result.get('transfer_functions', {}).values() 
                           if v not in ["TIMEOUT_OR_ERROR", "SKIPPED_TOO_COMPLEX", "SKIPPED_NO_TRANSFER_FUNCTIONS"])
            mna_success = sum(1 for v in result.get('nodal_equations', {}).values() 
                            if v not in ["TIMEOUT_OR_ERROR", "SKIPPED_TOO_COMPLEX", "SKIPPED_NO_TRANSFER_FUNCTIONS"])
            
            print(f"{circuit_id} - Success (complexity: {score}, nodes: {nodes}, TF: {tf_success}, MNA: {mna_success})")
            
            # Show sample if requested
            if args.show_samples and 'transfer_functions' in result:
                for tf_name, tf_expr in result['transfer_functions'].items():
                    if tf_expr not in ["TIMEOUT_OR_ERROR", "SKIPPED_TOO_COMPLEX", "SKIPPED_NO_TRANSFER_FUNCTIONS"]:
                        print(f"  📈 Sample: {tf_name}: {tf_expr}")
                        break
    
    # Calculate detailed statistics
    timeout_count = sum(1 for r in results 
                       if any('TIMEOUT_OR_ERROR' in str(v) 
                             for v in r.get('transfer_functions', {}).values()) or
                          any('TIMEOUT_OR_ERROR' in str(v)
                             for v in r.get('nodal_equations', {}).values()))
    
    skipped_complex = sum(1 for r in results if r.get('skipped', False))
    
    # Complexity analysis
    if results:
        complexity_scores = [r.get('complexity_metrics', {}).get('complexity_score', 0) 
                           for r in results if 'complexity_metrics' in r]
        if complexity_scores:
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            max_complexity = max(complexity_scores)
            min_complexity = min(complexity_scores)
        else:
            avg_complexity = max_complexity = min_complexity = 0
    else:
        avg_complexity = max_complexity = min_complexity = 0
    
    # Count actual equation successes
    total_tf_success = sum(1 for r in results for v in r.get('transfer_functions', {}).values() 
                          if v not in ["TIMEOUT_OR_ERROR", "SKIPPED_TOO_COMPLEX", "SKIPPED_NO_TRANSFER_FUNCTIONS"])
    total_mna_success = sum(1 for r in results for v in r.get('nodal_equations', {}).values() 
                           if v not in ["TIMEOUT_OR_ERROR", "SKIPPED_TOO_COMPLEX", "SKIPPED_NO_TRANSFER_FUNCTIONS"])
    
    output = {
        'summary': {
            'total_circuits': len(circuit_items),
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
            'timeout_count': timeout_count,
            'skipped_complex': skipped_complex,
            'success_rate': successful / len(circuit_items) if circuit_items else 0,
            'equation_counts': {
                'transfer_functions': total_tf_success,
                'mna_equations': total_mna_success
            },
            'complexity_stats': {
                'average_complexity': avg_complexity,
                'max_complexity': max_complexity,
                'min_complexity': min_complexity
            },
            'error_breakdown': error_types,
            'analysis_settings': {
                'max_components': args.max_components,
                'fast_mode': args.fast_mode,
                'multiprocessing_timeouts': True,
                'permissive_thresholds': True,
                'used_converted_netlists': args.use_converted_netlists,
                'data_source': data_file
            }
        },
        'results': results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📊 Final Analysis Summary:")
    print(f"   Successful circuits: {successful}")
    print(f"   ransfer functions: {total_tf_success}")
    print(f"   MNA equations: {total_mna_success}")
    print(f"   Skipped (complex): {skipped}") 
    print(f"   Failed: {failed}")
    print(f"   Timeouts: {timeout_count}")
    print(f"   Circuit success rate: {successful/len(circuit_items)*100:.1f}%")
    print(f"   Complexity range: {min_complexity:.1f} - {max_complexity:.1f} (avg: {avg_complexity:.1f})")
    print(f"   Used converted netlists: {args.use_converted_netlists}")
    
    if error_types:
        print(f"\nError breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {error_type}: {count}")
    
    print(f"   💾 Results saved to {args.output_file}")

if __name__ == "__main__":
    main()