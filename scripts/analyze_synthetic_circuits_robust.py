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
                                                                            
    try:
        func, args = func_data
        result = func(*args)
        queue.put(('success', result))
    except Exception as e:
        queue.put(('error', str(e)))

# sympy under lcapy can get stuck indefinitely on hard integrals or simplifications. This will run the actual analysis function in a separate process with a timeout.
def run_with_timeout(func, args, timeout_seconds):
                                                                              
    queue = multiprocessing.Queue()
    func_data = (func, args)
    process = multiprocessing.Process(target=_multiprocessing_target, args=(queue, func_data))
    process.start()
    process.join(timeout=timeout_seconds)
    
    if process.is_alive():
                                      
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

# this function will run the actual analysis function in a separate process with a timeout.
def safe_computation_mp(func, args, timeout_seconds=30, description="computation"):
                                                                
    print(f"🔧 Starting {description} (timeout: {timeout_seconds}s)...")
    start_time = time.time()
    
    result, error = run_with_timeout(func, args, timeout_seconds)
    
    elapsed = time.time() - start_time
    if error:
        if "Timeout" in error:
            print(f"{description} timed out after {timeout_seconds}s")
        else:
            print(f"{description} failed: {error}")
        return None
    else:
        print(f"{description} completed in {elapsed:.1f}s")
        return result

def limit_ad_to_infinity_str(expr_str):
  
    try:
        if expr_str is None or ('Ad' not in str(expr_str)):
            return expr_str

                                                                                    
        Ad = sp.symbols('Ad', positive=True)
        s_sym = sp.symbols('s')                    

                                                                             
        expr = sp.sympify(str(expr_str), locals={'s': s_sym, 'Ad': Ad})
        limited = sp.limit(expr, Ad, sp.oo)
        simplified = sp.simplify(limited)
        return str(simplified)
    except Exception:
                                                      
        return expr_str

def _compute_transfer_function(circuit, vs_nodes, comp):
                                                                                   
    tf = str(circuit.transfer(vs_nodes, comp))
    return limit_ad_to_infinity_str(tf)

def _compute_mna_analysis(circuit, domain='t'):                                                           
    try:                                                                    
        if domain == 's':
            circuit_domain = circuit.laplace()
        else:
            circuit_domain = circuit                                                           
        try:
            mna_obj = mna.MNA(circuit_domain, solver_method='scipy')
        except Exception as mna_creation_error:
            try:
                mna_obj = mna.MNA(circuit_domain, solver_method='numpy')
            except Exception:
                try:
                    mna_obj = mna.MNA(circuit_domain, solver_method='sympy')
                except Exception as final_error:
                    return f"MNA Creation Error: Failed with all solver methods. Original: {str(mna_creation_error)}, Final: {str(final_error)}"
                              
        matrix_eqs = mna_obj.matrix_equations()
        if matrix_eqs is None:
            return f"MNA Error: matrix_equations() returned None"                                                                       
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

# the raw MNA output is a matrix object. this is parse it and convert it to a readable format.
def _convert_matrix_to_readable(matrix_eqs, mna_obj):
                                                         
    try:                                                               
        unknowns = None                                  
        if hasattr(mna_obj, 'unknowns'):
            unknowns = mna_obj.unknowns
        elif hasattr(mna_obj, 'x'):
            unknowns = mna_obj.x
        elif hasattr(mna_obj, 'variables'):
            unknowns = mna_obj.variables
        elif hasattr(mna_obj, '_unknowns'):
            unknowns = mna_obj._unknowns
        else:                                          
            if hasattr(matrix_eqs, 'lhs') and hasattr(matrix_eqs.lhs, 'free_symbols'):
                unknowns = list(matrix_eqs.lhs.free_symbols)
            elif hasattr(matrix_eqs, 'args') and len(matrix_eqs.args) >= 2:                                                        
                unknowns_vec = matrix_eqs.args[0]
                if hasattr(unknowns_vec, '__iter__'):
                    unknowns = list(unknowns_vec)                                              
        if unknowns is None:                             
            try:
                n_vars = matrix_eqs.A.cols if hasattr(matrix_eqs, 'A') else len(str(matrix_eqs).split(','))
                unknowns = [f"x{i}" for i in range(n_vars)]
            except:
                return f"Could not determine unknowns and matrix structure.\nMatrix form:\n{str(matrix_eqs)}"
                                    
        if hasattr(matrix_eqs, 'A') and hasattr(matrix_eqs, 'b'):
            A_matrix = matrix_eqs.A
            b_vector = matrix_eqs.b
        else:                                  
            return f"Matrix structure not in expected A*x = b format.\nMatrix form:\n{str(matrix_eqs)}"
        equations = []
                                          
        for i in range(A_matrix.rows):
            lhs_terms = []                          
            for j in range(min(A_matrix.cols, len(unknowns))):
                try:
                    coeff = A_matrix[i, j]
                                 
                    if hasattr(coeff, 'is_zero') and coeff.is_zero:
                        continue
                    if str(coeff) == '0' or coeff == 0:
                        continue  
                    unknown = str(unknowns[j])
                    coeff_str = str(coeff)
                                 
                    if coeff_str == '1':
                        lhs_terms.append(f"{unknown}")
                    elif coeff_str == '-1':
                        lhs_terms.append(f"-{unknown}")
                    else:
                                                                  
                        if any(op in coeff_str for op in ['+', '-', '*', '/', '^', 's']):
                            lhs_terms.append(f"({coeff_str})*{unknown}")
                        else:
                            lhs_terms.append(f"{coeff_str}*{unknown}")
                            
                except Exception as coeff_error:                            
                    print(f"Warning: Skipping coefficient at [{i},{j}]: {coeff_error}")
                    continue
                  
            try:
                rhs = str(b_vector[i])                           
                if rhs == '0':
                    rhs = '0'
            except Exception as rhs_error:
                rhs = f"<RHS_ERROR: {rhs_error}>"
                 
            if lhs_terms:
                lhs = " + ".join(lhs_terms).replace("+ -", "- ")
                                               
                lhs = lhs.replace("- -", "+ ").replace("+ +", "+ ")
                equation = f"{lhs} = {rhs}"
                equations.append(equation)
            elif rhs != '0':                                   
                equations.append(f"0 = {rhs}")
        
        if equations:
            return "\n".join(equations)
        else:
            return f"Matrix form (no readable equations generated):\n{str(matrix_eqs)}"
        
    except Exception as e:                                             
        try:                                 
            mna_attrs = [attr for attr in dir(mna_obj) if not attr.startswith('_')]
            matrix_attrs = [attr for attr in dir(matrix_eqs) if not attr.startswith('_')]
            
            debug_info = f"MNA object attributes: {mna_attrs[:10]}...\n"
            debug_info += f"Matrix object attributes: {matrix_attrs[:10]}...\n"
            
            if hasattr(matrix_eqs, 'A'):
                debug_info += f"Matrix dimensions: {matrix_eqs.A.rows}x{matrix_eqs.A.cols}\n"
            
            return f"Conversion Error: {str(e)}\n{debug_info}Matrix form:\n{str(matrix_eqs)}"
        except:
            return f"Conversion Error: {str(e)}\nMatrix form:\n{str(matrix_eqs)}"

# lcapy's spice syntax is a bit different from the standard spice syntax. This function will clean the netlist to be compatible with lcapy.
# it will remove simulation commands like .ac .tran .control which are for numerical analysis with ngspice.
# standardizes dependent sources E, G, F, H 
# sanitizes node names (replacing special characters with underscores)
def clean_netlist_for_lcapy(spice_netlist):
 
    lines = []
    skip_control_block = False
    
    for line in spice_netlist.strip().split('\n'):
        line = line.strip()
        
                                       
        if not line or line.startswith('*'):
            continue
            
                             
        if line.startswith('.control'):
            skip_control_block = True
            continue
        if line.startswith('.endc'):
            skip_control_block = False
            continue
        if skip_control_block:
            continue
            
                               
        if line.startswith('.') or line.startswith('print') or ';' in line:
            continue
            
                                      
        if line.startswith(('ac ', 'dc ', 'tran ', 'op')):
            continue
        
        parts = line.split()
        if len(parts) < 4:
            continue
            
        component = parts[0]
        
                                                      
        if not component[0].upper() in ['R', 'L', 'C', 'V', 'I', 'E', 'F', 'G', 'H']:
            continue
        
                                                                   
        def preserve_symbol(value):
            if value == '<Empty>':
                return '1'
                                        
            if re.match(r'^x_\d+$', value):
                return value
            return value
        
                                                                      
        if component.upper().startswith('E'):
                                                          
            if len(parts) >= 6:                                
                output_p, output_n = parts[1], parts[2]
                input_p, input_n = parts[3], parts[4]
                gain = parts[5]
                ac_gain = parts[6] if len(parts) > 6 else "0"
                
                                                                                           
                output_p = re.sub(r'[^\w]', '_', output_p)
                output_n = re.sub(r'[^\w]', '_', output_n)
                input_p = re.sub(r'[^\w]', '_', input_p)
                input_n = re.sub(r'[^\w]', '_', input_n)
                
                                         
                gain = preserve_symbol(gain)
                if ac_gain == '<Empty>':
                    ac_gain = "0"
                
                lines.append(f"{component} {output_p} {output_n} {input_p} {input_n} {gain} {ac_gain}")
            else:
                continue
        elif component.upper().startswith('G'):
                                                   
            if len(parts) >= 5:                                
                output_p, output_n = parts[1], parts[2]
                input_p, input_n = parts[3], parts[4]
                value = parts[5] if len(parts) > 5 else "1"
                
                                  
                output_p = re.sub(r'[^\w]', '_', output_p)
                output_n = re.sub(r'[^\w]', '_', output_n)
                input_p = re.sub(r'[^\w]', '_', input_p)
                input_n = re.sub(r'[^\w]', '_', input_n)
                
                value = preserve_symbol(value)
                
                lines.append(f"{component} {output_p} {output_n} {input_p} {input_n} {value}")
            else:
                continue
        elif component.upper().startswith('F'):
                                                    
            if len(parts) >= 4:
                output_p, output_n = parts[1], parts[2]
                vcontrol = parts[3]
                value = parts[4] if len(parts) > 4 else "1"
                
                                  
                output_p = re.sub(r'[^\w]', '_', output_p)
                output_n = re.sub(r'[^\w]', '_', output_n)
                vcontrol = re.sub(r'[^\w]', '_', vcontrol)
                
                value = preserve_symbol(value)
                
                lines.append(f"{component} {output_p} {output_n} {vcontrol} {value}")
            else:
                continue
        elif component.upper().startswith('H'):
                                                    
            if len(parts) >= 4:
                output_p, output_n = parts[1], parts[2]
                vcontrol = parts[3]
                value = parts[4] if len(parts) > 4 else "1"
                
                                  
                output_p = re.sub(r'[^\w]', '_', output_p)
                output_n = re.sub(r'[^\w]', '_', output_n)
                vcontrol = re.sub(r'[^\w]', '_', vcontrol)
                
                value = preserve_symbol(value)
                
                lines.append(f"{component} {output_p} {output_n} {vcontrol} {value}")
            else:
                continue
        else:
                                                       
            node1, node2 = parts[1], parts[2]
            value = parts[3]
            
                                                                              
            node1 = re.sub(r'[^\w]', '_', node1)
            node2 = re.sub(r'[^\w]', '_', node2)
            
                                                                                         
            if component.startswith('V_meas') or component.startswith('VI'):
                print(f"⚠️ Warning: Found measurement component {component} in supposedly converted netlist")
                continue
            
                                 
            if value == '<Empty>':
                value = component
            
                                                                                   
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
    if isinstance(data_source, (str, Path)):
        data_path = Path(data_source)
        if not data_path.exists():
            raise FileNotFoundError(f"File not found: {data_path}") 
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        data = data_source
    circuits = {}                              
    if 'results' in data:
                                                
        results = data['results']
        for result in results:
            circuit_id = result.get('circuit_id')
            if use_converted_netlists and 'cleaned_netlist' in result:
                                                   
                netlist = result['cleaned_netlist']
                print(f"Using converted netlist for {circuit_id}")
            elif 'original_netlist_with_measurements' in result:
                                    
                original = result['original_netlist_with_measurements']
                netlist = convert_netlist_remove_n_nodes(original)
                print(f"Converting netlist for {circuit_id}")
            elif 'cleaned_netlist' in result:
                                                                
                original = result['cleaned_netlist']
                if 'N' in original and ('V_meas' in original or 'VI' in original):
                    netlist = convert_netlist_remove_n_nodes(original)
                    print(f"Converting fallback netlist for {circuit_id}")
                else:
                    netlist = original
                    print(f"Using existing clean netlist for {circuit_id}")
            else:
                print(f"No suitable netlist found for {circuit_id}")
                continue
            if circuit_id and netlist:
                circuits[circuit_id] = netlist
                
    elif isinstance(data, dict):                                          
        for circuit_id, netlist in data.items():
            if use_converted_netlists:
                                               
                if 'N' in netlist and ('V_meas' in netlist or 'VI' in netlist):
                    netlist = convert_netlist_remove_n_nodes(netlist)
                    print(f"Converting netlist for {circuit_id}")
                else:
                    print(f"Using existing clean netlist for {circuit_id}")
            circuits[circuit_id] = netlist
    else:
        raise ValueError("Unknown data format")
    
    print(f"Loaded {len(circuits)} circuits (converted N-nodes: {use_converted_netlists})")
    return circuits

def analyze_circuit(netlist, circuit_id):
    try:
                                                                         
        cleaned = clean_netlist_for_lcapy(netlist)
        print(f"\nCircuit {circuit_id}:")
        print(f"Cleaned netlist:\n{cleaned}")
        
        if not cleaned:
            print("No components after cleaning")
            return {'circuit_id': circuit_id, 'error': 'No components after cleaning netlist'}
            
                                              
        lines = [line for line in cleaned.split('\n') if line.strip()]
        num_components = len(lines)
        
                                                                   
        num_capacitors = len([line for line in lines if line.startswith('C')])
        num_inductors = len([line for line in lines if line.startswith('L')])
        num_opamps = len([line for line in lines if line.startswith('E')])
        num_nodes = len(set(node for line in lines for node in line.split()[1:3] if node != '0'))
        
                                    
        complexity_score = num_components + num_capacitors * 2 + num_inductors * 2 + num_opamps * 3
        matrix_size_estimate = num_nodes
        
        print(f"Circuit complexity: {num_components} components, {num_nodes} nodes, score: {complexity_score}")
        print(f"Estimated matrix size: {matrix_size_estimate}×{matrix_size_estimate}")
        
                                                                           
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
        
                                                        
        max_transfer_functions = min(1, len(components))                             
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
                             
    if args.converted_file:
        data_file = args.converted_file
        print(f"Using converted netlists from: {data_file}")
    else:
        data_file = args.labels_file
        print(f"Using data from: {data_file}")
    if not Path(data_file).exists():
        print(f"File not found: {data_file}")
        return
    
                                                
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
                     
    error_types = {}
    for i, (circuit_id, netlist) in enumerate(circuit_items, 1):
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
                                         
            error_key = error_msg.split(':')[0] if ':' in error_msg else error_msg[:50]
            error_types[error_key] = error_types.get(error_key, 0) + 1
            
            print(f"{circuit_id} - Error: {error_msg}")
        else:
            successful += 1
            results.append(result)                                          
            metrics = result.get('complexity_metrics', {})
            score = metrics.get('complexity_score', 0)
            nodes = metrics.get('num_nodes', 0)
            
                                       
            tf_success = sum(1 for v in result.get('transfer_functions', {}).values() 
                           if v not in ["TIMEOUT_OR_ERROR", "SKIPPED_TOO_COMPLEX", "SKIPPED_NO_TRANSFER_FUNCTIONS"])
            mna_success = sum(1 for v in result.get('nodal_equations', {}).values() 
                            if v not in ["TIMEOUT_OR_ERROR", "SKIPPED_TOO_COMPLEX", "SKIPPED_NO_TRANSFER_FUNCTIONS"])
            
            print(f"{circuit_id} - Success (complexity: {score}, nodes: {nodes}, TF: {tf_success}, MNA: {mna_success})")                  
            if args.show_samples and 'transfer_functions' in result:
                for tf_name, tf_expr in result['transfer_functions'].items():
                    if tf_expr not in ["TIMEOUT_OR_ERROR", "SKIPPED_TOO_COMPLEX", "SKIPPED_NO_TRANSFER_FUNCTIONS"]:
                        print(f"  Sample: {tf_name}: {tf_expr}")
                        break                 
    timeout_count = sum(1 for r in results 
                       if any('TIMEOUT_OR_ERROR' in str(v) 
                             for v in r.get('transfer_functions', {}).values()) or
                          any('TIMEOUT_OR_ERROR' in str(v)
                             for v in r.get('nodal_equations', {}).values()))
    skipped_complex = sum(1 for r in results if r.get('skipped', False))
                         
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
    print(f"   Final Analysis Summary:")
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
    
    print(f"   Results saved to {args.output_file}")

if __name__ == "__main__":
    main()