import json
import argparse
from pathlib import Path
from lcapy import Circuit, s, t
import re
import multiprocessing
import time
from functools import partial

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

def _compute_transfer_function(circuit, vs_nodes, comp):
    """Top-level function for transfer function computation (must be pickleable)"""
    return str(circuit.transfer(vs_nodes, comp))

def _compute_nodal_analysis(circuit):
    """Top-level function for nodal analysis computation (must be pickleable)"""
    return str(circuit.nodal_analysis(node_prefix='n').nodal_equations())

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
        
        # Handle op-amps (E components) specially - they need all parameters
        if component.upper().startswith('E'):
            if len(parts) >= 7:  # Eint1 9 0 0 Ninv1 100000 0
                output_p, output_n = parts[1], parts[2]
                input_p, input_n = parts[3], parts[4]
                gain, ac_gain = parts[5], parts[6]
                
                # Clean node names
                output_p = re.sub(r'[^\w]', '_', output_p)
                output_n = re.sub(r'[^\w]', '_', output_n)
                input_p = re.sub(r'[^\w]', '_', input_p)
                input_n = re.sub(r'[^\w]', '_', input_n)
                
                # Handle empty values in op-amp gains (use defaults)
                if gain == '<Empty>':
                    gain = "100000"
                if ac_gain == '<Empty>':
                    ac_gain = "0"
                
                lines.append(f"{component} {output_p} {output_n} {input_p} {input_n} {gain} {ac_gain}")
            else:
                # Not enough parameters for op-amp, skip
                continue
        else:
            # Handle regular components (R, L, C, V, I)
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
            print("❌ No components after cleaning")
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
        
        print(f"📊 Circuit complexity: {num_components} components, {num_nodes} nodes, score: {complexity_score}")
        print(f"🧮 Estimated matrix size: {matrix_size_estimate}×{matrix_size_estimate}")
        
        # More permissive thresholds - only skip extremely complex circuits
        if num_components > 20 or complexity_score > 40 or matrix_size_estimate > 12:
            print(f"⚠️ Circuit too complex for symbolic analysis, skipping")
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
            print(f"✅ Circuit created successfully")
        except Exception as e:
            print(f"❌ Failed to create lcapy circuit: {e}")
            return {'circuit_id': circuit_id, 'error': f'Circuit creation failed: {str(e)}'}
        
        print(f"🔍 Finding voltage sources and components...")
        try:
            voltage_sources = find_voltage_sources(circuit)
            components = find_components(circuit)
            print(f"Voltage sources: {voltage_sources}")
            print(f"Components: {components}")
        except Exception as e:
            print(f"❌ Failed to analyze circuit elements: {e}")
            return {'circuit_id': circuit_id, 'error': f'Element analysis failed: {str(e)}'}
        
        if not voltage_sources:
            print("❌ No voltage sources found")
            return {'circuit_id': circuit_id, 'error': 'No voltage sources found'}
            
        if not components:
            print("❌ No components found")
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
        
        print(f"⏱️ Using generous timeouts: TF={timeout_tf}s, Nodal={timeout_nodal}s")
        
        # Try at least one transfer function for testing
        max_transfer_functions = min(1, len(components))  # Start with just 1 to test
        for i, comp in enumerate(components[:max_transfer_functions]):
            print(f"🔄 Analyzing transfer function {i+1}/{max_transfer_functions}: {vs_name} -> {comp}")
            tf_result = safe_computation_mp(
                _compute_transfer_function,
                (circuit, vs_nodes, comp),
                timeout_seconds=timeout_tf,
                description=f"transfer function {vs_name} -> {comp}"
            )
            if tf_result is not None:
                result['transfer_functions'][f"{vs_name}_to_{comp}"] = tf_result
                print(f"✅ Transfer function success: {tf_result[:100]}...")
            else:
                result['transfer_functions'][f"{vs_name}_to_{comp}"] = "TIMEOUT_OR_ERROR"
                print(f"⏰ Transfer function timed out or failed")
        
        # Only try nodal analysis if we got at least one transfer function
        if any(v != "TIMEOUT_OR_ERROR" for v in result['transfer_functions'].values()):
            print(f"🔄 Attempting T-domain nodal equations...")
            nodal_t = safe_computation_mp(
                _compute_nodal_analysis,
                (circuit,),
                timeout_seconds=timeout_nodal,
                description="T-domain nodal equations"
            )
            if nodal_t is not None:
                result['nodal_equations']['t_domain'] = nodal_t
                print(f"✅ T-domain equations success")
            else:
                result['nodal_equations']['t_domain'] = "TIMEOUT_OR_ERROR"
                print(f"⏰ T-domain equations timed out or failed")
        else:
            print(f"⏭️ Skipping nodal analysis (no successful transfer functions)")
            result['nodal_equations']['t_domain'] = "SKIPPED_NO_TRANSFER_FUNCTIONS"
        
        # Skip S-domain for now to focus on getting basic analysis working
        print(f"⏭️ Skipping S-domain analysis (focusing on T-domain first)")
        result['nodal_equations']['s_domain'] = "SKIPPED_FOCUSING_ON_T_DOMAIN"
        
        return result
        
    except Exception as e:
        print(f"❌ Circuit {circuit_id} failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {'circuit_id': circuit_id, 'error': f'Exception: {str(e)}'}

def main():
    # Use default multiprocessing method (fork on Linux, spawn on Windows)
    # This avoids pickling issues with complex lcapy objects
    pass
    
    parser = argparse.ArgumentParser(description="Analyze synthetic circuits with robust timeout handling")
    parser.add_argument('--labels_file', default="datasets/equations_2/labels.json")
    parser.add_argument('--output_file', default="symbolic_equations.json")
    parser.add_argument('--max_circuits', type=int, default=None)
    parser.add_argument('--show_samples', action='store_true', help='Show sample equations during analysis')
    parser.add_argument('--max_components', type=int, default=20, help='Skip circuits with more than this many components (increased default)')
    parser.add_argument('--fast_mode', action='store_true', help='Use shorter timeouts for faster processing')
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
    skipped = 0
    failed = 0
    
    print(f"🔬 Analyzing {len(circuits)} circuits with PERMISSIVE settings...")
    print(f"⚙️ Max components limit: {args.max_components} (increased for better success rate)")
    print(f"⚡ Fast mode: {'ON' if args.fast_mode else 'OFF'}")
    print(f"🛡️ Using multiprocessing timeouts for robust analysis")
    print(f"🎯 Focus: Getting basic transfer functions working first")
    
    # Track detailed error types
    error_types = {}
    
    for i, (circuit_id, netlist) in enumerate(circuits, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(circuits)}] Processing {circuit_id}...")
        
        result = analyze_circuit(netlist, circuit_id)
        
        if result is None:
            failed += 1
            print(f"❌ {circuit_id} - Analysis returned None")
        elif result.get('skipped', False):
            skipped += 1
            results.append(result)
            reason = result.get('reason', 'Unknown')
            print(f"⏭️ {circuit_id} - Skipped: {reason}")
        elif 'error' in result:
            failed += 1
            results.append(result)
            error_msg = result['error']
            
            # Track error types for debugging
            error_key = error_msg.split(':')[0] if ':' in error_msg else error_msg[:50]
            error_types[error_key] = error_types.get(error_key, 0) + 1
            
            print(f"❌ {circuit_id} - Error: {error_msg}")
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
            nodal_success = sum(1 for v in result.get('nodal_equations', {}).values() 
                              if v not in ["TIMEOUT_OR_ERROR", "SKIPPED_TOO_COMPLEX", "SKIPPED_NO_TRANSFER_FUNCTIONS", "SKIPPED_FOCUSING_ON_T_DOMAIN"])
            
            print(f"✅ {circuit_id} - Success (complexity: {score}, nodes: {nodes}, TF: {tf_success}, Nodal: {nodal_success})")
            
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
    total_nodal_success = sum(1 for r in results for v in r.get('nodal_equations', {}).values() 
                             if v not in ["TIMEOUT_OR_ERROR", "SKIPPED_TOO_COMPLEX", "SKIPPED_NO_TRANSFER_FUNCTIONS", "SKIPPED_FOCUSING_ON_T_DOMAIN"])
    
    output = {
        'summary': {
            'total_circuits': len(circuits),
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
            'timeout_count': timeout_count,
            'skipped_complex': skipped_complex,
            'success_rate': successful / len(circuits) if circuits else 0,
            'equation_counts': {
                'transfer_functions': total_tf_success,
                'nodal_equations': total_nodal_success
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
                'permissive_thresholds': True
            }
        },
        'results': results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📊 Final Analysis Summary:")
    print(f"   ✅ Successful circuits: {successful}")
    print(f"   📈 Transfer functions: {total_tf_success}")
    print(f"   🧮 Nodal equations: {total_nodal_success}")
    print(f"   ⏭️ Skipped (complex): {skipped}") 
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏰ Timeouts: {timeout_count}")
    print(f"   📈 Circuit success rate: {successful/len(circuits)*100:.1f}%")
    print(f"   🧮 Complexity range: {min_complexity:.1f} - {max_complexity:.1f} (avg: {avg_complexity:.1f})")
    
    if error_types:
        print(f"\n❌ Error breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {error_type}: {count}")
    
    print(f"   💾 Results saved to {args.output_file}")

if __name__ == "__main__":
    main()