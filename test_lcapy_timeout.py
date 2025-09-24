#!/usr/bin/env python3

import multiprocessing
import time
from pathlib import Path

def run_with_timeout(func, timeout_seconds):
                                                      
    def target_func(queue, func):
        try:
            result = func()
            queue.put(('success', result))
        except Exception as e:
            queue.put(('error', str(e)))
    
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target_func, args=(queue, func))
    process.start()
    process.join(timeout=timeout_seconds)
    
    if process.is_alive():
        print(f"⏰ Lcapy computation timed out after {timeout_seconds}s, terminating...")
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

def test_simple_circuit():
                                                             
    try:
        from lcapy import Circuit
        netlist = """
        R1 1 0 1k
        R2 1 2 2k
        V1 2 0 5
        """
        circuit = Circuit(netlist)
        return str(circuit.transfer('2', '0', '1', '0'))
    except Exception as e:
        return f"Error: {e}"

def test_complex_circuit():
                                                             
    try:
        from lcapy import Circuit
                                                
        netlist = """
        R1 1 0 1k
        R2 1 2 2k
        R3 2 3 1k
        R4 3 4 2k
        R5 4 5 1k
        R6 5 6 2k
        R7 6 0 1k
        C1 2 4 1u
        C2 3 5 2u
        L1 1 3 1m
        L2 4 6 2m
        V1 6 0 10
        """
        circuit = Circuit(netlist)
                                                           
        return str(circuit.laplace().nodal_analysis().nodal_equations())
    except Exception as e:
        return f"Error: {e}"

def test_lcapy_timeouts():
                                
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    print("🧪 Testing lcapy timeout mechanism...")
    
                                           
    print("\n1️⃣ Testing simple circuit (should succeed quickly):")
    result, error = run_with_timeout(test_simple_circuit, 10)
    if error:
        print(f"❌ Unexpected error: {error}")
    else:
        print(f"✅ Success: {result}")
    
                                                
    print("\n2️⃣ Testing complex circuit with short timeout (might timeout):")
    result, error = run_with_timeout(test_complex_circuit, 5)
    if error:
        print(f"⏰ Timed out as expected: {error}")
    else:
        print(f"✅ Completed successfully: {result[:100]}...")
    
                                                 
    print("\n3️⃣ Testing complex circuit with longer timeout:")
    result, error = run_with_timeout(test_complex_circuit, 20)
    if error:
        print(f"⏰ Still timed out: {error}")
    else:
        print(f"✅ Completed with longer timeout: {result[:100]}...")
    
    print("\n🎯 Lcapy timeout mechanism test completed!")

if __name__ == "__main__":
    test_lcapy_timeouts() 