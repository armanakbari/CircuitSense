#!/usr/bin/env python3

import json
import sys
import os
from pathlib import Path

# Add the scripts directory to path
sys.path.append('scripts')

def test_simple_netlist():
    """Test with a very simple netlist"""
    from analyze_synthetic_circuits_robust import analyze_circuit
    
    # Simple test circuit
    simple_netlist = """R1 1 0 R1
R2 1 2 R2  
V1 2 0 5"""
    
    print("🧪 Testing with simple netlist:")
    print(simple_netlist)
    
    result = analyze_circuit(simple_netlist, "test_simple")
    
    if result:
        print(f"✅ Simple test result: {result.get('transfer_functions', {})}")
        return True
    else:
        print("❌ Simple test failed")
        return False

def test_actual_circuit():
    """Test with an actual circuit from the dataset"""
    # Try to find a labels file
    possible_paths = [
        "datasets/robust_test/labels.json",
        "datasets/grid_v11_240831/labels.json"
    ]
    
    labels_file = None
    for path in possible_paths:
        if Path(path).exists():
            labels_file = path
            break
    
    if not labels_file:
        print("❌ No labels file found for testing")
        return False
    
    with open(labels_file, 'r') as f:
        circuit_data = json.load(f)
    
    # Test the first circuit
    circuit_id, netlist = list(circuit_data.items())[0]
    
    print(f"\n🧪 Testing actual circuit: {circuit_id}")
    print(f"Netlist preview: {netlist[:200]}...")
    
    from analyze_synthetic_circuits_robust import analyze_circuit
    result = analyze_circuit(netlist, circuit_id)
    
    if result and 'error' not in result:
        print(f"✅ Actual circuit test successful!")
        tf_count = len([v for v in result.get('transfer_functions', {}).values() 
                       if v not in ["TIMEOUT_OR_ERROR", "SKIPPED_TOO_COMPLEX", "SKIPPED_NO_TRANSFER_FUNCTIONS"]])
        print(f"   Transfer functions: {tf_count}")
        return True
    else:
        error = result.get('error', 'Unknown error') if result else 'No result'
        print(f"❌ Actual circuit test failed: {error}")
        return False

def main():
    print("🔍 Debug Analysis - Testing Circuit Processing")
    print("=" * 60)
    
    # Test 1: Simple netlist
    simple_ok = test_simple_netlist()
    
    # Test 2: Actual circuit
    actual_ok = test_actual_circuit()
    
    print("\n" + "=" * 60)
    print("🎯 Debug Summary:")
    print(f"   Simple test: {'✅ PASS' if simple_ok else '❌ FAIL'}")
    print(f"   Actual test: {'✅ PASS' if actual_ok else '❌ FAIL'}")
    
    if simple_ok and actual_ok:
        print("🎉 Both tests passed! The analysis should work now.")
    elif simple_ok:
        print("⚠️ Simple test passed but actual failed. Check netlist format/complexity.")
    else:
        print("❌ Basic functionality is broken. Check lcapy installation.")

if __name__ == "__main__":
    main() 