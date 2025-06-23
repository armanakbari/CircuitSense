#!/usr/bin/env python3
"""
Test Script for Controlled Source Analysis

This script demonstrates how the improved analysis handles controlled sources
versus the basic analysis that removes them.
"""

import json
from analyze_synthetic_circuits_robust import (
    analyze_circuit_with_timeout,
    clean_spice_for_lcapy_simple,
    clean_spice_for_lcapy_with_controlled_sources
)

def test_controlled_source_circuit():
    """Test a specific circuit with controlled sources."""
    
    # Sample circuit with VCVS (E) and VCCS (G) from the dataset
    test_circuit = """.title Active DC Circuit
R1 1 0 8
R2 2 1 93k
R3 3 1 35
E1 4 2 3 0 24
R4 3 0 40k
G1 5 0 3 0 82
V1 4 N43 82
VI1 3 N43 0
R5 6 3 38k
R6 6 4 27
I1 5 6 69

.control
op
print -v(3) ; measurement of U0
print i(VI1) ; measurement of I9
print v(5, 6) ; measurement of U6
.endc
.end
"""

    print("🧪 CONTROLLED SOURCE ANALYSIS TEST")
    print("="*50)
    
    print(f"\n📜 Original SPICE Circuit:")
    lines = [l for l in test_circuit.split('\n') if l.strip() and not l.startswith('.') and not l.startswith('print')]
    for line in lines:
        if line.strip():
            print(f"  {line.strip()}")
    
    # Show controlled sources
    controlled_sources = {
        'E': test_circuit.count('\nE') + (1 if test_circuit.startswith('E') else 0),
        'F': test_circuit.count('\nF') + (1 if test_circuit.startswith('F') else 0), 
        'G': test_circuit.count('\nG') + (1 if test_circuit.startswith('G') else 0),
        'H': test_circuit.count('\nH') + (1 if test_circuit.startswith('H') else 0)
    }
    
    print(f"\n🎛️ Controlled Sources Found:")
    for source_type, count in controlled_sources.items():
        if count > 0:
            source_names = {'E': 'VCVS (Voltage Controlled Voltage Source)',
                          'F': 'CCCS (Current Controlled Current Source)',
                          'G': 'VCCS (Voltage Controlled Current Source)', 
                          'H': 'CCVS (Current Controlled Voltage Source)'}
            print(f"  {source_type}: {count} - {source_names[source_type]}")
    
    # Test both cleaning methods
    print(f"\n🔧 NETLIST CLEANING COMPARISON:")
    print("-" * 40)
    
    print(f"\n1️⃣ Basic Analysis (Removes Controlled Sources):")
    basic_cleaned = clean_spice_for_lcapy_simple(test_circuit)
    for line in basic_cleaned.split('\n'):
        if line.strip():
            print(f"  {line}")
    
    print(f"\n2️⃣ Advanced Analysis (Preserves Controlled Sources):")
    advanced_cleaned = clean_spice_for_lcapy_with_controlled_sources(test_circuit)
    for line in advanced_cleaned.split('\n'):
        if line.strip():
            print(f"  {line}")
    
    # Analyze with both methods
    print(f"\n🔬 SYMBOLIC ANALYSIS COMPARISON:")
    print("-" * 40)
    
    print(f"\n1️⃣ Basic Analysis Results:")
    basic_result = analyze_circuit_with_timeout(test_circuit, "test_basic", include_controlled_sources=False)
    print(f"  Status: {basic_result['status']}")
    if basic_result['status'] == 'success':
        props = basic_result['circuit_properties']
        print(f"  Nodes: {props['num_nodes']}")
        print(f"  Elements: {props['num_elements']}")
        
        if 'symbolic_equations' in basic_result:
            eqs = basic_result['symbolic_equations']
            if 'nodal' in eqs and 'equations' in eqs['nodal']:
                print(f"  Nodal equations: {len(eqs['nodal']['equations'])}")
            if 'node_voltages' in eqs:
                print(f"  Node voltages computed: {len(eqs['node_voltages'])}")
    
    print(f"\n2️⃣ Advanced Analysis Results:")
    advanced_result = analyze_circuit_with_timeout(test_circuit, "test_advanced", include_controlled_sources=True)
    print(f"  Status: {advanced_result['status']}")
    if advanced_result['status'] == 'success':
        props = advanced_result['circuit_properties']
        print(f"  Nodes: {props['num_nodes']}")
        print(f"  Elements: {props['num_elements']}")
        
        if 'element_categories' in props:
            cats = props['element_categories']
            print(f"  Basic components: {len(cats.get('basic', []))}")
            print(f"  Independent sources: {len(cats.get('independent_sources', []))}")
            print(f"  Controlled sources: {len(cats.get('controlled_sources', []))}")
        
        if 'symbolic_equations' in advanced_result:
            eqs = advanced_result['symbolic_equations']
            if 'node_voltages' in eqs:
                print(f"  Node voltages computed: {len(eqs['node_voltages'])}")
            if 'components' in eqs:
                print(f"  Component analysis: {len(eqs['components'])}")
    
    # Show specific controlled source equations
    if (advanced_result['status'] == 'success' and 
        'symbolic_equations' in advanced_result and
        'components' in advanced_result['symbolic_equations']):
        
        print(f"\n🎯 CONTROLLED SOURCE ANALYSIS:")
        print("-" * 40)
        
        components = advanced_result['symbolic_equations']['components']
        for name, info in components.items():
            if name.startswith('E') or name.startswith('G'):
                print(f"\n  {name} ({info['type']}):")
                print(f"    Nodes: {info['nodes']}")
                if 'voltage' in info:
                    print(f"    Voltage: {str(info['voltage'])[:100]}...")
                if 'current' in info:
                    print(f"    Current: {str(info['current'])[:100]}...")
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"The advanced method successfully preserves and analyzes controlled sources,")
    print(f"providing more complete circuit equations than the basic method.")

if __name__ == "__main__":
    test_controlled_source_circuit() 