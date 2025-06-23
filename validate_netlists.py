#!/usr/bin/env python3
"""
Netlist Validation Script

This script systematically validates that cleaned netlists properly represent
the original circuit topology without losing essential components.

Author: Assistant  
Date: 2024
"""

import json
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict

def parse_spice_components(netlist: str) -> Dict[str, Dict]:
    """
    Parse SPICE netlist and extract all components with their properties.
    
    Returns:
        Dictionary of components with their nodes, types, and values
    """
    components = {}
    lines = netlist.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines, comments, and SPICE commands
        if (not line or line.startswith('*') or line.startswith('.') or 
            line.startswith('print') or ';' in line):
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
            
        name = parts[0]
        node1 = parts[1] if len(parts) > 1 else None
        node2 = parts[2] if len(parts) > 2 else None
        value = parts[3] if len(parts) > 3 else None
        
        component_type = name[0].upper()
        
        components[name] = {
            'type': component_type,
            'nodes': (node1, node2),
            'value': value,
            'full_line': line
        }
    
    return components

def get_circuit_topology(components: Dict[str, Dict]) -> Dict:
    """
    Extract circuit topology information.
    
    Returns:
        Dictionary with nodes, connectivity, and component counts
    """
    nodes = set()
    edges = []
    component_types = defaultdict(int)
    
    for name, comp in components.items():
        node1, node2 = comp['nodes']
        if node1: nodes.add(node1)
        if node2: nodes.add(node2)
        
        if node1 and node2:
            edges.append((node1, node2, name))
        
        component_types[comp['type']] += 1
    
    return {
        'nodes': sorted(list(nodes)),
        'num_nodes': len(nodes),
        'edges': edges,
        'num_edges': len(edges),
        'component_types': dict(component_types),
        'total_components': len(components)
    }

def validate_cleaning_rules(original_components: Dict, cleaned_components: Dict) -> Dict:
    """
    Validate that the cleaning process follows expected rules.
    
    Returns:
        Dictionary with validation results and any issues found
    """
    issues = []
    transformations = []
    
    # Check VI source conversions
    vi_sources = {name: comp for name, comp in original_components.items() 
                  if name.startswith('VI')}
    
    for vi_name, vi_comp in vi_sources.items():
        expected_new_name = f"V_meas{vi_name[2:]}"
        if expected_new_name in cleaned_components:
            cleaned_comp = cleaned_components[expected_new_name]
            # Check if nodes are preserved (with possible name cleaning)
            orig_nodes = vi_comp['nodes']
            clean_nodes = cleaned_comp['nodes']
            
            # Clean node names for comparison
            orig_cleaned = (re.sub(r'[^\w]', '_', orig_nodes[0]), 
                          re.sub(r'[^\w]', '_', orig_nodes[1]))
            
            if clean_nodes == orig_cleaned:
                transformations.append(f"✓ {vi_name} → {expected_new_name} (nodes preserved)")
            else:
                issues.append(f"✗ {vi_name} → {expected_new_name} but nodes changed: {orig_nodes} → {clean_nodes}")
        else:
            issues.append(f"✗ VI source {vi_name} not properly converted to {expected_new_name}")
    
    # Check for missing basic components
    orig_basic = {name: comp for name, comp in original_components.items() 
                  if comp['type'] in ['R', 'L', 'C', 'V', 'I'] and not name.startswith('VI')}
    
    for name, comp in orig_basic.items():
        if name not in cleaned_components:
            issues.append(f"✗ Basic component {name} ({comp['type']}) missing from cleaned netlist")
        else:
            # Check if nodes and value are preserved  
            orig_nodes = comp['nodes']
            clean_comp = cleaned_components[name]
            clean_nodes = clean_comp['nodes']
            
            # Clean node names for comparison
            orig_cleaned = (re.sub(r'[^\w]', '_', orig_nodes[0]), 
                          re.sub(r'[^\w]', '_', orig_nodes[1]))
            
            if clean_nodes == orig_cleaned:
                transformations.append(f"✓ {name} preserved with nodes {clean_nodes}")
            else:
                issues.append(f"✗ {name} nodes changed: {orig_nodes} → {clean_nodes}")
    
    # Check controlled sources are removed (for basic cleaning)
    controlled_sources = {name: comp for name, comp in original_components.items() 
                         if comp['type'] in ['E', 'F', 'G', 'H']}
    
    controlled_in_cleaned = {name: comp for name, comp in cleaned_components.items() 
                           if comp['type'] in ['E', 'F', 'G', 'H']}
    
    if controlled_in_cleaned:
        issues.append(f"✗ Controlled sources found in basic cleaned netlist: {list(controlled_in_cleaned.keys())}")
    else:
        if controlled_sources:
            transformations.append(f"✓ Controlled sources properly removed: {list(controlled_sources.keys())}")
    
    return {
        'issues': issues,
        'transformations': transformations,
        'num_issues': len(issues),
        'is_valid': len(issues) == 0
    }

def validate_topology_preservation(orig_topology: Dict, clean_topology: Dict) -> Dict:
    """
    Validate that essential circuit topology is preserved.
    """
    issues = []
    
    # Node count should be reasonable (may differ due to controlled source removal)
    if clean_topology['num_nodes'] == 0:
        issues.append("✗ No nodes in cleaned circuit")
    
    # Should have some components
    if clean_topology['total_components'] == 0:
        issues.append("✗ No components in cleaned circuit")
    
    # Check if too many components were removed
    component_loss_ratio = 1 - (clean_topology['total_components'] / orig_topology['total_components'])
    if component_loss_ratio > 0.7:  # More than 70% of components lost
        issues.append(f"✗ Too many components lost: {component_loss_ratio:.1%} ({orig_topology['total_components']} → {clean_topology['total_components']})")
    
    # Check basic component preservation
    orig_basic = sum(count for comp_type, count in orig_topology['component_types'].items() 
                    if comp_type in ['R', 'L', 'C', 'V', 'I'])
    clean_basic = sum(count for comp_type, count in clean_topology['component_types'].items() 
                     if comp_type in ['R', 'L', 'C', 'V', 'I'])
    
    # Account for VI → V_meas conversions
    orig_vi_count = orig_topology['component_types'].get('V', 0)  # This includes VI sources as V type
    expected_basic_count = orig_basic  # VI sources become V_meas sources
    
    basic_loss_ratio = 1 - (clean_basic / expected_basic_count) if expected_basic_count > 0 else 0
    if basic_loss_ratio > 0.2:  # More than 20% of basic components lost
        issues.append(f"✗ Too many basic components lost: {basic_loss_ratio:.1%} ({expected_basic_count} → {clean_basic})")
    
    return {
        'issues': issues,
        'component_loss_ratio': component_loss_ratio,
        'basic_loss_ratio': basic_loss_ratio,
        'is_valid': len(issues) == 0
    }

def validate_single_circuit(circuit_data: Dict) -> Dict:
    """
    Validate a single circuit's original vs cleaned netlist.
    """
    circuit_id = circuit_data['circuit_id']
    original_netlist = circuit_data['original_netlist']
    cleaned_netlist = circuit_data['cleaned_netlist']
    
    # Parse both netlists
    orig_components = parse_spice_components(original_netlist)
    clean_components = parse_spice_components(cleaned_netlist)
    
    # Get topology information
    orig_topology = get_circuit_topology(orig_components)
    clean_topology = get_circuit_topology(clean_components)
    
    # Validate cleaning rules
    cleaning_validation = validate_cleaning_rules(orig_components, clean_components)
    
    # Validate topology preservation
    topology_validation = validate_topology_preservation(orig_topology, clean_topology)
    
    # Overall validation
    is_valid = cleaning_validation['is_valid'] and topology_validation['is_valid']
    
    return {
        'circuit_id': circuit_id,
        'is_valid': is_valid,
        'original': {
            'components': orig_components,
            'topology': orig_topology
        },
        'cleaned': {
            'components': clean_components,
            'topology': clean_topology
        },
        'cleaning_validation': cleaning_validation,
        'topology_validation': topology_validation,
        'summary': {
            'total_issues': cleaning_validation['num_issues'] + len(topology_validation['issues']),
            'component_count': f"{orig_topology['total_components']} → {clean_topology['total_components']}",
            'node_count': f"{orig_topology['num_nodes']} → {clean_topology['num_nodes']}"
        }
    }

def validate_all_circuits(json_file: str) -> Dict:
    """
    Validate all circuits in the JSON file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = []
    valid_count = 0
    total_count = len(data['circuit_results'])
    
    print(f"🔍 VALIDATING {total_count} CIRCUITS FROM {json_file}")
    print("=" * 80)
    
    for circuit_data in data['circuit_results']:
        validation = validate_single_circuit(circuit_data)
        results.append(validation)
        
        if validation['is_valid']:
            valid_count += 1
            status = "✅ VALID"
        else:
            status = "❌ INVALID"
        
        print(f"{status} | {validation['circuit_id']:<8} | "
              f"Components: {validation['summary']['component_count']:<12} | "
              f"Nodes: {validation['summary']['node_count']:<8} | "
              f"Issues: {validation['summary']['total_issues']}")
        
        # Show issues for invalid circuits
        if not validation['is_valid']:
            for issue in validation['cleaning_validation']['issues']:
                print(f"    {issue}")
            for issue in validation['topology_validation']['issues']:
                print(f"    {issue}")
    
    print("=" * 80)
    print(f"📊 VALIDATION SUMMARY: {valid_count}/{total_count} circuits valid ({valid_count/total_count:.1%})")
    
    return {
        'total_circuits': total_count,
        'valid_circuits': valid_count,
        'success_rate': valid_count / total_count,
        'results': results
    }

def show_detailed_comparison(circuit_validation: Dict):
    """
    Show detailed comparison for a specific circuit.
    """
    print(f"\n🔍 DETAILED ANALYSIS: {circuit_validation['circuit_id']}")
    print("=" * 60)
    
    orig = circuit_validation['original']
    clean = circuit_validation['cleaned']
    
    print(f"\n📋 ORIGINAL CIRCUIT:")
    print(f"  Components: {orig['topology']['total_components']}")
    print(f"  Nodes: {orig['topology']['num_nodes']} - {orig['topology']['nodes']}")
    print(f"  Types: {orig['topology']['component_types']}")
    
    print(f"\n🧹 CLEANED CIRCUIT:")
    print(f"  Components: {clean['topology']['total_components']}")
    print(f"  Nodes: {clean['topology']['num_nodes']} - {clean['topology']['nodes']}")
    print(f"  Types: {clean['topology']['component_types']}")
    
    print(f"\n🔧 TRANSFORMATIONS:")
    for transform in circuit_validation['cleaning_validation']['transformations']:
        print(f"  {transform}")
    
    if circuit_validation['cleaning_validation']['issues']:
        print(f"\n❌ ISSUES FOUND:")
        for issue in circuit_validation['cleaning_validation']['issues']:
            print(f"  {issue}")
    
    if circuit_validation['topology_validation']['issues']:
        print(f"\n⚠️  TOPOLOGY ISSUES:")
        for issue in circuit_validation['topology_validation']['issues']:
            print(f"  {issue}")

def main():
    """Main validation function."""
    
    # Validate the basic-only analysis
    print("🧪 VALIDATING BASIC-ONLY ANALYSIS")
    basic_results = validate_all_circuits('synthetic_circuits_robust_analysis_basic_only.json')
    
    # Show detailed analysis for any invalid circuits
    invalid_circuits = [r for r in basic_results['results'] if not r['is_valid']]
    
    if invalid_circuits:
        print(f"\n🔍 DETAILED ANALYSIS OF INVALID CIRCUITS:")
        for circuit in invalid_circuits[:3]:  # Show first 3 invalid circuits
            show_detailed_comparison(circuit)
    else:
        print(f"\n✅ ALL CIRCUITS VALID! Showing example of a valid circuit:")
        if basic_results['results']:
            show_detailed_comparison(basic_results['results'][0])

if __name__ == "__main__":
    main() 