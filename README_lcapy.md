# Using Lcapy for Symbolic Circuit Analysis

## Overview

This guide shows how to use the [Lcapy Python library](https://lcapy.readthedocs.io/) to extract symbolic equations from SPICE circuit netlists. Lcapy is a powerful tool for linear circuit analysis that can provide symbolic solutions to circuit equations.

## What We've Accomplished

✅ **Successfully integrated Lcapy** with the circuit reasoning dataset  
✅ **Converted SPICE netlists** to Lcapy-compatible format  
✅ **Extracted symbolic equations** representing circuit behavior  
✅ **Generated system matrices** (A*x = b form) for nodal analysis  
✅ **Calculated node voltage expressions** in symbolic form  

## Scripts Created

### 1. `lcapy_circuit_analysis_fixed.py`
- Comprehensive analysis script
- Handles netlist conversion and cleanup
- Analyzes multiple circuits from the dataset
- Provides detailed error handling

### 2. `extract_equations.py`
- Focused on symbolic equation extraction
- Clean display of results
- Shows system matrices and node voltage expressions

## Key Netlist Modifications for Lcapy

### Original SPICE Format Issues:
```spice
.title Active DC Circuit
R2 2 1 93k           ← Unit suffixes need conversion
E1 4 2 3 0 24        ← Controlled sources not supported
VI1 3 N43 0          ← Measurement sources need conversion
.control             ← SPICE commands must be removed
print -v(3)          ← Analysis commands must be removed
.endc
```

### Lcapy-Compatible Format:
```
R1 1 0 8
R2 2 1 93000.0       ← Units converted to numeric
R3 3 1 35
V1 4 N43 82          
V_VI1 3 N43 0        ← Measurement sources converted to 0V sources
I1 5 6 69
```

## Example Results

### Circuit 40_1 Analysis:

**System Matrix (A*x = b):**
```
A = [[-1, 0, 0, 1, 0, 0, 0, 0],
     [-1/92, 185/8556, 0, 0, 0, -1/93, 0, 0],
     [0, 0, 427/744, -1/24, 0, 0, -1/2, 0],
     ...]

b = [[31], [0], [0], [31], [-14], [0], [0], [0]]
```

**Node Voltage Expressions:**
```
V_1: -550361144977/18727182091
V_2: -353569269477/18727182091  
V_3: 0 (ground reference)
V_4: 30181499844/18727182091
...
```

## Installation Requirements

```bash
pip install lcapy sympy numpy matplotlib
```

## Usage

```python
from lcapy import Circuit

# Load and clean netlist
cleaned_netlist = clean_netlist_for_lcapy(original_spice_netlist)

# Create circuit
circuit = Circuit(cleaned_netlist)

# Get symbolic equations
nodal = circuit.nodal_analysis()
equations = nodal.equations
system_matrix = nodal.A
system_vector = nodal.b

# Get node voltages
for node in circuit.nodes:
    voltage = circuit[node].V
    print(f"V({node}): {voltage}")
```

## Current Limitations & Future Work

### Handled:
- ✅ Basic passive components (R, L, C)
- ✅ Independent voltage/current sources  
- ✅ Unit conversion (k, m, μ, n, p)
- ✅ Measurement voltage sources
- ✅ SPICE command removal

### Not Yet Implemented:
- ❌ Controlled sources (VCVS, CCCS, etc.)
- ❌ Nonlinear components (diodes, transistors)
- ❌ AC analysis (currently DC only)
- ❌ Frequency domain analysis

### Potential Enhancements:
1. **Add controlled source support** using Lcapy's controlled source syntax
2. **Implement AC analysis** for frequency response
3. **Add transfer function extraction** for specific input/output pairs
4. **Generate LaTeX equations** for documentation
5. **Create circuit visualization** using Lcapy's drawing capabilities

## Benefits of Symbolic Analysis

1. **Parameter Sensitivity**: See how circuit behavior changes with component values
2. **Design Optimization**: Optimize component values analytically  
3. **Educational Value**: Understand circuit relationships mathematically
4. **Verification**: Compare with numerical simulation results
5. **Documentation**: Generate human-readable equations for reports

## Example Applications

- **Filter Design**: Analyze frequency response symbolically
- **Amplifier Analysis**: Derive gain and impedance expressions  
- **Power Electronics**: Analyze converter topologies
- **Sensor Circuits**: Derive sensitivity equations
- **Academic Teaching**: Demonstrate circuit principles

The symbolic equations provided by Lcapy offer deep insights into circuit behavior that numerical analysis alone cannot provide. 