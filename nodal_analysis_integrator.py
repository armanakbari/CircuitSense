import lcapy as lc
from lcapy import Circuit, s, t, oo
import sympy as sp

print("="*60)
print("NODAL ANALYSIS OF CIRCUIT WITH IDEAL INTEGRATOR OP AMP")
print("="*60)

# Original circuit with integrator
original_netlist = """R9 1 0 R9
R1 1 2 R1
V1 1 2 DC 10
R2 0 3 R2
R3 2 4 R3
R4 5 1 R4
C1 4 3 C1
R5 3 4 R5
L2 5 4 L2
L1 6 5 L1
R6 4 7 R6
L3 4 6 L3
R7 8 4 R7
R8 6 9 R8
R10 8 7 R10
Rint1 8 Ninv1 Rint1
Cint1 9 Ninv1 Cint1
Eint1 9 0 8 0 GainIntegrator"""

circuit = Circuit(original_netlist)

print("\n1. CIRCUIT TOPOLOGY")
print(f"Nodes: {circuit.node_list}")
print("\nIntegrator components:")
print("- Rint1: Input resistance from node 8 to virtual ground (Ninv1)")
print("- Cint1: Integration capacitor from node 9 to virtual ground (Ninv1)")
print("- Eint1: VCVS that makes V9 = GainIntegrator × V8")

print("\n2. WHY AUTOMATIC NODAL ANALYSIS FAILS")
print("lcapy cannot automatically handle the dependent source (VCVS) Eint1")

# Remove the VCVS to analyze the passive part
passive_netlist = """R9 1 0 R9
R1 1 2 R1
V1 1 2 DC 10
R2 0 3 R2
R3 2 4 R3
R4 5 1 R4
C1 4 3 C1
R5 3 4 R5
L2 5 4 L2
L1 6 5 L1
R6 4 7 R6
L3 4 6 L3
R7 8 4 R7
R8 6 9 R8
R10 8 7 R10
Rint1 8 Ninv1 Rint1
Cint1 9 Ninv1 Cint1"""

passive_circuit = Circuit(passive_netlist)

print("\n3. NODAL ANALYSIS OF PASSIVE CIRCUIT (without VCVS)")
try:
    # Perform nodal analysis on passive part
    nodal_analysis = passive_circuit.nodal_analysis()
    equations = nodal_analysis.nodal_equations()
    
    print(f"\nNumber of nodal equations: {len(equations)}")
    print("\nNodeal equations (passive circuit):")
    for i, eq in enumerate(equations):
        print(f"Node equation {i+1}: {eq}")
    
    # Solve the passive circuit
    print("\n4. DC SOLUTION OF PASSIVE CIRCUIT")
    solution = passive_circuit.solve()
    print("Node voltages (without integrator feedback):")
    for node, voltage in solution.items():
        print(f"V_{node}: {voltage}")
        
except Exception as e:
    print(f"Error in passive analysis: {e}")

print("\n5. MANUAL INTEGRATOR ANALYSIS")
print("\nFor an ideal integrator op amp, we have:")
print("a) Virtual short: V_Ninv1 = V_+ = 0 (assuming non-inverting input at ground)")
print("b) No input current: I_input = 0")
print("c) Integrator transfer function: V9/V8 = -1/(Rint1 × Cint1 × s)")

print("\n6. COMPLETE NODAL ANALYSIS WITH INTEGRATOR")
print("\nTo solve the complete circuit with integrator:")
print("\nStep 1: Write KCL equations for all nodes except reference (node 0)")
print("Step 2: Apply integrator constraint: V9 = -(V8)/(Rint1 × Cint1 × s)")
print("Step 3: Apply virtual ground: V_Ninv1 = 0")
print("Step 4: Solve the resulting system of equations")

print("\n7. KEY INTEGRATOR RELATIONSHIPS")
print("\nAt the integrator:")
print("- Input node: 8")
print("- Virtual ground: Ninv1 (≈ 0V)")  
print("- Output node: 9")
print("- Input current through Rint1: I_in = V8/Rint1")
print("- This current flows through Cint1 to create integration")
print("- Output: V9 = -(1/(Rint1 × Cint1 × s)) × V8")

print("\n8. FREQUENCY DOMAIN ANALYSIS")
print("\nIn the s-domain (Laplace transform):")
print("- Capacitor impedance: Z_C1 = 1/(C1×s)")
print("- Inductor impedance: Z_L = L×s")  
print("- Integrator gain: H(s) = -1/(Rint1 × Cint1 × s)")

# Create a simple example to demonstrate the principle
print("\n9. SIMPLIFIED INTEGRATOR EXAMPLE")
simple_integrator = """
Vin 1 0 AC 1
Rint 1 2 1k
Cfb 3 2 1u
E1 3 0 1 0 -1e6"""

print("\nSimple integrator circuit:")
print("Vin: Input voltage source")
print("Rint: Input resistance") 
print("Cfb: Feedback capacitor")
print("E1: Op amp (high gain VCVS)")
print("\nFor ideal integration: Vout/Vin = -1/(R×C×s)")

print("\n" + "="*60)
print("SUMMARY FOR YOUR CIRCUIT")
print("="*60)
print("\n1. The VCVS (Eint1) prevents automatic nodal analysis")
print("2. Analyze the passive part first to get baseline node voltages")
print("3. Apply integrator constraint: V9 = -(V8)/(Rint1 × Cint1 × s)")
print("4. Use virtual ground assumption: V_Ninv1 = 0")
print("5. The integrator creates frequency-dependent behavior")
print("6. At DC (s=0), the integrator has infinite gain")
print("7. The circuit will have different behavior in time/frequency domain")

print("\nTo complete your analysis:")
print("- Replace 'GainIntegrator' with '-1/(Rint1 × Cint1 × s)' for exact analysis")
print("- Consider DC operating point separately (integrator saturates at DC)")
print("- Use AC analysis for frequency response")
print("- Use transient analysis for time domain behavior") 