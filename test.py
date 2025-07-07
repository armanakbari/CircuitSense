import lcapy as lc
from lcapy import Circuit, s, t

a = Circuit("""
            R7 0 1 R7
            R1 0 4 R1
            R8 1 2 R8
            L1 5 1 L1
            R9 2 3 R9
            R2 6 2 R2
            R3 3 7 R3
            R10 5 4 R10
            V1 4 8 DC 10 
            C1 6 5 C1   
            R2_1 9 5 R2_1
            L2 7 6 L2
            R4 6 10 R4
            R3_2 7 11 R3_2
            R11 9 8 R11
            R5 8 12 R5
            R12 10 9 R12
            R13 11 10 R13
            R4_3 13 10 R4_3
            R6 11 N1113 R6
            VI1 N1113 13 0 
            R14 13 12 R14""")

netlist_1_1 = Circuit("""
            R2 0 1 R2
            C1 1 2 C1
            R1 0 3 R1
            L1 4 2 L1
            V2 3 4 DC 10""")
netlist_1_2 = """
            R5 0 1 R5
            V1 4 0 DC 10 AC 10
            R6 2 1 R6
            R1 4 1 R1
            L3 2 3 L3
            R2 3 N35 R2
            VI1 N35 5 0
            C1 6 4 C1
            C2 4 2 C2
            R3 4 7 R3
            R3_1 5 2 R3_1
            R2_2 2 5 R2_2
            R7 6 4 R7
            L1 8 6 L1
            R8 4 7 R8
            L2 9 4 L2
            R9 7 5 R9
            R4 7 5 R4
            R10 9 8 R10
            R11 7 9 R11
            R12 7 5 R12"""


test = """R6 0 1 R6
V1 4 0 10
R7 2 1 R7
R1 4 1 R1
L3 2 3 L3
R2 3 N35 R2
V_meas1 N35 5 0
C1 6 4 C1
C2 4 2 C2
R3 4 7 R3
R8 5 2 R8
R4 2 5 R4
R9 6 4 R9
L1 8 6 L1
R10 4 7 R10
L2 9 4 L2
R11 7 5 R11
R5 7 5 R5
R12 9 8 R12
R13 7 9 R13
R14 7 5 R14"""


test_1_1 = Circuit(test)

print(test_1_1.transfer('L1', 'C1'))









# H = netlist_1_1.transfer((3,4), 'C1')
# print(H)

n = test_1_1.nodal_analysis(node_prefix='n')
print(n.nodal_equations())



