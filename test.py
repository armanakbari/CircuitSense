import lcapy as lc
from lcapy import Circuit, s, t

a = Circuit("""
            R1 0 1
            R2 2 1
            R3 3 1
            R4 2 4
            I1 3 0
            I2 0 5
            R5 3 4
            R6 6 3
            R7 7 4
            C1 6 5
            C2 6 7""")


n = a.nodal_analysis(node_prefix="n")
print(n.nodal_equations())
# print(a.C2.V(t))

#print(a.transfer(3,0,1,0))
# print(a[2].V(t))

# l = a.laplace().mesh_analysis()
# print(l.mesh_equations())