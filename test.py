import lcapy as lc
from lcapy import Circuit, s, t, oo, NodalAnalysis, mna
from lcapy import *
from sympy import Eq, Matrix, Symbol, simplify, limit, symbols, MatMul, Wild

# def extract_node_equations(matrix_equation):
#     """
#     Extract individual node equations from a modified nodal analysis matrix equation.
    
#     Parameters:
#     -----------
#     matrix_equation : sympy.Equality
#         The matrix equation in the form: Eq(unknowns, A^(-1) * b)
    
#     Returns:
#     --------
#     list of tuples: Each tuple contains (equation_type, equation_string, equation_object)
#     """
    
#     # Case 1: matrix_equation provides A and b directly (preferred)
#     if hasattr(matrix_equation, 'A') and hasattr(matrix_equation, 'b'):
#         A = matrix_equation.A
#         b = matrix_equation.b
#         unknowns_vector = getattr(matrix_equation, 'x', None)
#         if unknowns_vector is None:
#             unknowns_vector = getattr(matrix_equation, 'unknowns', None)
#         if unknowns_vector is None and hasattr(matrix_equation, 'lhs'):
#             unknowns_vector = matrix_equation.lhs
#     else:
#         # Case 2: matrix_equation is an Eq(unknowns, solution) form
#         if hasattr(matrix_equation, 'lhs') and hasattr(matrix_equation, 'rhs'):
#             unknowns_vector = matrix_equation.lhs
#             right_side = matrix_equation.rhs

#             A = None
#             b = None

#             # Try pattern: A**-1 * b
#             if hasattr(right_side, 'args') and len(right_side.args) > 0:
#                 mats = []
#                 for arg in right_side.args:
#                     if hasattr(arg, 'exp') and getattr(arg, 'exp', None) == -1 and hasattr(getattr(arg, 'base', None), 'shape'):
#                         A = arg.base
#                     elif hasattr(arg, 'shape'):
#                         mats.append(arg)
#                 # Pick b as the column vector among remaining factors
#                 if b is None and mats:
#                     col_mats = [m for m in mats if getattr(m, 'shape', (0, 0))[1] == 1]
#                     b = col_mats[0] if col_mats else mats[-1]

#             # Try MatMul terms
#             if (A is None or b is None) and hasattr(right_side, 'as_coeff_Mul'):
#                 coeff, matrices = right_side.as_coeff_Mul()
#                 if hasattr(matrices, 'args'):
#                     mats = []
#                     for term in matrices.args:
#                         if hasattr(term, 'exp') and getattr(term, 'exp', None) == -1 and hasattr(getattr(term, 'base', None), 'shape'):
#                             A = term.base
#                         elif hasattr(term, 'shape'):
#                             mats.append(term)
#                     if b is None and mats:
#                         col_mats = [m for m in mats if getattr(m, 'shape', (0, 0))[1] == 1]
#                         b = col_mats[0] if col_mats else mats[-1]

#             # Try pattern: A.LUsolve(b) or similar
#             if (A is None or b is None) and hasattr(right_side, 'func') and hasattr(right_side, 'args'):
#                 func_name = getattr(right_side.func, '__name__', '')
#                 if func_name.lower().endswith('lusolve') and len(right_side.args) == 2:
#                     A, b = right_side.args

#             # Try pattern: Eq(A*unknowns, b)
#             if (A is None or b is None) and isinstance(matrix_equation.lhs, MatMul):
#                 lhs_mul = matrix_equation.lhs
#                 # Pick a square matrix as A
#                 for factor in lhs_mul.args:
#                     if hasattr(factor, 'shape'):
#                         r, c = factor.shape
#                         if r == c and r > 0:
#                             A = factor
#                             break
#                 # Unknowns vector is the rightmost column vector
#                 for factor in reversed(lhs_mul.args):
#                     if hasattr(factor, 'shape'):
#                         r, c = factor.shape
#                         if c == 1:
#                             unknowns_vector = factor
#                             break
#                 b = matrix_equation.rhs if hasattr(matrix_equation.rhs, 'shape') else None

#             # Try pattern: Eq(unknowns, A**-1 * b) - already attempted above as right_side MatMul,
#             # but ensure unknowns_vector is set if not indexable
#             if unknowns_vector is None and hasattr(matrix_equation, 'lhs'):
#                 unknowns_vector = matrix_equation.lhs

#             if A is None or b is None:
#                 raise ValueError("Could not extract A and b from the equation")
#         else:
#             raise ValueError("Unsupported matrix_equation type; expected attributes A/b or an Eq with lhs/rhs")
    
#     # Build equations from A * unknowns = b
#     equations = []
#     # Prefer size from A
#     num_equations = getattr(A, 'rows', None) or getattr(b, 'rows', None)
#     if num_equations is None:
#         num_equations = len(b)

#     # Build an indexable list of unknowns; if not available, create placeholders
#     unknowns_list = None
#     try:
#         if isinstance(unknowns_vector, Matrix):
#             r, c = unknowns_vector.shape
#             if c == 1:
#                 unknowns_list = [unknowns_vector[i, 0] for i in range(num_equations)]
#             elif r == 1:
#                 unknowns_list = [unknowns_vector[0, i] for i in range(num_equations)]
#     except Exception:
#         unknowns_list = None
#     if unknowns_list is None:
#         # Fall back to simple symbols x0..x{n-1}
#         unknowns_list = symbols('x0:%d' % num_equations)
    
#     for i in range(num_equations):
#         # Build the left side of equation i
#         lhs = 0
#         terms = []
        
#         for j in range(num_equations):
#             coeff = A[i, j]
#             if coeff != 0:
#                 var = unknowns_list[j]
#                 if coeff == 1:
#                     terms.append(f"{var}")
#                 elif coeff == -1:
#                     terms.append(f"-{var}")
#                 else:
#                     coeff_str = str(coeff).replace('**', '^')
#                     terms.append(f"({coeff_str})*{var}")
                
#                 lhs += coeff * var
        
#         # Get the right side
#         rhs = b[i]
        
#         # Create the equation
#         eq = Eq(lhs, rhs)
        
#         # Format the equation as a string
#         eq_str = " + ".join(terms).replace(" + -", " - ")
#         eq_str = f"{eq_str} = {rhs}"
        
#         # Determine equation type (placeholder)
#         eq_type = determine_equation_type(unknowns_list[i], i)
        
#         equations.append((eq_type, eq_str, eq))
    
#     return equations

# def determine_equation_type(var, index):
#     # Minimal stub; customize as needed
#     try:
#         name = str(var)
#         if 'I(' in name or name.startswith('I'):
#             return 'current_eq'
#         return 'node_eq'
#     except Exception:
#         return 'node_eq'

# a = Circuit("""
#             L1 1 0 L1
#             Rint1 2 31 Rint1
#             Cint1 1 31 Cint1
#             Eint1 1 0 0 31 Ad 0
#             R1 1 0 R1
#             R4 2 3 R4
#             R2 0 2 R2
#             R3 3 5 R3
#             V1 5 0 step
#             """)
# b = mna.MNA(a.laplace(), solver_method='scipy')  
# me = b.matrix_equations()
# b = extract_node_equations(me)

# # Helpful mapping from x indices to physical variables (from matrix_equations lhs)
# try:
#     lhs = getattr(me, 'lhs', None)
#     mapping = []
#     if isinstance(lhs, Matrix):
#         r, c = lhs.shape
#         if c == 1:
#             mapping = [str(lhs[i, 0]) for i in range(r)]
#         elif r == 1:
#             mapping = [str(lhs[0, i]) for i in range(c)]
#     if mapping:
#         print("\nUnknown mapping (xk -> variable):")
#         for i, name in enumerate(mapping):
#             print(f"x{i} -> {name}")
# except Exception:
#     pass

# print("\nExtracted equations:")
# for item in b:
#     print(item)

# # Apply Ad -> oo to equations: detect linear Ad term (Ad*var + rest = 0) and enforce var = 0
# Ad = Symbol('Ad')
# varW = Wild('var', exclude=[Ad])
# restW = Wild('rest', exclude=[Ad])

# limited_eqs = []
# for eq_type, eq_str, eq_obj in b:
#     lhs, rhs = eq_obj.lhs, eq_obj.rhs
#     resid = simplify(lhs - rhs)
#     if Ad in resid.free_symbols:
#         m = resid.match(Ad*varW + restW)
#         if m is not None and Ad not in m.get('rest', 0).free_symbols:
#             new_eq = Eq(simplify(m['var']), 0)
#             limited_eqs.append((eq_type, f"{str(m['var'])} = 0", new_eq))
#             continue
#         # Fallback: try formal limit on both sides
#         try:
#             new_lhs = simplify(limit(lhs, Ad, oo))
#             new_rhs = simplify(limit(rhs, Ad, oo))
#             limited_eqs.append((eq_type, f"{new_lhs} = {new_rhs}", Eq(new_lhs, new_rhs)))
#             continue
#         except Exception:
#             pass
#     # No Ad or no transformation
#     limited_eqs.append((eq_type, eq_str, eq_obj))

# print("\nAfter Ad -> oo limit:")
# for item in limited_eqs:
#     print(item)

# we already now what is the impedance of each component

# total impedance seen from the input (impedance between V1 after removing that)
#print(a.impedance(2,0))

# impedance of output (remove v1, nodes of output)


import sympy as sp
from lcapy import s

# declare symbols
R1,R2,R3,Rint1 = sp.symbols('R1 R2 R3 Rint1', positive=True)
C1,Cint1       = sp.symbols('C1 Cint1', positive=True)
Ad,V1          = sp.symbols('Ad V1', real=True)

num = (
    Ad*C1*Cint1*R2*R3*Rint1*s**2
    + Ad*Cint1*R2*Rint1*s + Ad*Cint1*R3*Rint1*s
    + C1*Cint1*R2*R3*Rint1*s**2
    + C1*R2*R3*s
    + Cint1*R2*Rint1*s + Cint1*R3*Rint1*s
    + R2 + R3
)

den = s * (
    Ad*C1*Cint1*R1*R2*R3*s**2
    + Ad*C1*Cint1*R1*R2*Rint1*s**2
    + Ad*C1*Cint1*R1*R3*Rint1*s**2
    + Ad*C1*Cint1*R2*R3*Rint1*s**2
    + Ad*C1*R1*R2*s + Ad*Cint1*R1*R2*s + Ad*Cint1*R1*R3*s
    + Ad*Cint1*R2*Rint1*s + Ad*Cint1*R3*Rint1*s
    + C1*Cint1*R1*R2*R3*s**2
    + C1*Cint1*R1*R2*Rint1*s**2
    + C1*Cint1*R1*R3*Rint1*s**2
    + C1*Cint1*R2*R3*Rint1*s**2
    + C1*R1*R2*s + C1*R1*R3*s + C1*R2*R3*s
    + Cint1*R1*R2*s + Cint1*R1*R3*s
    + Cint1*R2*Rint1*s + Cint1*R3*Rint1*s
    + R2 + R3
)

V = V1 * num / den
# V is a SymPy/lcapy s-domain expression; you can use simplify, subs, etc.
V_inf = sp.simplify(sp.limit(V, Ad, sp.oo))
print(V_inf)