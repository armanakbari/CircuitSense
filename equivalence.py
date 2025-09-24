import re
import random
from typing import Dict, Iterable, Optional, Tuple, Union
from lcapy import Circuit
import sympy as sp
from sympy.parsing.latex import parse_latex


SympyLike = Union[str, sp.Expr]


def _normalize_symbol_name(name: str) -> str:
\
\
\
\
\
\
\
\
\
       
                         
    cleaned = name.replace("{", "").replace("}", "")
                                   
    cleaned = cleaned.replace("_", "")
                   
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def _rename_symbols(expr: sp.Expr) -> sp.Expr:
\
\
\
       
    if not isinstance(expr, sp.Basic):
        return expr
    replacements = {}
    for sym in expr.free_symbols:
        new_name = _normalize_symbol_name(sym.name)
        if new_name != sym.name:
                                                                      
            replacements[sym] = sp.Symbol(new_name, **sym.assumptions0)
    if not replacements:
        return expr
    return expr.xreplace(replacements)


def _strip_latex_wrappers(latex: str) -> str:
    text = latex.strip()
                                   
    text = text.replace("$$", "")
    text = text.replace("$", "")
                                                            
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("\\cdot", "*")
    text = text.replace("\\times", "*")
    text = text.replace("\\quad", " ")
                              
    text = text.replace("\\,", "")
    text = text.replace("\\!", "")
    text = text.replace("\\;", ";")
                                          
    text = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", text)
    return text


def _extract_aux_definitions(latex: str) -> Tuple[str, Dict[str, str]]:
\
\
\
\
\
\
\
\
\
       
    defs: Dict[str, str] = {}

                                    
    parts = re.split(r"\bwhere\b", latex, flags=re.IGNORECASE)
    if len(parts) < 2:
        return latex, defs

    main = parts[0]
    defs_text = "".join(parts[1:])

                                                        
    func_pattern = re.compile(r"([A-Za-z\\][A-Za-z0-9_\\{}]*?)\s*\(s\)\s*=\s*([^$;\n]+)")
    for m in func_pattern.finditer(defs_text):
        fname_raw = m.group(1)
        rhs = m.group(2).strip().rstrip(".;,")
        fname_norm = _normalize_symbol_name(fname_raw.replace("\\", ""))
        defs[fname_norm] = rhs

                                                    
    sym_pattern = re.compile(r"([A-Za-z\\][A-Za-z0-9_\\{}]*)\s*=\s*([^$;\n]+)")
    for m in sym_pattern.finditer(defs_text):
        name_raw = m.group(1)
        rhs = m.group(2).strip().rstrip(".;,")
                                                                
        name_norm = _normalize_symbol_name(name_raw.replace("\\", ""))
        if name_norm not in defs:
            defs[name_norm] = rhs

    return main, defs


def _parse_latex_expression(latex: str) -> sp.Expr:
\
\
\
       
    text = _strip_latex_wrappers(latex)

                                                    
    main_text, defs = _extract_aux_definitions(text)

                                                    
    if "=" in main_text:
                                                                    
        rhs_text = main_text.split("=")[-1]
    else:
        rhs_text = main_text

                                        
    try:
        expr = parse_latex(rhs_text)
    except ImportError as e:
        raise ImportError(
            "SymPy LaTeX parsing requires antlr4 runtime. Install via pip: "
            "python -m pip install 'antlr4-python3-runtime==4.11.*' or via conda: "
            "conda install -y -c conda-forge antlr-python-runtime=4.11"
        ) from e

                                                  
    expr = _rename_symbols(expr)

                                                  
    if defs:
                                                                                   
        s_symbol = sp.Symbol("s")
        def_map: Dict[sp.Basic, sp.Expr] = {}
        for name_norm, rhs in defs.items():
            try:
                rhs_expr = _rename_symbols(parse_latex(rhs))
            except ImportError as e:
                raise ImportError(
                    "SymPy LaTeX parsing requires antlr4 runtime. Install via pip: "
                    "python -m pip install 'antlr4-python3-runtime==4.11.*' or via conda: "
                    "conda install -y -c conda-forge antlr-python-runtime=4.11"
                ) from e
                                    
            f = sp.Function(name_norm)
            def_map[f(s_symbol)] = rhs_expr
                               
            def_map[sp.Symbol(name_norm)] = rhs_expr
        expr = expr.xreplace(def_map)

    return expr


def _ensure_sympy(expr: SympyLike) -> sp.Expr:
    if isinstance(expr, sp.Expr):
        return expr
    if isinstance(expr, str):
                                                                    
                                                              
                                                   
        return sp.sympify(expr)
                                                           
    sym = getattr(expr, "sympy", None)
    if sym is not None:
        if callable(sym):
                                                   
            return sym()
        return sym
    raise TypeError("Unsupported expression type for conversion to SymPy")


def _algebraic_equivalence(e1: sp.Expr, e2: sp.Expr) -> bool:
    try:
        diff = sp.simplify(sp.cancel(sp.together(e1 - e2)))
        return diff == 0
    except Exception:
        return False


def _numeric_equivalence(
    e1: sp.Expr,
    e2: sp.Expr,
    samples: int = 10,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> bool:
    rng = random.Random(42)
    symbols: Iterable[sp.Symbol] = sorted(list((e1.free_symbols | e2.free_symbols)), key=lambda s: s.name)
                                                
    if not symbols:
        try:
            v1 = complex(e1.evalf())
            v2 = complex(e2.evalf())
            return abs(v1 - v2) <= max(abs_tol, rel_tol * max(abs(v1), abs(v2), 1.0))
        except Exception:
            return False

                                                                          
    def sample_positive() -> float:
        return 10 ** rng.uniform(-1.0, 1.0)                       

                                         
    def sample_s() -> complex:
                                          
        if rng.random() < 0.4:
            return rng.uniform(0.05, 5.0)
        real = rng.uniform(0.05, 5.0)
        imag = rng.uniform(0.05, 5.0)
        return complex(real, imag)

                                        
    s_sym: Optional[sp.Symbol] = None
    for sym in symbols:
        if _normalize_symbol_name(sym.name) == "s":
            s_sym = sym
            break

    success = 0
    trials = 0
    while trials < samples and success < samples:
        trials += 1
        subs: Dict[sp.Symbol, Union[float, complex]] = {}
        for sym in symbols:
            if sym == s_sym:
                subs[sym] = sample_s()
            else:
                subs[sym] = sample_positive()
        try:
            v1 = complex(e1.evalf(subs=subs))
            v2 = complex(e2.evalf(subs=subs))
        except Exception:
                                                                         
            continue
        err = abs(v1 - v2)
        scale = max(1.0, abs(v1), abs(v2))
        if err <= max(abs_tol, rel_tol * scale):
            success += 1

                                                            
    return success >= max(3, samples // 2)


def compare_expressions(
    expr_lcapy: SympyLike,
    expr_ai_latex: str,
    require_numeric_confirmation: bool = False,
) -> Tuple[bool, Dict[str, str]]:
\
\
\
\
\
\
\
\
       
    e1 = _ensure_sympy(expr_lcapy)
    e1 = _rename_symbols(e1)

    e2 = _parse_latex_expression(expr_ai_latex)

                                     
    if _algebraic_equivalence(e1, e2):
        return True, {"method": "symbolic", "detail": "Expressions simplify to identical forms."}

                                                                                  
    if require_numeric_confirmation:
        if _numeric_equivalence(e1, e2):
            return True, {"method": "numeric", "detail": "Expressions match across random samples."}
        return False, {"method": "numeric", "detail": "Numeric sampling did not confirm equivalence."}

                            
    if _numeric_equivalence(e1, e2):
        return True, {"method": "numeric", "detail": "Expressions match across random samples."}

    return False, {"method": "both", "detail": "Neither symbolic simplification nor numeric sampling matched."}


__all__ = [
    "compare_expressions",
]


if __name__ == "__main__":
                   


    a = Circuit("""
R1 3 0 R1
R2 4 2 R2
R3 5 2 R3
C1 4 3 C1
L1 5 4 L1
R4 5 4 R4
R5 3 5 R5
""")

                                         
                                     


                                                                                                                                                                                                                                                                          

                                                                                    
                                                           
                                                                             
                                        


    a = r"H(s) = (R2*R4) / ((R1 + R4 + R6)*(R2 + R3 + R5) + R2*(R3 + R5))"
    b = r"(R2*R4/(R1*R2 + R1*R3 + R1*R5 + R2*R3 + R2*R4 + R2*R5 + R2*R6 + R3*R4 + R3*R6 + R4*R5 + R5*R6))*1"
    a = _parse_latex_expression(a)
    b = _parse_latex_expression(b)
    is_eq_complex, info_complex = compare_expressions(a, b)
    print("sdfkj",is_eq_complex, info_complex)