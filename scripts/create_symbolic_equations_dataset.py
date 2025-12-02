#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path
import re
import sympy as sp
from sympy import symbols, Matrix, Eq, solve, simplify, Symbol, Function
import glob


def extract_symbols_from_matrix(matrix_eq):                                  
    all_symbols = matrix_eq.free_symbols
    all_functions = set()
                                          
    unknowns_vector = matrix_eq.lhs
    for unknown in unknowns_vector:
        if hasattr(unknown, 'func'):
            all_functions.add(unknown)
                                         
    def extract_functions(expr):
        funcs = set()
        if hasattr(expr, 'atoms'):
            for atom in expr.atoms(sp.Function):
                funcs.add(atom)
        return funcs

    all_functions.update(extract_functions(matrix_eq.rhs))
                  
    categorized = {
        's': None,
        'resistors': [],
        'capacitors': [],
        'inductors': [],
        'other_symbols': [],
        'node_voltages': [],
        'currents': [],
        'unknowns': list(unknowns_vector)
    }
    
                  
    for sym in all_symbols:
        name = str(sym)
        if name == 's':
            categorized['s'] = sym
        elif name.startswith('R') and name[1:].isdigit():
            categorized['resistors'].append(sym)
        elif name.startswith('C') and name[1:].isdigit():
            categorized['capacitors'].append(sym)
        elif name.startswith('L') and name[1:].isdigit():
            categorized['inductors'].append(sym)
        else:
            categorized['other_symbols'].append(sym)
    
                    
    for func in all_functions:
        func_str = str(func)
        if 'Vn' in func_str or 'V_n' in func_str:
            categorized['node_voltages'].append(func)
        elif func_str.startswith('I') or func_str.startswith('i'):
            categorized['currents'].append(func)
    
                                        
    def get_number(item):
        match = re.search(r'\d+', str(item))
        return int(match.group()) if match else 0
    
    for key in ['resistors', 'capacitors', 'inductors', 'node_voltages', 'currents']:
        categorized[key] = sorted(categorized[key], key=get_number)
    
    return categorized

def extract_node_equations(matrix_equation):                                                            
    unknowns_vector = matrix_equation.lhs
    right_side = matrix_equation.rhs
                                                        
    A = None
    b = None
                                               
    if hasattr(right_side, 'args') and len(right_side.args) > 0:
        for arg in right_side.args:
            if hasattr(arg, 'exp') and arg.exp == -1 and isinstance(arg.base, Matrix):
                A = arg.base
            elif isinstance(arg, Matrix):
                b = arg
    
    if A is None or b is None:
                                
        if hasattr(right_side, 'as_coeff_Mul'):
            coeff, matrices = right_side.as_coeff_Mul()
            if hasattr(matrices, 'args'):
                for term in matrices.args:
                    if hasattr(term, 'exp') and term.exp == -1:
                        A = term.base
                    elif isinstance(term, Matrix):
                        b = term
    
    if A is None or b is None:
        raise ValueError("Could not extract A and b from the equation")
    
                                           
    equations = []
    num_equations = len(unknowns_vector)
    
    for i in range(num_equations):
                                           
        lhs = 0
        terms = []
        
        for j in range(num_equations):
            coeff = A[i, j]
            if coeff != 0:
                var = unknowns_vector[j]
                if coeff == 1:
                    terms.append(f"{var}")
                elif coeff == -1:
                    terms.append(f"-{var}")
                else:
                    coeff_str = str(coeff).replace('**', '^')
                    terms.append(f"({coeff_str})*{var}")
                
                lhs += coeff * var          
        rhs = b[i]    
        eq = Eq(lhs, rhs)                              
        eq_str = " + ".join(terms).replace(" + -", " - ")
        eq_str = f"{eq_str} = {rhs}"                      
        eq_type = determine_equation_type(unknowns_vector[i], i)
        equations.append((eq_type, eq_str, eq))
    return equations

def extract_node_equations_alternative(matrix_equation):
    unknowns_vector = matrix_equation.lhs
    result_matrix = matrix_equation.rhs
                  
    if hasattr(result_matrix, 'doit'):
        result_matrix = result_matrix.doit()
    equations = []

    for i in range(len(unknowns_vector)):
        unknown = unknowns_vector[i]
        expr = result_matrix[i]
        eq = Eq(unknown, expr)
        eq_type = determine_equation_type(unknown, i) + " Solution"                                           
        eq_str = f"{unknown} = {expr}"
        equations.append((eq_type, eq_str, eq))
    
    return equations

def determine_equation_type(variable, index):  
    var_str = str(variable)                        
    if '(' in var_str:
        var_name = var_str.split('(')[0]
    else:
        var_name = var_str
                         
    if 'Vn' in var_name or 'V_n' in var_name:
        numbers = re.findall(r'\d+', var_name)
        if numbers:
            return f"Node {numbers[0]}"
        return f"Node voltage {var_name}"
    elif var_name.startswith('I'):
        if 'V' in var_name:
            return f"Voltage source current {var_name}"
        elif 'L' in var_name:
            return f"Inductor current {var_name}"
        elif 'C' in var_name:
            return f"Capacitor current {var_name}"
        else:
            return f"Current {var_name}"
    else:
        return f"Equation {index + 1}"

def print_equations(equations, show_symbolic=False):       
    print("\nCircuit Equations:")
    print("=" * 60)
    
    for eq_type, eq_str, eq_obj in equations:
        print(f"\n{eq_type}:")
        print(f"  {eq_str}")
        if show_symbolic:
            print(f"  Symbolic: {eq_obj}")

def process_circuit_matrix(matrix_eq_str):                                                                                  
    symbol_pattern = r'\b([RLC][A-Za-z]*\d+|[A-Z][a-z]*\d*|s)\b'
    found_symbols = set(re.findall(symbol_pattern, matrix_eq_str))                                                                             
    function_pattern = r'\b(Vn[A-Za-z]*\d+|I[A-Za-z]*\d+)\b'
    found_functions = set(re.findall(function_pattern, matrix_eq_str))                               
    namespace = {}   
    namespace['s'] = sp.Symbol('s')                        
    for sym_name in found_symbols:
        if sym_name != 's':
            namespace[sym_name] = sp.Symbol(sym_name, positive=True)
                        
    for func_name in found_functions:
        namespace[func_name] = sp.Function(func_name)
                               
    namespace.update({
        'Matrix': Matrix,
        'Eq': Eq,
        'Equation': Eq,                                      
        'sp': sp,
        'symbols': symbols,
        'Function': Function
    })
    try:                                                 
        matrix_eq = eval(matrix_eq_str, namespace)                     
        symbols_found = extract_symbols_from_matrix(matrix_eq)
        print(f"\nFound symbols:")
        print(f"  Frequency variable: s")
        if symbols_found['resistors']:
            print(f"  Resistors: {[str(r) for r in symbols_found['resistors']]}")
        if symbols_found['capacitors']:
            print(f"  Capacitors: {[str(c) for c in symbols_found['capacitors']]}")
        if symbols_found['inductors']:
            print(f"  Inductors: {[str(l) for l in symbols_found['inductors']]}")
        print(f"  Unknowns: {[str(u) for u in symbols_found['unknowns']]}")
                         
        try:
            print("\nExtracting circuit equations...")
            equations = extract_node_equations(matrix_eq)
            print("Successfully extracted equations using standard method.")
            return equations
        except ValueError as e:
            print(f"Standard method failed: {e}")
            print("Trying alternative method for solved systems...")
            try:
                equations = extract_node_equations_alternative(matrix_eq)
                print("Successfully extracted solutions.")
                return equations
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")
                raise ValueError("Could not extract equations from the matrix.")
                
    except Exception as e:
        print(f"Error processing matrix equation: {e}")
        print(f"Debug info - Found symbols: {found_symbols}")
        print(f"Debug info - Found functions: {found_functions}")
        raise

                                     
def analyze_circuit(matrix_eq_str):    
    try:
        equations = process_circuit_matrix(matrix_eq_str)
        print_equations(equations)
        return equations
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None


def extract_transfer_function_components(tf_name):
    if '_to_' in tf_name:
        parts = tf_name.split('_to_')
        return parts[0], parts[1]
    return None, None

def create_transfer_function_question(source, dest, circuit_id):
                                                                   
    return f"What is the transfer function from {source} to {dest} in this circuit?"

def create_nodal_analysis_question():
                                                                
    return "What is the nodal analysis of this circuit in the s-domain? Provide the matrix equation in the form x = A^(-1)b."

def extract_variable_name_from_equation_type(equation_type):                                                                    
    if "current" in equation_type:
                                                                    
        parts = equation_type.split()
        if parts:
            return parts[-1].lower()                                              
    return equation_type.lower()

def create_individual_equation_question(equation_type):
                                                                          
    if "Node" in equation_type and "Solution" not in equation_type:
                                                                             
        node_name = equation_type.replace("Node ", "")
        return f"Derive the nodal equation for node {node_name} in the s-domain. Express the equation using only the circuit elements and their values as labeled in the diagram. Make sure the final answer is just the symbolic equation Vn{node_name}(s) = ..., where the right side contains only the labeled components and sources from the circuit diagram.."
    elif "Solution" in equation_type:
        node_part = equation_type.replace(" Solution", "")
        if "Node" in node_part:
            node_name = node_part.replace("Node ", "")
            return f"Derive the nodal equation for node {node_name} in the s-domain. Express the equation using only the circuit elements and their values as labeled in the diagram. Make sure the final answer is just the symbolic equation Vn{node_name}(s) = ..., where the right side contains only the labeled components and sources from the circuit diagram."
        else:
                                                                                            
            variable_name = extract_variable_name_from_equation_type(node_part)
            return f"What is the solution for the {node_part.lower()} in s-domain of this circuit? Express the equation using only the circuit elements and their values as labeled in the diagram.Make sure the final answer is just the symbolic equation {variable_name}(s) = ..., where the right side contains only the labeled components and sources from the circuit diagram.."
    elif "Voltage source current" in equation_type:
        return f"What is the equation for the {equation_type.lower()} in this circuit?"
    elif "current" in equation_type.lower():
        return f"What is the equation for the {equation_type.lower()} in this circuit?"
    else:
        return f"What is the {equation_type.lower()} equation for this circuit in the s-domain?"


def is_valid_data(data, data_type):
    if isinstance(data, str):
        error_patterns = [
            "TIMEOUT_OR_ERROR",
            "MNA Creation Error",
            "SKIPPED_NO_TRANSFER_FUNCTIONS",
            "Error:",
            "unsupported operand type"
        ]
        for pattern in error_patterns:
            if pattern in data:
                return False
    return True

def copy_circuit_image(circuit_id, target_image_path, input_dir):
                         
    input_dir = Path(input_dir)

                                                                            
    possible_paths = [
        input_dir / f'{circuit_id}.jpg',
        input_dir / f'{circuit_id}.png',
    ]
    
    for source_image in possible_paths:
        if source_image.exists():
            shutil.copy2(source_image, target_image_path)
            return True
    
                                                                 
    print(f"WARNING: Image not found for circuit {circuit_id}, looking for alternative...")
    
                                                               
    available_images = glob.glob(str(input_dir / '*.jpg'))
    if available_images:
        fallback_image = Path(available_images[0])
        shutil.copy2(fallback_image, target_image_path)
        print(f"Using fallback image: {fallback_image.name}")
        return True
    
    print(f"ERROR: No images found at all!")
    return False

def create_symbolic_equations_dataset(input_dir='datasets/mllm_level1_v7'):                   
    input_dir = Path(input_dir)
    json_file = input_dir / 'symbolic_equations.json'
    if not json_file.exists():
        print(f"JSON file not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)                           
    dataset_dir = Path('datasets/symbolic_level15_27')
    dataset_dir.mkdir(exist_ok=True)
    
                                 
    results = data.get('results', [])
    question_counter = 1
    
    total_transfer_functions = 0
    total_nodal_analysis_circuits = 0
    total_nodal_equations = 0
    skipped_transfer_functions = 0
    skipped_nodal_analysis = 0
    missing_images = 0
    
    for result in results:
        circuit_id = result['circuit_id']
        cleaned_netlist = result.get('cleaned_netlist', '')                         
        transfer_functions = result.get('transfer_functions', {})
        for tf_name, tf_equation in transfer_functions.items():
            total_transfer_functions += 1
            
            if not is_valid_data(tf_equation, 'transfer_function'):
                skipped_transfer_functions += 1
                print(f"Skipping transfer function {tf_name} for circuit {circuit_id}: contains error")
                continue
                                  
            question_folder = dataset_dir / f'q{question_counter}'
            question_folder.mkdir(exist_ok=True)                              
            source, dest = extract_transfer_function_components(tf_name)
            if not source or not dest:
                source, dest = "input", "output"
                     
            question_text = create_transfer_function_question(source, dest, circuit_id)
            question_file = question_folder / f'q{question_counter}_question.txt'
            with open(question_file, 'w') as f:
                f.write(question_text)
                  
            answer_file = question_folder / f'q{question_counter}_ta.txt'
            with open(answer_file, 'w') as f:
                f.write(tf_equation)
                             
            netlist_file = question_folder / f'q{question_counter}_netlist.txt'
            with open(netlist_file, 'w') as f:
                f.write(cleaned_netlist)
                                                
            target_image = question_folder / f'q{question_counter}_image.png'
            if copy_circuit_image(circuit_id, target_image, input_dir):
                print(f"Created q{question_counter}: Transfer function {tf_name} for circuit {circuit_id}")
            else:
                missing_images += 1

            question_counter += 1
                                                                    
        nodal_equations = result.get('nodal_equations', {})
        s_domain = nodal_equations.get('s_domain', '')
        
        if s_domain:
            total_nodal_analysis_circuits += 1
            if not is_valid_data(s_domain, 'nodal_analysis'):
                skipped_nodal_analysis += 1
                print(f"Skipping nodal analysis for circuit {circuit_id}: contains error")
                continue
            try:                                                               
                equations = process_circuit_matrix(s_domain)
                if equations:
                    print(f"Extracted {len(equations)} equations from circuit {circuit_id}")
                    total_nodal_equations += len(equations)
                                                       
                    for eq_type, eq_str, eq_obj in equations:
                                                
                        question_folder = dataset_dir / f'q{question_counter}'
                        question_folder.mkdir(exist_ok=True)                                                        
                        question_text = create_individual_equation_question(eq_type)
                        question_file = question_folder / f'q{question_counter}_question.txt'
                        with open(question_file, 'w') as f:
                            f.write(question_text)
                                                 
                        answer_file = question_folder / f'q{question_counter}_ta.txt'
                        with open(answer_file, 'w') as f:
                            f.write(eq_str)
                                                                      
                        netlist_file = question_folder / f'q{question_counter}_netlist.txt'
                        with open(netlist_file, 'w') as f:
                            f.write(cleaned_netlist)
                                                              
                        target_image = question_folder / f'q{question_counter}_image.png'
                        if copy_circuit_image(circuit_id, target_image, input_dir):
                            print(f"Created q{question_counter}: {eq_type} equation for circuit {circuit_id}")
                        else:
                            missing_images += 1
                        
                        question_counter += 1
                else:
                    print(f"No equations extracted from circuit {circuit_id}")
                    skipped_nodal_analysis += 1
            except Exception as e:
                print(f"Error extracting equations from circuit {circuit_id}: {e}")
                skipped_nodal_analysis += 1
                continue
    
                   
    print(f"\nDataset creation complete!")
    print(f"Total questions created: {question_counter - 1}")
    print(f"Transfer functions: {total_transfer_functions} total, {total_transfer_functions - skipped_transfer_functions} valid, {skipped_transfer_functions} skipped")
    print(f"Nodal analysis: {total_nodal_analysis_circuits} circuits processed, {total_nodal_equations} individual equations created, {skipped_nodal_analysis} circuits skipped")
    print(f"Missing images: {missing_images}")
    print(f"Dataset folder: {dataset_dir.absolute()}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create dataset from symbolic equations JSON and images.')
    parser.add_argument('--input_dir', type=str, default='datasets/mllm_level1_v7',
                        help='Directory containing symbolic_equations.json and circuit images')
    args = parser.parse_args()
    create_symbolic_equations_dataset(input_dir=args.input_dir)