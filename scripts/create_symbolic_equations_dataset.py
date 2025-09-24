#!/usr/bin/env python3
"""
Script to create dataset folder structure from symbolic equations JSON.
Creates folders q1, q2, q3, ... with corresponding image, question, and answer files.
Handles both transfer functions and nodal analysis from s_domain data.
"""

import json
import os
import shutil
from pathlib import Path
import re
import sympy as sp
from sympy import symbols, Matrix, Eq, solve, simplify, Symbol, Function



def extract_symbols_from_matrix(matrix_eq):
    """
    Automatically extract all symbols and functions from a matrix equation.
    
    Parameters:
    -----------
    matrix_eq : sympy.Equality
        The matrix equation
    
    Returns:
    --------
    dict : Dictionary containing categorized symbols
    """
    # Get all symbols from the equation
    all_symbols = matrix_eq.free_symbols
    all_functions = set()
    
    # Extract functions from the unknowns vector
    unknowns_vector = matrix_eq.lhs
    for unknown in unknowns_vector:
        if hasattr(unknown, 'func'):
            all_functions.add(unknown)
    
    # Also extract from the right side if needed
    def extract_functions(expr):
        funcs = set()
        if hasattr(expr, 'atoms'):
            for atom in expr.atoms(sp.Function):
                funcs.add(atom)
        return funcs
    
    all_functions.update(extract_functions(matrix_eq.rhs))
    
    # Categorize symbols
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
    
    # Sort symbols
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
    
    # Sort functions
    for func in all_functions:
        func_str = str(func)
        if 'Vn' in func_str or 'V_n' in func_str:
            categorized['node_voltages'].append(func)
        elif func_str.startswith('I') or func_str.startswith('i'):
            categorized['currents'].append(func)
    
    # Sort lists by their numeric suffix
    def get_number(item):
        match = re.search(r'\d+', str(item))
        return int(match.group()) if match else 0
    
    for key in ['resistors', 'capacitors', 'inductors', 'node_voltages', 'currents']:
        categorized[key] = sorted(categorized[key], key=get_number)
    
    return categorized

def extract_node_equations(matrix_equation):
    """
    Extract individual node equations from a modified nodal analysis matrix equation.
    
    Parameters:
    -----------
    matrix_equation : sympy.Equality
        The matrix equation in the form: Eq(unknowns, A^(-1) * b)
    
    Returns:
    --------
    list of tuples: Each tuple contains (equation_type, equation_string, equation_object)
    """
    
    # Extract the left side (unknowns vector) and right side
    unknowns_vector = matrix_equation.lhs
    right_side = matrix_equation.rhs
    
    # Extract the coefficient matrix A and source vector b
    A = None
    b = None
    
    # Check if right_side is a matrix multiplication
    if hasattr(right_side, 'args') and len(right_side.args) > 0:
        for arg in right_side.args:
            if hasattr(arg, 'exp') and arg.exp == -1 and isinstance(arg.base, Matrix):
                A = arg.base
            elif isinstance(arg, Matrix):
                b = arg
    
    if A is None or b is None:
        # Try to parse as MatMul
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
    
    # Build equations from A * unknowns = b
    equations = []
    num_equations = len(unknowns_vector)
    
    for i in range(num_equations):
        # Build the left side of equation i
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
        
        # Get the right side
        rhs = b[i]
        
        # Create the equation
        eq = Eq(lhs, rhs)
        
        # Format the equation as a string
        eq_str = " + ".join(terms).replace(" + -", " - ")
        eq_str = f"{eq_str} = {rhs}"
        
        # Determine equation type
        eq_type = determine_equation_type(unknowns_vector[i], i)
        
        equations.append((eq_type, eq_str, eq))
    
    return equations

def extract_node_equations_alternative(matrix_equation):
    """
    Alternative method for when the matrix is already evaluated to solutions.
    
    Parameters:
    -----------
    matrix_equation : sympy.Equality
        The matrix equation with solutions
    
    Returns:
    --------
    list of tuples: Each tuple contains (equation_type, equation_string, equation_object)
    """
    
    unknowns_vector = matrix_equation.lhs
    result_matrix = matrix_equation.rhs
    
    # Evaluate if needed
    if hasattr(result_matrix, 'doit'):
        result_matrix = result_matrix.doit()
    
    equations = []
    
    for i in range(len(unknowns_vector)):
        unknown = unknowns_vector[i]
        expr = result_matrix[i]
        
        eq = Eq(unknown, expr)
        eq_type = determine_equation_type(unknown, i) + " Solution"
        
        # Format equation string - NO SIMPLIFICATION
        eq_str = f"{unknown} = {expr}"
        
        equations.append((eq_type, eq_str, eq))
    
    return equations

def determine_equation_type(variable, index):
    """
    Determine the type of equation based on the variable.
    """
    var_str = str(variable)
    
    # Remove function notation
    if '(' in var_str:
        var_name = var_str.split('(')[0]
    else:
        var_name = var_str
    
    # Identify based on patterns
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
    """
    Print equations in a readable format.
    """
    print("\nCircuit Equations:")
    print("=" * 60)
    
    for eq_type, eq_str, eq_obj in equations:
        print(f"\n{eq_type}:")
        print(f"  {eq_str}")
        if show_symbolic:
            print(f"  Symbolic: {eq_obj}")

def process_circuit_matrix(matrix_eq_str):
    """
    Main function that takes a matrix equation string and processes it automatically.
    
    Parameters:
    -----------
    matrix_eq_str : str
        String representation of the matrix equation (can be copied directly from output)
    
    Returns:
    --------
    list of tuples: The extracted equations
    """
    
    print("Analyzing circuit matrix equation...")
    
    # First, we need to evaluate the string to get the actual equation
    # This requires creating the necessary symbols
    
    # Extract all symbol names from the string
    # Updated pattern to handle VCVS-related symbols like Rint1, Cint1, Ad, Ac, Gm, etc.
    symbol_pattern = r'\b([RLC][A-Za-z]*\d+|[A-Z][a-z]*\d*|s)\b'
    found_symbols = set(re.findall(symbol_pattern, matrix_eq_str))
    
    # Extract function names (just the base names without (s))
    # Updated pattern to handle VCVS-related names like VnNinv1, IEint1, IE1, etc.
    function_pattern = r'\b(Vn[A-Za-z]*\d+|I[A-Za-z]*\d+)\b'
    found_functions = set(re.findall(function_pattern, matrix_eq_str))
    
    # Create namespace with symbols
    namespace = {}
    
    # Create s
    namespace['s'] = sp.Symbol('s')
    
    # Create component symbols
    for sym_name in found_symbols:
        if sym_name != 's':
            namespace[sym_name] = sp.Symbol(sym_name, positive=True)
    
    # Create function classes
    for func_name in found_functions:
        namespace[func_name] = sp.Function(func_name)
    
    # Add necessary imports to namespace
    namespace.update({
        'Matrix': Matrix,
        'Eq': Eq,
        'Equation': Eq,  # Sometimes it's written as Equation
        'sp': sp,
        'symbols': symbols,
        'Function': Function
    })
    
    try:
        # Evaluate the string to get the matrix equation
        matrix_eq = eval(matrix_eq_str, namespace)
        
        # Display found symbols
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
        
        # Try to extract equations
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

# Convenience function for direct use
def analyze_circuit(matrix_eq_str):
    """
    Analyze a circuit given its matrix equation as a string.
    
    Parameters:
    -----------
    matrix_eq_str : str
        The matrix equation as a string (can be copied from output)
    
    Example:
    --------
    >>> eq_str = '''Eq(Matrix([
    ...     [Vn1(s)],
    ...     [IV1(s)]
    ... ]), Matrix([
    ...     [1/R1, 1],
    ...     [1, 0]
    ... ])**(-1)*Matrix([
    ...     [0],
    ...     [10/s]
    ... ]))'''
    >>> analyze_circuit(eq_str)
    """
    try:
        equations = process_circuit_matrix(matrix_eq_str)
        print_equations(equations)
        return equations
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None













def extract_transfer_function_components(tf_name):
    """
    Extract source and destination components from transfer function name.
    E.g., 'V1_to_L1' -> ('V1', 'L1')
    """
    if '_to_' in tf_name:
        parts = tf_name.split('_to_')
        return parts[0], parts[1]
    return None, None

def create_transfer_function_question(source, dest, circuit_id):
    """Create a natural language question for transfer function."""
    return f"What is the transfer function from {source} to {dest} in this circuit?"

def create_nodal_analysis_question():
    """Create a natural language question for nodal analysis."""
    return "What is the nodal analysis of this circuit in the s-domain? Provide the matrix equation in the form x = A^(-1)b."

def extract_variable_name_from_equation_type(equation_type):
    """Extract the variable name from equation type string."""
    # For types like "Voltage source current IV1", "Inductor current IL1", etc.
    if "current" in equation_type:
        # Split by spaces and take the last part (the variable name)
        parts = equation_type.split()
        if parts:
            return parts[-1].lower()  # Return just the variable name in lowercase
    return equation_type.lower()

def create_individual_equation_question(equation_type):
    """Create a natural language question for a specific equation type."""
    if "Node" in equation_type and "Solution" not in equation_type:
        # Extract node number/name from equation_type (e.g., "Node 1" -> "1")
        node_name = equation_type.replace("Node ", "")
        return f"Derive the nodal equation for node {node_name} in the s-domain. Express the equation using only the circuit elements and their values as labeled in the diagram. Make sure the final answer is just the symbolic equation Vn{node_name}(s) = ..., where the right side contains only the labeled components and sources from the circuit diagram.."
    elif "Solution" in equation_type:
        node_part = equation_type.replace(" Solution", "")
        if "Node" in node_part:
            node_name = node_part.replace("Node ", "")
            return f"Derive the nodal equation for node {node_name} in the s-domain. Express the equation using only the circuit elements and their values as labeled in the diagram. Make sure the final answer is just the symbolic equation Vn{node_name}(s) = ..., where the right side contains only the labeled components and sources from the circuit diagram."
        else:
            # Extract just the variable name (e.g., "iv1" from "Voltage source current IV1")
            variable_name = extract_variable_name_from_equation_type(node_part)
            return f"What is the solution for the {node_part.lower()} in s-domain of this circuit? Express the equation using only the circuit elements and their values as labeled in the diagram.Make sure the final answer is just the symbolic equation {variable_name}(s) = ..., where the right side contains only the labeled components and sources from the circuit diagram.."
    elif "Voltage source current" in equation_type:
        return f"What is the equation for the {equation_type.lower()} in this circuit?"
    elif "current" in equation_type.lower():
        return f"What is the equation for the {equation_type.lower()} in this circuit?"
    else:
        return f"What is the {equation_type.lower()} equation for this circuit in the s-domain?"


def is_valid_data(data, data_type):
    """
    Check if the data is valid (not an error).
    """
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
    """
    Copy circuit image to target path, trying different locations and extensions.
    Ensures all question folders get an image.
    """
    import glob
    
    # Normalize input directory
    input_dir = Path(input_dir)

    # Try different possible image locations and extensions within input_dir
    possible_paths = [
        input_dir / f'{circuit_id}.jpg',
        input_dir / f'{circuit_id}.png',
    ]
    
    for source_image in possible_paths:
        if source_image.exists():
            shutil.copy2(source_image, target_image_path)
            return True
    
    # If no image found, try to use a default from the same batch
    print(f"WARNING: Image not found for circuit {circuit_id}, looking for alternative...")
    
    # Try to find any image in the same directory as a fallback
    available_images = glob.glob(str(input_dir / '*.jpg'))
    if available_images:
        fallback_image = Path(available_images[0])
        shutil.copy2(fallback_image, target_image_path)
        print(f"Using fallback image: {fallback_image.name}")
        return True
    
    print(f"ERROR: No images found at all!")
    return False

def create_symbolic_equations_dataset(input_dir='datasets/mllm_level1_v7'):
    """Main function to create the dataset."""
    
    # Load the JSON data
    input_dir = Path(input_dir)
    json_file = input_dir / 'symbolic_equations.json'
    if not json_file.exists():
        print(f"JSON file not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create the main dataset folder
    dataset_dir = Path('datasets/symbolic_level15_27')
    dataset_dir.mkdir(exist_ok=True)
    
    # Process each circuit result
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
        
        # Process transfer functions
        transfer_functions = result.get('transfer_functions', {})
        for tf_name, tf_equation in transfer_functions.items():
            total_transfer_functions += 1
            
            if not is_valid_data(tf_equation, 'transfer_function'):
                skipped_transfer_functions += 1
                print(f"Skipping transfer function {tf_name} for circuit {circuit_id}: contains error")
                continue
            
            # Create question folder
            question_folder = dataset_dir / f'q{question_counter}'
            question_folder.mkdir(exist_ok=True)
            
            # Extract source and destination
            source, dest = extract_transfer_function_components(tf_name)
            if not source or not dest:
                source, dest = "input", "output"
            
            # Create question file
            question_text = create_transfer_function_question(source, dest, circuit_id)
            question_file = question_folder / f'q{question_counter}_question.txt'
            with open(question_file, 'w') as f:
                f.write(question_text)
            
            # Create answer file
            answer_file = question_folder / f'q{question_counter}_ta.txt'
            with open(answer_file, 'w') as f:
                f.write(tf_equation)
            
            # Create netlist file (context)
            netlist_file = question_folder / f'q{question_counter}_netlist.txt'
            with open(netlist_file, 'w') as f:
                f.write(cleaned_netlist)
            
            # Copy image file (ensure all folders get an image)
            target_image = question_folder / f'q{question_counter}_image.png'
            if copy_circuit_image(circuit_id, target_image, input_dir):
                print(f"Created q{question_counter}: Transfer function {tf_name} for circuit {circuit_id}")
            else:
                missing_images += 1
            
            question_counter += 1
        
        # Process nodal analysis (s_domain only) - Extract individual equations
        nodal_equations = result.get('nodal_equations', {})
        s_domain = nodal_equations.get('s_domain', '')
        
        if s_domain:
            total_nodal_analysis_circuits += 1
            
            if not is_valid_data(s_domain, 'nodal_analysis'):
                skipped_nodal_analysis += 1
                print(f"Skipping nodal analysis for circuit {circuit_id}: contains error")
                continue
            
            try:
                # Extract individual equations using the analyze_circuit function
                equations = process_circuit_matrix(s_domain)
                
                if equations:
                    print(f"Extracted {len(equations)} equations from circuit {circuit_id}")
                    total_nodal_equations += len(equations)
                    
                    # Create a separate question for each extracted equation
                    for eq_type, eq_str, eq_obj in equations:
                        # Create question folder
                        question_folder = dataset_dir / f'q{question_counter}'
                        question_folder.mkdir(exist_ok=True)
                        
                        # Create question file asking for this specific equation
                        question_text = create_individual_equation_question(eq_type)
                        question_file = question_folder / f'q{question_counter}_question.txt'
                        with open(question_file, 'w') as f:
                            f.write(question_text)
                        
                        # Create answer file with only this specific equation
                        answer_file = question_folder / f'q{question_counter}_ta.txt'
                        with open(answer_file, 'w') as f:
                            f.write(eq_str)
                        
                        # Create netlist file (context) - same for all equations from this circuit
                        netlist_file = question_folder / f'q{question_counter}_netlist.txt'
                        with open(netlist_file, 'w') as f:
                            f.write(cleaned_netlist)
                        
                        # Copy image file (same image for all equations from this circuit)
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
    
    # Print summary
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