#!/usr/bin/env python3
"""
Create Question-Answering Dataset from Circuit Analysis Results

This script processes the circuit analysis JSON file and creates a Q&A dataset where:
- Questions are derived from SPICE .control block measurement statements
- Answers are ground truth values from SPICE simulation
- Context is the cleaned circuit netlist
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# Add utils to path for SPICE simulation
sys.path.append('utils/simulation')
import subprocess
import tempfile
import os

def parse_control_block(netlist: str) -> List[Dict[str, str]]:
    """
    Parse the .control block to extract measurement statements.
    
    Args:
        netlist: Original SPICE netlist string
        
    Returns:
        List of measurement dictionaries with 'command', 'description', and 'variable'
    """
    measurements = []
    
    # Find .control block
    control_match = re.search(r'\.control\s*\n(.*?)\n\.endc', netlist, re.DOTALL | re.IGNORECASE)
    if not control_match:
        return measurements
    
    control_content = control_match.group(1)
    
    # Parse print statements
    for line in control_content.split('\n'):
        line = line.strip()
        
        # Skip empty lines and 'op' command
        if not line or line == 'op':
            continue
            
        # Parse print statements with comments
        # Format: print i(VI1) ; measurement of I4
        # Format: print v(5, 7) ; measurement of U0
        print_match = re.match(r'print\s+([^;]+)\s*;\s*measurement\s+of\s+(.+)', line, re.IGNORECASE)
        if print_match:
            command = print_match.group(1).strip()
            measurement_var = print_match.group(2).strip()
            
            measurements.append({
                'command': command,
                'measurement_variable': measurement_var,
                'full_line': line
            })
    
    return measurements

def generate_questions_from_measurements(measurements: List[Dict[str, str]], circuit_id: str) -> List[Dict[str, str]]:
    """
    Generate natural language questions from measurement commands.
    
    Args:
        measurements: List of measurement dictionaries
        circuit_id: Circuit identifier
        
    Returns:
        List of question dictionaries
    """
    questions = []
    
    for i, meas in enumerate(measurements):
        command = meas['command']
        var_name = meas['measurement_variable']
        
        # Parse different types of measurements
        if command.startswith('i('):
            # Current measurement: i(VI1) -> "What is the current I4?"
            question = f"What is the current {var_name} in this circuit?"
            measurement_type = "current"
            unit = "A"
            
        elif command.startswith('v('):
            # Voltage measurement: v(5, 7) -> "What is the voltage U0?"
            if ',' in command:
                # Differential voltage
                question = f"What is the voltage {var_name} in this circuit?"
            else:
                # Node voltage
                question = f"What is the voltage {var_name} in this circuit?"
            measurement_type = "voltage"
            unit = "V"
            
        else:
            # Generic measurement
            question = f"What is the value of {var_name} in this circuit?"
            measurement_type = "unknown"
            unit = ""
        
        questions.append({
            'question_id': f"{circuit_id}_q{i+1}",
            'circuit_id': circuit_id,
            'question': question,
            'measurement_command': command,
            'measurement_variable': var_name,
            'measurement_type': measurement_type,
            'unit': unit,
            'original_line': meas['full_line']
        })
    
    return questions

def simulate_circuit_for_answers(original_netlist: str, measurements: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Run NgSpice simulation (DC or transient) to obtain ground-truth answers for the
    requested measurement commands.

    The function now automatically detects whether the original netlist uses a
    transient analysis (``.tran``) or a DC operating-point analysis (``.op``)
    and builds an auxiliary netlist accordingly.  For transient analysis we ask
    NgSpice to print only the *last* simulation value via the ``[-1]`` vector
    index so that the parser can reliably capture a single scalar value.
    
    Args:
        original_netlist: The SPICE netlist produced by the generator (contains
            components, a ``.control`` block and ``.end``).
        measurements:    A list of measurement dictionaries (output of
            ``parse_control_block``).
        
    Returns:
        A mapping {measurement_command -> numeric_value}.  If a value cannot be
        extracted it will be set to ``None``.
    """

    answers: Dict[str, float] = {}
    
    try:
        # ---------------------------------------------------------------
        # Determine whether this is a transient or DC analysis
        # ---------------------------------------------------------------
        tran_match = re.search(r'\btran\s+[^\n]*', original_netlist, re.IGNORECASE)
        is_transient = tran_match is not None
        tran_line = tran_match.group(0).strip() if tran_match else ''

        # ---------------------------------------------------------------
        # Split the netlist at the first .control (if any) to reuse only
        # the component section.  We reconstruct a fresh .control block so
        # that we have full control over the print statements.
        # ---------------------------------------------------------------
        parts = re.split(r'(?i)\.control', original_netlist, maxsplit=1)
        components_part = parts[0].rstrip()  # everything before the first .control

        if not components_part.strip():
            # Fallback – no .control block found; use the whole netlist
            components_part = original_netlist.strip()

        # ---------------------------------------------------------------
        # Build new .control block
        # ---------------------------------------------------------------
        control_lines = [".control"]
        if is_transient:
            control_lines.append(tran_line)
        else:
            control_lines.append("op")

        # Map measurement commands -> commands inserted in print statements.
        # For transient analysis we append '[-1]' so that NgSpice returns only
        # the last time-point value.  We also keep track of possible leading
        # minus sign so that we can adjust after parsing.
        mapped_commands: Dict[str, Tuple[str, int]] = {}

        for meas in measurements:
            orig_cmd = meas['command'].strip()
            sign = 1
            if orig_cmd.startswith('-'):
                sign = -1
                base_cmd = orig_cmd[1:].strip()
            else:
                base_cmd = orig_cmd

            if is_transient:
                # Remove whitespaces inside the expression for consistency
                print_cmd = re.sub(r'\s+', '', base_cmd) + '[-1]'
            else:
                print_cmd = base_cmd

            mapped_commands[orig_cmd] = (print_cmd, sign)
            control_lines.append(f"print {print_cmd}")

        control_lines.append(".endc")

        # Final netlist
        full_netlist = components_part + "\n\n" + "\n".join(control_lines) + "\n.end\n"

        # ---------------------------------------------------------------
        # Write to a temporary file and run NgSpice
        # ---------------------------------------------------------------
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as tmp_file:
            tmp_file.write(full_netlist)
            tmp_filename = tmp_file.name
        
        try:
            result = subprocess.run(
                ['ngspice', '-b', tmp_filename],
                capture_output=True,
                text=True,
                timeout=60  # allow a bit more time for transient sims
            )

            output_text = result.stdout

            # -------------------------------------------------------
            # Extract measurement values from NgSpice output
            # -------------------------------------------------------
            for orig_cmd, (print_cmd, sign) in mapped_commands.items():
                # Remove whitespace for robust matching
                pattern_cmd = re.sub(r'\s+', '', print_cmd, flags=re.IGNORECASE)
                # Regex: command (possibly lower/upper) followed by = value
                pattern = rf"{re.escape(pattern_cmd)}\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)"
                match = re.search(pattern, output_text, re.IGNORECASE)
                
                if match:
                    try:
                        value = float(match.group(1)) * sign
                        answers[orig_cmd] = value
                    except ValueError:
                        answers[orig_cmd] = None
                else:
                    answers[orig_cmd] = None
            
            # Report NgSpice error if no measurements were parsed successfully
            if all(v is None for v in answers.values()) and result.returncode != 0:
                print(f"  NgSpice error: {result.stderr}")
        
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
                
    except Exception as e:
        print(f"  Simulation error: {e}")
        for meas in measurements:
            answers[meas['command']] = None
    
    return answers

def create_qa_dataset(analysis_file: str, output_file: str = None, max_circuits: int = None) -> Dict[str, Any]:
    """
    Create question-answering dataset from circuit analysis results.
    
    Args:
        analysis_file: Path to circuit analysis JSON file
        output_file: Output file path for QA dataset
        max_circuits: Maximum number of circuits to process
        
    Returns:
        QA dataset dictionary
    """
    
    print(f"Loading circuit analysis from {analysis_file}...")
    
    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)
    
    # Get successful circuit results
    successful_circuits = []
    if 'circuit_results' in analysis_data:
        successful_circuits = [
            result for result in analysis_data['circuit_results'] 
            if result['status'] == 'success'
        ]
    else:
        print("No circuit_results found in analysis file")
        return {}
    
    if max_circuits:
        successful_circuits = successful_circuits[:max_circuits]
    
    print(f"Processing {len(successful_circuits)} successful circuits...")
    
    qa_dataset = {
        'metadata': {
            'source_file': analysis_file,
            'total_circuits': len(successful_circuits),
            'total_questions': 0,
            'successful_simulations': 0,
            'failed_simulations': 0,
            'description': 'Circuit analysis question-answering dataset'
        },
        'questions': []
    }
    
    for i, circuit_result in enumerate(successful_circuits):
        circuit_id = circuit_result['circuit_id']
        original_netlist = circuit_result['original_netlist']
        cleaned_netlist = circuit_result['cleaned_netlist']
        
        print(f"Processing circuit {i+1}/{len(successful_circuits)}: {circuit_id}")
        
        # Parse measurements from original netlist
        measurements = parse_control_block(original_netlist)
        
        if not measurements:
            print(f"  No measurements found in {circuit_id}")
            continue
        
        print(f"  Found {len(measurements)} measurements")
        
        # Generate questions
        questions = generate_questions_from_measurements(measurements, circuit_id)
        
        # Get ground truth answers via simulation
        print(f"  Running simulation for ground truth...")
        answers = simulate_circuit_for_answers(original_netlist, measurements)
        
        # Combine questions with answers
        simulation_success = True
        for question in questions:
            command = question['measurement_command']
            answer_value = answers.get(command)
            
            if answer_value is not None:
                question['answer'] = answer_value
                question['answer_formatted'] = f"{answer_value:.6g} {question['unit']}"
                question['has_answer'] = True
            else:
                question['answer'] = None
                question['answer_formatted'] = "Simulation failed"
                question['has_answer'] = False
                simulation_success = False
            
            # Add circuit context
            question['circuit_netlist'] = cleaned_netlist
            question['original_netlist'] = original_netlist
            
            qa_dataset['questions'].append(question)
        
        if simulation_success:
            qa_dataset['metadata']['successful_simulations'] += 1
        else:
            qa_dataset['metadata']['failed_simulations'] += 1
    
    # Filter out questions without valid answers
    original_count = len(qa_dataset['questions'])
    qa_dataset['questions'] = [q for q in qa_dataset['questions'] if q['has_answer']]
    filtered_count = len(qa_dataset['questions'])
    
    qa_dataset['metadata']['total_questions'] = filtered_count
    qa_dataset['metadata']['filtered_out_questions'] = original_count - filtered_count
    
    # Save dataset
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(qa_dataset, f, indent=2)
        print(f"QA dataset saved to {output_file}")
    
    return qa_dataset

def show_dataset_examples(qa_dataset: Dict[str, Any], num_examples: int = 3):
    """Show examples from the QA dataset."""
    
    print(f"\n📊 DATASET SUMMARY")
    print("="*50)
    metadata = qa_dataset['metadata']
    print(f"Total circuits: {metadata['total_circuits']}")
    print(f"Total questions: {metadata['total_questions']}")
    print(f"Successful simulations: {metadata['successful_simulations']}")
    print(f"Failed simulations: {metadata['failed_simulations']}")
    
    if 'filtered_out_questions' in metadata:
        print(f"Questions filtered out (has_answer=False): {metadata['filtered_out_questions']}")
    
    if metadata['total_questions'] > 0:
        success_rate = metadata['successful_simulations'] / metadata['total_circuits'] * 100
        print(f"Simulation success rate: {success_rate:.1f}%")
        
        questions_per_circuit = metadata['total_questions'] / metadata['total_circuits']
        print(f"Average questions per circuit: {questions_per_circuit:.1f}")
    
    # Show examples
    print(f"\n📝 EXAMPLE QUESTIONS")
    print("="*50)
    
    valid_questions = [q for q in qa_dataset['questions'] if q['has_answer']]
    
    for i, question in enumerate(valid_questions[:num_examples]):
        print(f"\nExample {i+1}:")
        print(f"Circuit: {question['circuit_id']}")
        print(f"Question: {question['question']}")
        print(f"Answer: {question['answer_formatted']}")
        print(f"Measurement: {question['measurement_command']}")
        
        # Show part of the circuit
        netlist_lines = question['circuit_netlist'].split('\n')[:5]
        print(f"Circuit (first 5 lines):")
        for line in netlist_lines:
            if line.strip():
                print(f"  {line}")
        print("  ...")


def main():
    """Main function to create QA dataset."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Create QA Dataset from Circuit Analysis')
    parser.add_argument('--analysis-file', default='synthetic_circuits_robust_analysis_basic_only.json',
                       help='Input circuit analysis JSON file')
    parser.add_argument('--output-file', default='circuit_qa_dataset.json',
                       help='Output QA dataset JSON file')
    parser.add_argument('--max-circuits', type=int, default=None,
                       help='Maximum number of circuits to process')
    args = parser.parse_args()
    
    if not Path(args.analysis_file).exists():
        print(f"Analysis file not found: {args.analysis_file}")
        return
    
    print("🔬 CREATING CIRCUIT QA DATASET")
    print("="*50)
    
    # Create QA dataset
    qa_dataset = create_qa_dataset(
        analysis_file=args.analysis_file,
        output_file=args.output_file,
        max_circuits=args.max_circuits
    )
    
    # Show examples
    show_dataset_examples(qa_dataset)
    
    print(f"\n✅ QA dataset creation complete!")
    print(f"Dataset saved to: {args.output_file}")



if __name__ == "__main__":
    main() 