#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path


def setup_paths():
    script_dir = Path(__file__).parent.absolute()
    
    generate_script = script_dir / "ppm_construction" / "data_syn" / "generate.py"
    visualize_script = script_dir / "ppm_construction" / "ft_vlm" / "data_process" / "get_datasets_from_json_data.py"
    equation_script = script_dir / "scripts" / "analyze_synthetic_circuits_robust.py"
    data_dir = script_dir / "ppm_construction" / "data_syn" / "data"
    datasets_dir = script_dir / "datasets"
    
    if not generate_script.exists():
        raise FileNotFoundError(f"Circuit generation script not found: {generate_script}")
    if not visualize_script.exists():
        raise FileNotFoundError(f"Visualization script not found: {visualize_script}")
    if not equation_script.exists():
        raise FileNotFoundError(f"Equation derivation script not found: {equation_script}")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'generate_script': generate_script,
        'visualize_script': visualize_script,
        'equation_script': equation_script,
        'data_dir': data_dir,
        'datasets_dir': datasets_dir,
        'script_dir': script_dir
    }


def run_command(cmd, cwd=None, description=""):
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True
        )
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {description} failed with code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def generate_circuits(paths, args):
    data_file = paths['data_dir'] / f"{args.note}.json"
    
    cmd = [
        sys.executable,
        str(paths['generate_script']),
        "--note", args.circuit_note,
        "--gen_num", str(args.gen_num),
        "--save_path", str(data_file),
        "--num_proc", str(args.num_proc)
    ]
    
    if args.symbolic:
        cmd.extend(["--symbolic"])
    
    success = run_command(cmd, cwd=str(paths['script_dir']), description="Circuit generation")
    
    if success and not data_file.exists():
        print("Warning: Data file was not created!")
        return False
    
    return success


def visualize_circuits(paths, args):
    visualize_dir = paths['visualize_script'].parent
    
    cmd = [
        sys.executable,
        "get_datasets_from_json_data.py",
        "--note", args.note
    ]
    
    success = run_command(cmd, cwd=str(visualize_dir), description="Circuit visualization")
    
    if success:
        output_dir = paths['datasets_dir'] / args.note
        if not output_dir.exists():
            print("Warning: Output directory was not created!")
            return False
    
    return success


def derive_equations(paths, args):
    output_dir = paths['datasets_dir'] / args.note
    labels_file = output_dir / "labels.json"
    
    if not labels_file.exists():
        print(f"Error: Labels file not found: {labels_file}")
        print("Equation derivation requires visualization to be completed first.")
        return False
    
    equations_output = output_dir / "symbolic_equations.json"
    
    cmd = [
        sys.executable,
        str(paths['equation_script']),
        "--labels_file", str(labels_file),
        "--output_file", str(equations_output),
        "--max_circuits", str(args.max_equations)
    ]
    
    if args.show_sample_equations:
        cmd.append("--show_samples")
    
    # Add symbolic question generation if requested
    if hasattr(args, 'generate_symbolic_questions') and args.generate_symbolic_questions:
        cmd.append("--generate_symbolic_questions")
        print("🎓 Generating symbolic transfer function questions...")
    
    if hasattr(args, 'questions_only') and args.questions_only:
        cmd.append("--questions_only")
        print("🎓 Running in questions-only mode...")
    
    success = run_command(cmd, cwd=str(paths['script_dir']), description="Equation derivation")
    
    if success:
        if equations_output.exists():
            print(f"Symbolic equations saved to: {equations_output}")
            
            # Show symbolic questions summary if they were generated
            if hasattr(args, 'generate_symbolic_questions') and args.generate_symbolic_questions:
                print("🎓 Symbolic transfer function questions included in output!")
        else:
            print("Warning: Equation file was not created!")
            return False
    
    return success


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Circuit Generation and Visualization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --note my_circuits --gen_num 100
  %(prog)s --note test_circuits --gen_num 50 --num_proc 4 --circuit_note v12
  %(prog)s --note symbolic_circuits --gen_num 50 --symbolic
  %(prog)s --note analysis_circuits --gen_num 30 --derive_equations --max_equations 15
  %(prog)s --note full_pipeline --gen_num 50 --symbolic --derive_equations --show_sample_equations
  %(prog)s --note symbolic_questions --gen_num 20 --derive_equations --generate_symbolic_questions
  %(prog)s --note training_data --gen_num 30 --questions_only --show_sample_equations
  %(prog)s --note production --gen_num 1000 --num_proc 8 --skip_generation
  %(prog)s --note existing_data --skip_visualization --derive_equations --generate_symbolic_questions
        """)
    
    parser.add_argument(
        "--note",
        type=str,
        required=True,
        help="Dataset name for both the data file and output directory"
    )
    
    generation_group = parser.add_argument_group("Circuit Generation Options")
    generation_group.add_argument(
        "--circuit_note",
        type=str,
        default="v11",
        help="Circuit generation version/note (default: v11)"
    )
    generation_group.add_argument(
        "--gen_num",
        type=int,
        default=50,
        help="Number of circuits to generate (default: 50)"
    )
    generation_group.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes for generation (default: 1)"
    )
    generation_group.add_argument(
        "--symbolic",
        action="store_true",
        help="Generate symbolic circuits (component names like R1, C1, V1 instead of numerical values)"
    )
    
    analysis_group = parser.add_argument_group("Equation Analysis Options")
    analysis_group.add_argument(
        "--derive_equations",
        action="store_true",
        help="Derive symbolic equations from circuits using Lcapy"
    )
    analysis_group.add_argument(
        "--max_equations",
        type=int,
        default=20,
        help="Maximum number of circuits to analyze for equations (default: 20)"
    )
    analysis_group.add_argument(
        "--show_sample_equations",
        action="store_true",
        help="Display sample equations in the terminal during derivation"
    )
    analysis_group.add_argument(
        "--generate_symbolic_questions",
        action="store_true",
        help="Generate symbolic transfer function questions for training"
    )
    analysis_group.add_argument(
        "--questions_only",
        action="store_true",
        help="Only generate symbolic questions (implies --derive_equations and --generate_symbolic_questions)"
    )
    
    control_group = parser.add_argument_group("Pipeline Control")
    control_group.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip circuit generation step (use existing data)"
    )
    control_group.add_argument(
        "--skip_visualization",
        action="store_true",
        help="Skip visualization step (only generate data)"
    )
    control_group.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if output already exists"
    )
    
    args = parser.parse_args()
    
    # Handle questions_only flag
    if args.questions_only:
        args.derive_equations = True
        args.generate_symbolic_questions = True
    
    if args.gen_num <= 0:
        parser.error("--gen_num must be a positive integer")
    if args.num_proc <= 0:
        parser.error("--num_proc must be a positive integer")
    if args.max_equations <= 0:
        parser.error("--max_equations must be a positive integer")
    if args.skip_generation and args.skip_visualization and not args.derive_equations:
        parser.error("Cannot skip both generation and visualization without specifying another action")
    if args.derive_equations and args.skip_visualization and not args.skip_generation:
        parser.error("Cannot derive equations without visualization unless generation is also skipped (for existing datasets)")
    
    return args


def main():
    try:
        args = parse_arguments()
        paths = setup_paths()
        
        data_file = paths['data_dir'] / f"{args.note}.json"
        if data_file.exists() and not args.force and not args.skip_generation:
            response = input(f"Data file already exists: {data_file}\nContinue with existing data? [y/N]: ").strip().lower()
            if response != 'y':
                print("Aborted by user.")
                return 1
        
        if not args.skip_generation:
            if not generate_circuits(paths, args):
                print("Circuit generation failed!")
                return 1
        else:
            if not data_file.exists():
                print(f"Data file not found: {data_file}")
                return 1
        
        if not args.skip_visualization:
            if not visualize_circuits(paths, args):
                print("Circuit visualization failed!")
                return 1
        
        if args.derive_equations:
            if not derive_equations(paths, args):
                print("Equation derivation failed!")
                return 1
        
        print("Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 