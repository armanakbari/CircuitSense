# CircuitSense: A Hierarchical Circuit System Benchmark Bridging Visual Comprehension and Symbolic Reasoning in Engineering Design Process

Welcome to official repo of CircuitSense, a comprehensive visual circuit benchmark that evaluates models' capability in symbolic reasoning and visual mathematical derivation. We introduce a hierarchical synthetic generation pipeline consisting of a grid-based schematic generator and a block diagram generator with
auto-derived symbolic equation labels. This repository contains the code for our hierarchical synthetic generation pipeline.

For further information, please refer to our [preprint](https://arxiv.org/pdf/2509.22339).
[🌐 Webpage] [🤗 Hugginface Dataset] [📑 Paper]

## 💥 News
* **[2025.09.29]** 🔥 We release the code for our hierarchical synthetic generation pipeline
* **[2025.09.25]** The [arxiv paper](https://arxiv.org/pdf/2509.22339) is online

Requirements:
```
# System deps (LaTeX + ngspice)
yes | sudo apt install texlive-full
sudo apt-get install -y libngspice0-dev ngspice

# Python deps
pip install -r requirements_lcapy.txt
pip install PyMuPDF PySpice readchar httpx
```

### Quickstart

Run the end-to-end pipeline via the CLI wrapper in `main.py`.

```bash
# Example 1: default generation + visualization
PYTHONPATH=. python main.py \
  --note grid_v11_240831 \
  --gen_num 50 \
  --num_proc 4

# Example 2: generate symbolic circuits and derive equations
PYTHONPATH=. python main.py \
  --note symbolic_circuits \
  --gen_num 30 \
  --symbolic \
  --derive_equations \
  --show_sample_equations

# Example 3: questions-only mode (implies derive + generate_symbolic_questions)
PYTHONPATH=. python main.py \
  --note training_data \
  --gen_num 30 \
  --questions_only

# Example 4: use existing data (skip generation) and derive equations fast
PYTHONPATH=. python main.py \
  --note existing_data \
  --skip_generation \
  --derive_equations \
  --fast_analysis \
  --max_components 10
```

Key outputs are written under `datasets/<note>/`, including `labels.json` and, when enabled, `symbolic_equations.json`.

### CLI options (from main.py)

- `--note`: Dataset name used for the data JSON and output directory (required)
- Generation:
  - `--circuit_note`: version/note used by generator (default: v11)
  - `--gen_num`: number of circuits to generate (default: 50)
  - `--num_proc`: processes for generation (default: 1)
  - `--symbolic`: generate symbolic circuits
  - `--simple_circuits`: generate simpler circuits
  - `--integrator`: enforce one integrator op-amp per circuit
  - `--rlc`: generate RLC networks (one AC source and at least one reactive component)
  - `--no-meas`: hide all probe drawings except those required
- Analysis:
  - `--derive_equations`: run Lcapy-based symbolic derivation
  - `--max_equations`: maximum circuits to analyze (default: 20)
  - `--show_sample_equations`: print sample equations during derivation
  - `--generate_symbolic_questions`: include symbolic TF questions in output
  - `--questions_only`: only generate symbolic questions (sets the above accordingly)
  - `--max_components`: skip circuits with more than this many components (default: 20)
  - `--fast_analysis`: shorter timeouts for faster processing
- Control:
  - `--skip_generation`: skip generation (use existing data)
  - `--skip_visualization`: skip visualization step
  - `--force`: overwrite existing data without prompt

### Pipeline details

Under the hood, `main.py` orchestrates:

1) Generation: `ppm_construction/data_syn/generate.py` produces `ppm_construction/data_syn/data/<note>.json`

2) Visualization: `ppm_construction/ft_vlm/data_process/get_datasets_from_json_data.py --note <note>` transforms JSON into `datasets/<note>/` with rendered images and `labels.json`.

3) Equation derivation: `scripts/analyze_synthetic_circuits_robust.py` reads `datasets/<note>/labels.json` and writes `datasets/<note>/symbolic_equations.json` (when enabled).

### Legacy script usage (optional)

If you prefer calling scripts directly:

```bash
# 1) Generate netlists/LaTeX
bash ./ppm_construction/data_syn/scripts/run_gen.sh

# 2) Visualize circuit images with LaTeX
PYTHONPATH=. \
python ./ppm_construction/ft_vlm/data_process/get_datasets_from_json_data.py \
  --note grid_v11_240831

# 3) Derive equations
PYTHONPATH=. \
python scripts/analyze_synthetic_circuits_robust.py \
  --labels_file datasets/grid_v11_240831/labels.json \
  --output_file datasets/grid_v11_240831/symbolic_equations.json \
  --max_circuits 50
```

## Citation:
If you find CircuitSense helpful for your research please cite our work:

```
@misc{akbari2025circuitsensehierarchicalcircuitbenchmark,
      title={CircuitSense: A Hierarchical Circuit System Benchmark Bridging Visual Comprehension and Symbolic Reasoning in Engineering Design Process}, 
      author={Arman Akbari and Jian Gao and Yifei Zou and Mei Yang and Jinru Duan and Dmitrii Torbunov and Yanzhi Wang and Yihui Ren and Xuan Zhang},
      year={2025},
      eprint={2509.22339},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.22339}, 
}
```

This repository is based on [MAPS: Advancing Multi-modal Reasoning in Expert-level Physical Science](https://arxiv.org/abs/2501.10768). 


