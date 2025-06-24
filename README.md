Generating synthetic circuits



Requirements:
```
# install latex / ngspice
yes | sudo apt install texlive-full
sudo apt-get install -y libngspice0-dev ngspice

# python libraries
pip install PyMuPDF PySpice readchar httpx
```

#### Whole pipeline:


```bash
# Generating netlist/latex
bash ./ppm_construction/data_syn/scripts/run_gen.sh

# Visualizing circuit images with latex
PYTHONPATH=. \
python ./ppm_construction/ft_vlm/data_process/get_datasets_from_json_data.py \
--note grid_v11_240831


# Inputting netlists into Lcapy after some modification
python analyze_synthetic_circuits_robust.py --max-circuits 50 --labels-file dataset/grid_v11_240831/labels.json

# creating question-answer pairs based on the netlist control block with ngspice
python create_qa_dataset.py --analysis-file synthetic_circuits_robust_analysis_basic_only.json --output-file circuit_qa_dataset.json --max-circuit 50
```


This repository is based on [MAPS: Advancing Multi-modal Reasoning in Expert-level Physical Science](https://arxiv.org/abs/2501.10768). 