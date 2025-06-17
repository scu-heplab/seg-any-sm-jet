# __<p align=center>Segment Any SM-Jet</p>__

__<p align=center>Detecting Htt event</p>__
<div align=center>
   <figure>
      <img src="./result/jet.png" alt="htt"/>
   </figure>
</div>

Paper: arxiv.xxxx 

## Requirements
* Only Linux is supported.
* 64-bit Python3.10(or higher, recommend 3.10) installation.
* One or more high-end NVIDIA GPUs(at least 24 GB of DRAM).
* Pytorch2.4+cuda11.8, causal-conv1d==1.14.0, mamba-ssm==2.2.2, pulp==3.1.1.

## Environment Configuration
* Mamba install: https://github.com/state-spaces/mamba
* VMamba install: https://github.com/MzeroMiko/VMamba

## Preparing Dataset
* ### Compile the C++ Tool  
The code is in the `src/cpp` and includes a `Makefile` to simplify compilation.
```bash
cd src/cpp
# IMPORTANT: Replace /path/to/your/delphes with your actual path!
make DELPHES=/path/to/your/delphes
```
After running, a new executable file named `process` will be created in the `src/cpp`.

* ### Run the Tool on a Single File  
The `process` tool processes one file at a time. The command format is:
```bash
./process --input <path_to_your_input.root> --output <path_for_your_output.dat> --pids "<pid_list>"
```
`<pid_list>` specifies the mother particles for which final state assignment will be performed, e.g. "25, 6, -6, 5, -5, 24, -24", light-quark/gluon will be add automatically.
* ### Process Multiple Files
We provide a shell script (`run.sh`) in `src/cpp` that can perform multi-file processing with only minor modifications.
```bash
chmod +x run.sh
./run.sh
```
There are two root files for testing in the `src/cpp/input`, namely `htt.root` and `tttt.root`.

## Training
Our code supports distributed training and checkpointing.
