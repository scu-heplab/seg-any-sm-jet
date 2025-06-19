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
* One or more high-end NVIDIA GPUs(at least 24 GB of VRAM).
* Pytorch2.4+cuda11.8, causal-conv1d==1.14.0, mamba-ssm==2.2.2, pulp==3.1.1.

## Environment Configuration
* Mamba install: https://github.com/state-spaces/mamba
* VMamba install: https://github.com/MzeroMiko/VMamba

## Preparing Dataset
* ### Step 1: Compile the C++ Tool  
The code is in the `src/cpp` and includes a `Makefile` to simplify compilation.
```bash
cd src/cpp
# IMPORTANT: Replace /path/to/your/delphes with your actual path!
make DELPHES=/path/to/your/delphes
```
After running, a new executable file named `sajm` will be created in the `src/cpp`.

* ### Step 2: Run the Tool on a Single File  
The `sajm` tool processes one file at a time. The command format is:
```bash
./sajm --input <path_to_your_input.root> --output <path_for_your_output.dat> --pids "<pid_list>"
```
`<pid_list>` specifies the mother particles for which final state assignment will be performed, e.g. "25, 6, -6, 5, -5, 24, -24", light-quark/gluon will be add automatically.

* ### (Option) Process Multiple Files
We provide a shell script (`run.sh`) in `src/cpp` that can perform multi-file processing with only minor modifications.
```bash
chmod +x run.sh
./run.sh
```
There are two root files for testing in the `src/cpp/input`, namely `htt.root` and `tttt.root`.

* ### Step 3: Assign
Switch to `./src/python`, use the `transform.py` script to perform final state particle assignment. It converts the `.dat` files from the C++ tool into the final `event.npy` format required for training.
The script operates on a per-directory basis and has two commands: `transform` and `merge`.  
#### 1) `transform`: Convert `.dat` to chunked `.npy` files
This command processes all `.dat` files within a specified directory and saves the results into smaller `.npy` chunks.
```bash
python transform.py transform --input-dir <your_data_directory> [options]
```
Example:
```bash
# Process data in './data' and save chunks in the same directory
python transform.py transform --input-dir ./data

# Specify a separate output directory
python transform.py transform --input-dir ./data --output-dir ./temp

# Use the solver with more workers
python transform.py transform --input-dir ./data --workers 32
```
#### 2) `merge`: Combine chunks into a single `event.npy`
After transformation, this command merges all the generated chunks into a single file.
```bash
python transform.py merge --input-dir <your_data_directory> [options]
```
Example:
```bash
# Merge chunks in './data' into './data/event.npy'
python transform.py merge --input-dir ./data

# Merge chunks from a temp directory to a final destination file
python transform.py merge --input-dir ./temp --output-dir ./dataset
```

## Training
Our code supports distributed training and checkpointing. Use the following command to start training the model:
```bash
torchrun --master_addr <rank_0_ip_address> --master_port <port> --nproc_per_node <gpu_num_on_this_machine> --nnodes <machine_num> --node_rank <rank> train.py
```
Example:
```bash
# [1] machine, [4] GPUs, local training.
torchrun --master_addr 127.0.0.1 --master_port 12547 --nproc_per_node 4 --nnodes 1 --node_rank 0 train.py

# [2] machines, [2, 4] GPUs, distributed training.
# machine 1:
torchrun --master_addr 192.168.1.100 --master_port 12547 --nproc_per_node 2 --nnodes 2 --node_rank 0 train.py
# machine 2:
torchrun --master_addr 192.168.1.100 --master_port 12547 --nproc_per_node 4 --nnodes 2 --node_rank 1 train.py
```
___Notice: For distributed training, edit `init()` in `train.py` and change `os.environ["NCCL_SOCKET_IFNAME"] = "lo"`. You must replace `"lo"` with the actual network interface name for each specific machine.___

## Validate
We provide `validate.py` to test the performance of the model. Use the following commands to test:
```bash
python validate.py --test-dir ./test --output-path ./test/test.pkl --batch-size 20 --pretrain ./weight/state.pth
```
After the test is completed, you can use `plot.py` to visualize the relevant test results.

## Inference
We have not developed a dedicated script for inference. However, a Jupyter Notebook is provided that demonstrates the process and can be used as a reference.
