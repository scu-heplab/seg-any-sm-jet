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
* One or more high-end NVIDIA GPUs(at least 24 GB of DRAM).
* 64-bit Python3.10(or higher, recommend 3.10) installation.
* Pytorch2.4+cuda11.8, causal-conv1d==1.14.0, mamba-ssm==2.2.2

## Environment Configuration
* Mamba install: https://github.com/state-spaces/mamba
* VMamba install: https://github.com/MzeroMiko/VMamba

## Preparing Dataset
We provide a C++ code to convert the output of Madgraph+Delphes into the format required for model training.
___Notice:(the converted file can also be used directly for inference, but if only the inference function is used, a more convenient method is to directly convert the Delphes root file into a [H, W, 6] tensor input).
