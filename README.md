
# Quickstart

## 1. Setup

### Create environment

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 11.7 and CUDA 11.8) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.7:

```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

For CUDA 11.8:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

See [Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)
in the Installation documentation for more.

### Installing nerfstudio with GS

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
git checkout gaussian-splatting
pip install -e .
```

## 2. Training your first model!

The following will train a _gs_ model, our recommended model for real world scenes.

### training GS model 
```bash
# Download some test data:
# ns-download-data nerfstudio --capture-name=poster
# Train model
s-train gaussian-splatting --experiment-name <exp_name> colmap --colmap_path sparse/0  --data <data_path>
```
### viweing GS model 

```bash 
ns-viewer --load-config <.../config.yml> 
```


### Installation Issues
If you recieve an assertion error like the following: 
```
... /nerfstudio/lib/python3.8/site-packages/torch/autograd/function.py", line 506, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/opt/conda/envs/nerfstudio/lib/python3.8/site-packages/gsplat/sh.py", line 39, in forward
    assert coeffs.shape
```

solution reinstall gsplat from source:

```bash
pip install git+https://github.com/nerfstudio-project/gsplat
```