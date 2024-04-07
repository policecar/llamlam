# ReadMe

Small language models for running experiments.


## Usage

```
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
deepspeed --num_gpus $(nvidia-smi -L | wc -l) llamlam/train.py --batch_size 16 --learning_rate 1e-5 --run_name "test"

# ran: deepspeed --num_gpus $(nvidia-smi -L | wc -l) llamlam/train.py --learning_rate 1e-4 --head_width 32 --run_name "test"
#      with default batch_size 16
```
