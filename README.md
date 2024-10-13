# ReadMe

Small language models for running experiments.


## Usage

```
python -m llamlam.train
```

## TeuxDeux

- add no-decay vars
- tune hyperparameters
- add streaming datasets
- add gradient accumulation
- add model checkpointing
- add distributed training (zero optimization, etc.)


## Otter

```
accelerate config
accelerate launch llamlam/train.py --args
```


```
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization
```

```
torchrun \
    --nproc_per_node 1 pytorch/summarization/run_summarization.py \
    --fp16 \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

```
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --max_train_samples 50 \
    --max_eval_samples 50 \
    --max_predict_samples 50 \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --text_column text_column_name \
    --summary_column summary_column_name \
    <!-- --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \ -->
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    <!-- --overwrite_output_dir \ -->
    <!-- --output_dir previous_output_dir \  start from previous checkpoint -->
    --resume_from_checkpoint path_to_specific_checkpoint \
    --predict_with_generate
```