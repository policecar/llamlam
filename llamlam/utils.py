import numpy as np
import torch
import random


def set_seed(seed: int = 137):
    """Set the seed for the random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device=None, accelerator=None):
    """
    Evaluate the model on the given dataloader.

    Assumes that the model computes its own labels from the inputs
    and returns its loss.

    Args:
        model: The model to evaluate.
        dataloader: The dataloader to evaluate the model on.
        accelerator: Optional; The accelerator to use. Defaults to None.

    Returns:
        A tuple containing the loss and the perplexity.
    """
    losses = []

    model.eval()
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]  # .to(device)
        with torch.no_grad():
            outputs = model(input_ids)
        loss = outputs["loss"]
        if accelerator is not None:
            losses.append(accelerator.gather(loss.unsqueeze(0)))
        else:
            losses.append(loss.unsqueeze(0))
    loss = torch.mean(torch.cat(losses))
    model.train()

    try:
        perplexity = torch.exp(loss).item()
    except OverflowError:
        perplexity = float("inf")

    return loss.item(), perplexity


def get_grouped_params(model, weight_decay=0.1, no_decay=[]):
    """
    Get grouped parameters for the optimizer.

    This function groups model parameters for optimization, applying different
    hyperparameters based on parameter type. E.g., embedding parameters may
    require different learning rates, biases might be excluded from weight decay.

    This approach allows for fine-grained control over parameter optimization.

    Args:
        model: The model to get the grouped parameters for.
        weight_decay: The weight decay to use. Defaults to 0.1.
        no_decay: The parameters to not apply weight decay to.

    Returns:
        A list of dictionaries containing the grouped parameters.
    """
    opt_grouped_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            group_parameters = {}

            # Define weight decay for all params except those in no_decay
            if any(nd in n for nd in no_decay):
                group_parameters["weight_decay"] = 0.0
            else:
                group_parameters["weight_decay"] = weight_decay

            # # Define custom learning rate for specific types of params
            # is_embed = "embed" in n
            # if "embed" in n or any(nd in n for nd in no_decay):
            #     group_parameters["lr"] = learning_rate * (3.3 if is_embed else 1.0)
            # else:
            #     group_parameters["lr"] = learning_rate * (1 / dim_head)

            group_parameters["params"] = [p]
            opt_grouped_params.append(group_parameters)

    return opt_grouped_params


def save_checkpoint(
    model, optimizer, config, global_step, val_loss, tag, output_dir, max_ckpts=3
):
    """Save checkpoint to output directory."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "iter_num": global_step,
        "best_val_loss": val_loss,
    }
    torch.save(checkpoint, output_dir / f"ckpt_{tag}.pt")
