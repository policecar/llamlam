import numpy as np
import torch
import random


def set_seed(seed: int = 137):
    """Set the seed for the random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, accelerator=None):
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
    model.eval()

    losses = []
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(batch["input_ids"])
        if accelerator is not None:
            losses.append(accelerator.gather(outputs.loss))
        else:
            losses.append(outputs.loss)
    loss = torch.mean(torch.cat(losses))

    try:
        perplexity = torch.exp(loss).item()
    except OverflowError:
        perplexity = float("inf")

    model.train()

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
            #     group_parameters["lr"] = learning_rate * (1 / head_width)

            group_parameters["params"] = [p]
            opt_grouped_params.append(group_parameters)

    return opt_grouped_params
