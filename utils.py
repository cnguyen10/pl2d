from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=('num_classes'))
def confusion_matrix(predictions: jax.Array, labels: jax.Array, num_classes = 2):
    """calculate the confusion matrix given predictions and ground truth labels
    adopted from: https://github.com/jax-ml/jax/discussions/10078

    Args:
        predictions:
        labels:

    Returns:
        cm: confusion matrix, e.g., (numbers as indices)
        [
            [00 (TN), 01 (FP)],
            [10 (FN), 11 (TP)]
        ]
    """
    conf_mat, _ = jax.lax.scan(
        f=lambda carry, pair: (carry.at[pair].add(1), None), 
        init=jnp.zeros(shape=(num_classes, num_classes), dtype=jnp.uint32), 
        xs=(labels, predictions)
    )

    # normalise
    conf_mat /= jnp.sum(a=conf_mat)

    return conf_mat
