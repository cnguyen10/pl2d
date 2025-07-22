import jax
import jax.numpy as jnp

import optax

import grain.python as grain

from functools import partial

from transformations import (
    RandomCrop,
    Resize,
    CropAndPad,
    RandomHorizontalFlip,
    ToRGB,
    Normalize,
    ToFloat
)


def init_tx(
    dataset_length: int,
    lr: float,
    batch_size: int,
    num_epochs: int,
    weight_decay: float,
    momentum: float,
    clipped_norm: float,
    seed: int
) -> optax.GradientTransformationExtraArgs:
    """initialize parameters of an optimizer
    """
    # add L2 regularization(a.k.a. weight decay)
    l2_regularization = optax.masked(
        inner=optax.add_decayed_weights(
            weight_decay=weight_decay,
            mask=None
        ),
        mask=lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
    )

    num_iters_per_epoch = dataset_length // batch_size
    lr_schedule_fn = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=(num_epochs + 10) * num_iters_per_epoch
    )

    # define an optimizer
    tx = optax.chain(
        l2_regularization,
        optax.clip_by_global_norm(max_norm=clipped_norm) \
            if clipped_norm is not None else optax.identity(),
        optax.add_noise(eta=0.01, gamma=0.55, seed=seed),
        optax.sgd(learning_rate=lr_schedule_fn, momentum=momentum)
    )

    return tx


def initialize_dataloader(
    data_source: grain.RandomAccessDataSource,
    num_epochs: int,
    shuffle: bool,
    seed: int,
    batch_size: int,
    crop_size: tuple[int, int] | None = None,
    padding_px: int | list[int] | None = None,
    resize: tuple[int, int] | None = None,
    mean: float | None = None,
    std: float | None = None,
    p_flip: float | None = None,
    is_color_img: bool = True,
    num_workers: int = 0,
    num_threads: int = 1,
    prefetch_size: int = 1
) -> grain.DataLoader:
    """
    """
    index_sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=num_epochs,
        shuffle=shuffle,
        shard_options=grain.NoSharding(),
        seed=seed  # set the random seed
    )

    transformations = []

    if resize is not None:
        transformations.append(Resize(resize_shape=resize))
    
    if padding_px is not None:
        transformations.append(CropAndPad(px=padding_px))

    if crop_size is not None:
        transformations.append(RandomCrop(crop_size=crop_size))

    if p_flip is not None:
        transformations.append(RandomHorizontalFlip(p=p_flip))
        # transformations.append(RandomVerticalFlip(p=p_flip))

    if not is_color_img:
        transformations.append(ToRGB())

    transformations.append(ToFloat())

    if mean is not None and std is not None:
        transformations.append(Normalize(mean=mean, std=std))

    transformations.append(
        grain.Batch(
            batch_size=batch_size,
            drop_remainder=shuffle
        )
    )

    data_loader = grain.DataLoader(
        data_source=data_source,
        sampler=index_sampler,
        operations=transformations,
        worker_count=num_workers,
        shard_options=grain.NoSharding(),
        read_options=grain.ReadOptions(
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_size
        )
    )

    return data_loader


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