import jax
import optax

import random

import grain.python as grain

from transformations import (
    RandomCrop,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
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
    clipped_norm: float
) -> optax.GradientTransformationExtraArgs:
    """initialize parameters of an optimizer
    """
    # add L2 regularization(aka weight decay)
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
        optax.add_noise(eta=0.01, gamma=0.55, seed=random.randint(a=0, b=100)),
        optax.clip_by_global_norm(max_norm=clipped_norm) \
            if clipped_norm is not None else optax.identity(),
        optax.sgd(learning_rate=lr_schedule_fn, momentum=momentum)
    )

    return tx


def initialize_dataloader(
    data_source: grain.RandomAccessDataSource,
    num_epochs: int,
    shuffle: bool,
    seed: int,
    batch_size: int,
    crop_size: tuple[int, int] = None,
    resize: tuple[int, int] = None,
    mean: float = None,
    std: float = None,
    p_flip: float = None,
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

    if crop_size is not None:
        transformations.append(RandomCrop(crop_size=crop_size))

    if p_flip is not None:
        transformations.append(RandomHorizontalFlip(p=p_flip))
        transformations.append(RandomVerticalFlip(p=p_flip))

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