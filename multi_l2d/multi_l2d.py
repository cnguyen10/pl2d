import os
from pathlib import Path
import random
from functools import partial
from collections import Counter

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import flatdict

import numpy as np

import jax
import jax.numpy as jnp

from flax.nnx import metrics
from flax.core import FrozenDict

import orbax.checkpoint as ocp

import optax

from chex import Array, Scalar

import mlflow

import mlx.data as dx

# locally-imported
from utils import (
    TrainState,
    make_dataset,
    prepare_dataset,
    initialise_huggingface_resnet
)


def augmented_labels(y: Array, t: Array, num_classes: int) -> Array:
    """augment the labels for the unified gating + classifier model

    Args:
        y: ground truth labels (batch,)
        t: expert's annotations (missing is denoted as -1) (batch, num_experts)
        num_classes:

    Return:
        y_augmented:
    """
    y_one_hot = jax.nn.one_hot(x=y, num_classes=num_classes)  # (batch, num_classes)
    
    # binary flag of expert's predictions
    y_orthogonal = (t == y[:, None]) * 1  # (batch, num_experts)

    y_augmented = jnp.concatenate(arrays=(y_one_hot, y_orthogonal), axis=-1)  # (batch, num_classes + num_experts)

    return y_augmented


@partial(jax.jit, static_argnames=('cfg',), donate_argnames=('state',))
def train_step(x: Array, y_augmented: Array, state: TrainState, cfg: DictConfig) -> tuple[TrainState, Scalar]:
    """
    Args:
        x: input samples  (batch, dim)
        y_augmented: "augmented" labels  (batch, num_classes + num_experts)
        state:

    Returns:
        state:
        loss:
    """
    def softmax_loss_fn(params: FrozenDict) -> tuple[Scalar, FrozenDict]:
        """loss in Multi_L2D with softmax
        """
        logits, batch_stats = state.apply_fn(
            variables={'params': params, 'batch_stats': state.batch_stats},
            x=x,
            train=True,
            mutable=['batch_stats']
        )

        loss = optax.losses.softmax_cross_entropy(logits=logits, labels=y_augmented)
        loss = jnp.mean(a=loss, axis=0)

        log_softmax = jax.nn.log_softmax(x=logits, axis=-1)
        log_softmax_clf = jax.nn.logsumexp(a=log_softmax[:, :cfg.dataset.num_classes], axis=-1)
        log_logits_gating = jnp.concatenate(
            arrays=(log_softmax[:, cfg.dataset.num_classes:], log_softmax_clf[:, None]),
            axis=-1
        )
        loss_prior = - jnp.sum(
            a=(jnp.array(object=cfg.hparams.Dirichlet_concentration) - 1) * log_logits_gating,
            axis=-1
        )
        loss_prior = jnp.mean(a=loss_prior, axis=0)
        
        loss = loss + (len(x) / cfg.dataset.length) * loss_prior

        return loss, batch_stats
    
    def one_vs_all(params: FrozenDict) -> tuple[Scalar, FrozenDict]:
        """Multi-L2D with OvA loss
        """
        logits, batch_stats = state.apply_fn(
            variables={'params': params, 'batch_stats': state.batch_stats},
            x=x,
            train=True,
            mutable=['batch_stats']
        )

        loss = optax.losses.sigmoid_binary_cross_entropy(
            logits=logits,
            labels=y_augmented
        )
        loss = jnp.sum(a=loss, axis=-1)
        loss = jnp.mean(a=loss, axis=0)

        return loss, batch_stats


    grad_value_fn = jax.value_and_grad(
        # fun=softmax_loss_fn,
        fun=one_vs_all,
        argnums=0,
        has_aux=True
    )
    (loss, batch_stats), grads = grad_value_fn(state.params)

    # update parameters from gradients
    state = state.apply_gradients(grads=grads)

    # update batch statistics
    state = state.replace(batch_stats=batch_stats['batch_stats'])
    # endregion

    return state, loss


def train(dataset: dx._c.Buffer, state: TrainState, cfg: DictConfig) -> tuple[TrainState, Scalar]:
    """
    """
    # batching and shuffling the dataset
    dset = prepare_dataset(
        dataset=dataset,
        shuffle=True,
        batch_size=cfg.training.batch_size,
        prefetch_size=cfg.data_loading.prefetch_size,
        num_threads=cfg.data_loading.num_threads,
        mean=cfg.hparams.mean,
        std=cfg.hparams.std,
        random_crop_size=cfg.dataset.crop_size,
        prob_random_h_flip=cfg.hparams.prob_random_h_flip
    )

    # metric to track the training loss
    loss_accum = metrics.Average()

    for samples in tqdm(
        iterable=dset,
        desc='epoch',
        ncols=80,
        total=len(dataset)//cfg.training.batch_size,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)  # input samples
        y = jnp.asarray(a=samples['ground_truth'], dtype=jnp.int32)  # true int labels  (batch,)
        t = jnp.asarray(a=samples['label'], dtype=jnp.int32)  # annotated int labels (batch, num_experts)

        # augmented labels
        y_augmented = augmented_labels(y=y, t=t, num_classes=cfg.dataset.num_classes)

        state, loss = train_step(x=x, y_augmented=y_augmented, state=state, cfg=cfg)

        if jnp.isnan(loss):
            raise ValueError('Training loss is NaN.')

        # tracking
        loss_accum.update(values=loss)

    return state, loss_accum.compute()


@jax.jit
def prediction_step(x: Array, state: TrainState) -> Array:
    """
    """
    logits, _ = state.apply_fn(
        variables={'params': state.params, 'batch_stats': state.batch_stats},
        x=x,
        train=False,
        mutable=['batch_stats']
    )

    return logits


def evaluate(dataset: dx._c.Buffer, state: TrainState, cfg: DictConfig) -> tuple[Array, Counter]:
    """
    """
    # prepare dataset for training
    dset = prepare_dataset(
        dataset=dataset,
        shuffle=False,
        batch_size=cfg.training.batch_size,
        prefetch_size=cfg.data_loading.prefetch_size,
        num_threads=cfg.data_loading.num_threads,
        mean=cfg.hparams.mean,
        std=cfg.hparams.std,
        random_crop_size=cfg.dataset.crop_size,
        prob_random_h_flip=cfg.hparams.prob_random_h_flip
    )

    accuracy_accum = metrics.Accuracy()
    coverages = Counter()

    for samples in tqdm(
        iterable=dset,
        desc='evaluate',
        total=len(dataset)//cfg.training.batch_size + 1,
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)  # input samples
        y = jnp.asarray(a=samples['ground_truth'], dtype=jnp.int32)  # true labels (batch,)
        t = jnp.asarray(a=samples['label'], dtype=jnp.int32)  # annotated labels (batch, num_experts)

        logits = prediction_step(x=x, state=state)  # (batch, num_classes + num_experts)

        # classifier predictions
        clf_predictions = jnp.argmax(a=logits[:, :cfg.dataset.num_classes], axis=-1)  # (batch,)

        labels_concatenated = jnp.concatenate(arrays=(t, clf_predictions[:, None]), axis=-1)  # (batch, num_experts + 1)

        logits_max_id = jnp.argmax(a=logits, axis=-1)  # (batch,)
        logits_max_id = logits_max_id - cfg.dataset.num_classes

        # which samples are predicted by classifier
        samples_predicted_by_clf = (logits_max_id < 0) * 1  # (batch,)

        # which samples are deferred to which experts
        sample_expert_id = logits_max_id * (1 - samples_predicted_by_clf)  # (batch,)

        selected_expert_ids = samples_predicted_by_clf * len(cfg.dataset.test_files) + sample_expert_id  # (batch,)

        coverages.update(np.asarray(a=selected_expert_ids, dtype=np.int32))

        # system's predictions
        y_predicted = labels_concatenated[jnp.arange(y.shape[0]), selected_expert_ids]

        accuracy_accum.update(logits=jax.nn.one_hot(x=y_predicted, num_classes=cfg.dataset.num_classes), labels=y)

    return (accuracy_accum.compute(), coverages)


@hydra.main(version_base=None, config_path='conf', config_name='conf')
def main(cfg: DictConfig) -> None:
    jax.config.update('jax_disable_jit', cfg.jax.disable_jit)
    jax.config.update('jax_platforms', cfg.jax.platform)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)

    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
    )

    # region DATASETS
    dset_train = make_dataset(
        annotation_files=cfg.dataset.train_files,
        ground_truth_file=cfg.dataset.train_ground_truth_file,
        root=cfg.dataset.root,
        shape=cfg.dataset.resized_shape
    )

    dset_test = make_dataset(
        annotation_files=cfg.dataset.test_files,
        ground_truth_file=cfg.dataset.test_ground_truth_file,
        root=cfg.dataset.root,
        shape=cfg.dataset.resized_shape
    )

    # store length of the dataset
    OmegaConf.set_struct(conf=cfg, value=True)
    with open_dict(config=cfg):
        cfg.dataset.length = len(dset_train)
    # endregion


    # region MODELS
    # a functools.partial wrapper of resnet
    base_model = hydra.utils.instantiate(config=cfg.model)

    # parameter of gating function
    state = initialise_huggingface_resnet(
        model=base_model(
            num_classes=cfg.dataset.num_classes + len(cfg.dataset.train_files),
            input_shape=(1,) + tuple(cfg.dataset.crop_size) + (dset_train[0]['image'].shape[-1],),
            dtype=jnp.bfloat16
        ),
        sample=jnp.expand_dims(a=dset_train[0]['image'] / 255, axis=0),
        num_training_samples=len(dset_train),
        lr=cfg.training.gating_lr,
        batch_size=cfg.training.batch_size,
        num_epochs=cfg.training.num_epochs,
        key=jax.random.key(seed=random.randint(a=0, b=10_000)),
        max_norm=cfg.hparams.clipped_norm
    )

    # options to store models
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=100,
        max_to_keep=1,
        step_format_fixed_length=3,
        enable_async_checkpointing=True
    )
    # endregion

    # region Mlflow
    mlflow.set_tracking_uri(uri=cfg.experiment.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.experiment.name)
    mlflow.disable_system_metrics_logging()
    # mlflow.set_system_metrics_sampling_interval(interval=600)
    # mlflow.set_system_metrics_samples_before_logging(samples=1)

    # create a directory for storage (if not existed)
    if not os.path.exists(path=cfg.experiment.logdir):
        Path(cfg.experiment.logdir).mkdir(parents=True, exist_ok=True)
    # endregion

    # enable mlflow tracking
    with mlflow.start_run(
        run_id=cfg.experiment.run_id,
        log_system_metrics=False
    ) as mlflow_run:
        # append run id into the artifact path
        ckpt_dir = os.path.join(
            os.getcwd(),
            cfg.experiment.logdir,
            cfg.experiment.name,
            mlflow_run.info.run_id
        )

        # enable an orbax checkpoint manager to save model's parameters
        with ocp.CheckpointManager(directory=ckpt_dir, options=ckpt_options) as ckpt_mngr:

            if cfg.experiment.run_id is None:
                # log hyper-parameters
                mlflow.log_params(
                    params=flatdict.FlatDict(
                        value=OmegaConf.to_container(cfg=cfg),
                        delimiter='.'
                    )
                )

                # log source code
                mlflow.log_artifact(
                    local_path=os.path.abspath(path=__file__),
                    artifact_path='source_code'
                )

                start_epoch_id = 0
            else:
                start_epoch_id = ckpt_mngr.latest_step()

                checkpoint = ckpt_mngr.restore(
                    step=start_epoch_id,
                    args=ocp.args.StandardRestore(item=state)
                )

                state = checkpoint.state

                del checkpoint

            for epoch_id in tqdm(
                iterable=range(start_epoch_id, cfg.training.num_epochs, 1),
                desc='progress',
                ncols=80,
                leave=True,
                position=1,
                colour='green',
                disable=not cfg.data_loading.progress_bar
            ):
                state, loss = train(
                    dataset=dset_train,
                    state=state,
                    cfg=cfg
                )
                mlflow.log_metric(
                    key='loss',
                    value=loss,
                    step=epoch_id + 1,
                    synchronous=False
                )

                accuracy, coverages = evaluate(
                    dataset=dset_test,
                    state=state,
                    cfg=cfg
                )
                
                mlflow.log_metric(key='accuracy', value=accuracy, step=epoch_id + 1, synchronous=False)

                for i in range(len(cfg.dataset.train_files) + 1):
                    mlflow.log_metric(
                        key='coverage/{:d}'.format(i),
                        value=coverages[i] / coverages.total(),
                        step=epoch_id + 1,
                        synchronous=False
                    )

                # # save checkpoint
                # ckpt_mngr.save(
                #     step=epoch_id + 1,
                #     args=ocp.args.Composite(
                #         gating_state=ocp.args.StandardSave(gating_state),
                #         theta_state=ocp.args.StandardSave(theta_state)
                #     )
                # )

                # # wait for checkpoint manager completing the asynchronous saving
                # ckpt_mngr.wait_until_finished()
    return None


if __name__ == '__main__':
    # cache jax-compiled file if the compiling takes more than 2 minutes
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 120)

    main()