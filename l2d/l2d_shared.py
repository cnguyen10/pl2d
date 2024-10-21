import os
from pathlib import Path
import random
from functools import partial

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import flatdict

import jax
import jax.numpy as jnp

from flax.nnx import metrics
from flax.core import FrozenDict

import orbax.checkpoint as ocp

import optax

from tensorflow_probability.substrates import jax as tfp

from chex import Array, Scalar

import mlflow

import mlx.data as dx

from utils import (
    TrainState,
    make_dataset,
    prepare_dataset,
    initialise_huggingface_resnet
)
from probabilistic_l2d import constrained_posterior


@partial(jax.jit, static_argnames=('cfg',), donate_argnames=('state',))
def train_step(
    x: Array,
    y: Array,
    t: Array,
    state: TrainState,
    cfg: DictConfig
) -> tuple[TrainState, TrainState, Scalar]:
    """
    """
    t_one_hot = jax.nn.one_hot(x=t, num_classes=cfg.dataset.num_classes)
    t_one_hot = optax.smooth_labels(labels=t_one_hot, alpha=0.01)

    def loss_function(params: FrozenDict,) -> tuple[Scalar, FrozenDict]:
        """
        """
        # prediction of classifier Pr(t | x, theta)
        logits, batch_stats = state.apply_fn(
            variables={'params': params, 'batch_stats': state.batch_stats},
            x=x,
            train=True,
            mutable=['batch_stats']
        )
        logits_clf = logits[:, :cfg.dataset.num_classes]
        logits_gating = logits[:, cfg.dataset.num_classes:]

        p_t_x_clf = jax.nn.softmax(x=logits_clf, axis=-1)  # (batch, num_classes)

        # prediction of all experts
        p_t_x = jnp.concatenate(
            arrays=(t_one_hot, p_t_x_clf[:, None, :]),
            axis=1
        )  # (batch, num_experts + 1, num_classes)

        # prediction of gating model
        log_p_z_x_gamma = jax.nn.log_softmax(x=logits_gating, axis=-1)

        # region E-step: Pr(z | x, y, t, gamma)
        # calculate Pr(y | x, t)
        distr_y_t = tfp.distributions.Categorical(probs=p_t_x)
        log_p_y_t = distr_y_t.log_prob(value=y[:, None])  # (batch, num_experts + 1)

        # posterior
        log_q_z = jax.lax.stop_gradient(x=log_p_y_t + log_p_z_x_gamma)
        log_q_z -= jax.nn.logsumexp(a=log_q_z, axis=-1, keepdims=True)  # normalisation

        # constrained posterior
        q_z = constrained_posterior(
            q_z_uncon=jnp.exp(log_q_z),
            epsilon_upper=jnp.array(object=cfg.hparams.epsilon_upper, dtype=jnp.float32),
            epsilon_lower=jnp.array(object=cfg.hparams.epsilon_lower, dtype=jnp.float32)
        )
        # endregion

        loss = optax.losses.softmax_cross_entropy_with_integer_labels(
            logits=logits_clf,
            labels=y
        )
        loss = loss - jnp.sum(a=q_z * log_p_z_x_gamma, axis=-1)
        loss = jnp.mean(a=loss, axis=0)

        return loss, batch_stats

    grad_value_fn = jax.value_and_grad(fun=loss_function, argnums=0, has_aux=True)
    (loss, batch_stats), grads = grad_value_fn(state.params)

    # update parameters from gradients
    state = state.apply_gradients(grads=grads)

    # update batch statistics
    state = state.replace(batch_stats=batch_stats['batch_stats'])
    # endregion

    # return grads_gating, batch_stats_gating, grads_theta, batch_stats_theta, loss
    return state, loss


def train(dataset: dx._c.Buffer, state: TrainState, cfg: DictConfig) -> tuple[TrainState, Scalar]:
    """the main training procedure
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
        desc='train',
        total=len(dataset) // cfg.training.batch_size + 1,
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)  # input samples
        y = jnp.asarray(a=samples['ground_truth'], dtype=jnp.int32)  # true int labels  (batch,)
        t = jnp.asarray(a=samples['label'], dtype=jnp.int32)  # annotated int labels (batch, num_experts)

        state, loss = train_step(
            x=x,
            y=y,
            t=t,
            state=state,
            cfg=cfg
        )

        if jnp.isnan(loss):
            raise ValueError('Training loss is NaN.')

        # tracking
        loss_accum.update(values=loss)

    return state, loss_accum.compute()


@jax.jit
def prediction_step(x: Array, state: TrainState) -> Array:
    """
    """
    logits_p_z, _ = state.apply_fn(
        variables={'params': state.params, 'batch_stats': state.batch_stats},
        x=x,
        train=False,
        mutable=['batch_stats']
    )  # (batch, num_experts + 1)

    return logits_p_z


def evaluate(
    dataset: dx._c.Buffer,
    state: TrainState,
    cfg: DictConfig
) -> tuple[Scalar, Scalar, list[Array], Scalar]:
    """calculate the average cluster probability vector

    Args:
        dataset:
        state:
        cfg:
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

    p_z_accum = [metrics.Average() for _ in range(len(cfg.dataset.train_files) + 1)]
    accuracy_accum = metrics.Accuracy()
    coverage = metrics.Average()
    clf_accuracy_accum = metrics.Accuracy()

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

        t = jax.nn.one_hot(x=t, num_classes=cfg.dataset.num_classes)  # (batch, num_experts, num_classes)

        # Pr(z | x, gamma)
        logits = prediction_step(x=x, state=state)
        logits_p_z = logits[:, cfg.dataset.num_classes:]

        # tracking ln Pr(z | x, gamma)
        log_p_z = jax.nn.log_softmax(x=logits_p_z, axis=-1)
        
        for i in range(len(cfg.dataset.train_files) + 1):
            p_z_accum[i].update(values=jnp.exp(log_p_z[:, i]))

        selected_expert_ids = jnp.argmax(a=logits_p_z, axis=-1)  # (batch,)

        coverage.update(values=(selected_expert_ids == len(cfg.dataset.test_files)) * 1)

        # accuracy
        logits_p_y_x_theta = logits[:, :cfg.dataset.num_classes]
        human_and_model_predictions = jnp.concatenate(
            arrays=(t, logits_p_y_x_theta[:, None, :]),
            axis=1
        )  # (batch, num_experts + 1, num_classes)
        queried_predictions = human_and_model_predictions[jnp.arange(len(x)), selected_expert_ids, :]
        accuracy_accum.update(logits=queried_predictions, labels=y)
        clf_accuracy_accum.update(logits=logits_p_y_x_theta, labels=y)

    return (
        accuracy_accum.compute(),
        coverage.compute(),
        [p_z.compute() for p_z in p_z_accum],
        clf_accuracy_accum.compute()
    )


@hydra.main(version_base=None, config_path='conf', config_name='conf')
def main(cfg: DictConfig) -> None:
    jax.config.update('jax_disable_jit', cfg.jax.disable_jit)
    jax.config.update('jax_platforms', cfg.jax.platform)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)

    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
    )

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
    
    # region MODELS
    # a functools.partial wrapper of resnet
    base_model = hydra.utils.instantiate(config=cfg.model)

    # parameter of gating function
    state = initialise_huggingface_resnet(
        model=base_model(
            num_classes=len(cfg.dataset.train_files) + cfg.dataset.num_classes + 1,
            input_shape=(1,) + tuple(cfg.dataset.crop_size) + (dset_train[0]['image'].shape[-1],),
            dtype=jnp.float32
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
        with ocp.CheckpointManager(
                directory=ckpt_dir,
                options=ckpt_options) as ckpt_mngr:

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

                accuracy, coverage, p_z, clf_accuracy = evaluate(
                    dataset=dset_test,
                    state=state,
                    cfg=cfg
                )

                # wait for the previous checkpoint manager saving to complete
                ckpt_mngr.wait_until_finished()

                # ckpt_mngr.save(
                #     step=epoch_id + 1,
                #     args=ocp.args.Composite(
                #         gating_state=ocp.args.StandardSave(gating_state),
                #         theta_state=ocp.args.StandardSave(theta_state)
                #     )
                # )

                metric_dict = dict(
                    loss=loss,
                    accuracy=accuracy,
                    coverage=coverage,
                    clf_accuracy=clf_accuracy
                )

                for i in range(len(p_z)):
                    metric_dict['p_z_test/{:d}'.format(i)]=p_z[i]
                
                mlflow.log_metrics(
                    metrics=metric_dict,
                    step=epoch_id + 1,
                    synchronous=False
                )
    return None


if __name__ == '__main__':
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 120)

    main()
