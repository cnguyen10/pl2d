import os
from pathlib import Path
import random

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

import jax
import jax.numpy as jnp

from flax import nnx
from flax.traverse_util import flatten_dict

import optax

import orbax.checkpoint as ocp

import mlflow

from probabilistic_l2d import train, evaluate
from DataSource import ImageDataSource
from utils import init_tx


@hydra.main(version_base=None, config_path="./conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    """main procedure
    """
    # region JAX ENVIRONMENT
    jax.config.update('jax_disable_jit', cfg.jax.disable_jit)
    jax.config.update('jax_platforms', cfg.jax.platform)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)

    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
    )
    # endregion

    # region VALIDATE ARGS
    # validate if the input files are the same as the number of experts
    assert len(cfg.dataset.train_files) == len(cfg.dataset.train_complete_files)
    assert len(cfg.dataset.train_files) == len(cfg.dataset.test_files)
    # endregion

    # region DATASETS
    data_source_train = ImageDataSource(
        annotation_files=cfg.dataset.train_files,
        ground_truth_file=cfg.dataset.train_ground_truth_file,
        root=cfg.dataset.root
    )
    data_source_test = ImageDataSource(
        annotation_files=cfg.dataset.test_files,
        ground_truth_file=cfg.dataset.test_ground_truth_file,
        root=cfg.dataset.root
    )
    # endregion

    # region MODELS
    # a functools.partial wrapper of resnet
    model_fn = hydra.utils.instantiate(config=cfg.model)  # function to instanitate

    # parameter of gating function
    gating_model = model_fn(
        num_classes=len(cfg.dataset.train_files) + 1,  # add a classifier
        rngs=nnx.Rngs(jax.random.PRNGKey(seed=random.randint(a=0, b=100))),
        dtype=eval(cfg.jax.dtype)
    )
    gating_state = nnx.Optimizer(
        model=gating_model,
        tx=init_tx(
            dataset_length=len(data_source_train),
            lr=cfg.training.gating_lr,
            batch_size=cfg.training.batch_size,
            num_epochs=cfg.training.num_epochs,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.training.momentum,
            clipped_norm=cfg.hparams.clipped_norm
        )
    )

    del gating_model

    # vmap to create an ensemble of models modelling annotators
    @nnx.vmap(in_axes=0, out_axes=0)
    def create_expert_model(key: jax.random.PRNGKey) -> nnx.Module:
        return model_fn(
            num_classes=cfg.dataset.num_classes,
            rngs=nnx.Rngs(key),
            dtype=eval(cfg.jax.dtype)
        )

    keys = jax.random.split(
        key=jax.random.PRNGKey(seed=random.randint(a=0, b=100)),
        num=len(cfg.dataset.train_files) + 1
    )
    theta_model = create_expert_model(keys)
    theta_state = nnx.Optimizer(
        model=theta_model,
        tx=init_tx(
            dataset_length=len(data_source_train),
            lr=cfg.training.expert_lr,
            batch_size=cfg.training.batch_size,
            num_epochs=cfg.training.num_epochs,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.training.momentum,
            clipped_norm=cfg.hparams.clipped_norm
        )
    )

    del theta_model

    # options to store models
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=100,
        max_to_keep=1,
        step_format_fixed_length=3,
        enable_async_checkpointing=True
    )
    # endregion

    mlflow.set_tracking_uri(uri=cfg.experiment.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.experiment.name)
    mlflow.disable_system_metrics_logging()
    # mlflow.set_system_metrics_sampling_interval(interval=600)
    # mlflow.set_system_metrics_samples_before_logging(samples=1)

    # create a directory for storage (if not existed)
    if not os.path.exists(path=cfg.experiment.logdir):
        Path(cfg.experiment.logdir).mkdir(parents=True, exist_ok=True)

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
            item_names=('gating', 'theta'),
            options=ckpt_options
        ) as ckpt_mngr:

            if cfg.experiment.run_id is None:
                start_epoch_id = 0

                # log hyper-parameters
                mlflow.log_params(
                    params=flatten_dict(xs=OmegaConf.to_container(cfg=cfg), sep='.')
                )

                # log source code
                mlflow.log_artifact(
                    local_path=os.path.abspath(path=__file__),
                    artifact_path='source_code'
                )
            else:
                start_epoch_id = ckpt_mngr.latest_step()

                checkpoint = ckpt_mngr.restore(
                    step=start_epoch_id,
                    args=ocp.args.Composite(
                        gating=ocp.args.StandardRestore(item=gating_state.model),
                        theta=ocp.args.StandardRestore(item=theta_state.model)
                    )
                )

                gating_state = nnx.update(gating_state.model, checkpoint.gating)
                theta_state = nnx.update(theta_state.model, checkpoint.theta)

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
                gating_state, theta_state, loss, p_z_train = train(
                    data_source=data_source_train,
                    gating_state=gating_state,
                    theta_state=theta_state,
                    cfg=cfg
                )

                # wait until completing the asynchronous saving
                ckpt_mngr.wait_until_finished()

                # save parameters asynchronously
                ckpt_mngr.save(
                    step=epoch_id + 1,
                    args=ocp.args.Composite(
                        gating=ocp.args.StandardSave(nnx.state(gating_state.model)),
                        theta=ocp.args.StandardSave(nnx.state(theta_state.model))
                    )
                )

                accuracy, expert_accuracies, coverage, p_z, ece, conf_mat = evaluate(
                    data_source=data_source_test,
                    gating_state=gating_state,
                    theta_state=theta_state,
                    cfg=cfg
                )

                metric_dict = {
                    'loss': loss,
                    'accuracy': accuracy,
                    'coverage': coverage,
                    'ece': ece,
                    'ConfusionMat/TN': conf_mat[0, 0],
                    'ConfusionMat/FP': conf_mat[0, 1],
                    'ConfusionMat/FN': conf_mat[1, 0],
                    'ConfusionMat/TP': conf_mat[1, 1]
                }

                for i in range(len(expert_accuracies)):
                    metric_dict[f'Expert_accuracy/{i}'] = expert_accuracies[i]
                    metric_dict[f'p_z_test/{i}'] = p_z[i]
                    metric_dict[f'p_z_train/{i}'] = p_z_train[i]

                mlflow.log_metrics(
                    metrics=metric_dict,
                    step=epoch_id + 1,
                    synchronous=False
                )

    return None


if __name__ == '__main__':
    # cache Jax compilation to reduce compilation time in next runs
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 120)

    main()
