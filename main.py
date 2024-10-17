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

import orbax.checkpoint as ocp

import mlflow

from utils import (
    make_dataset,
    initialise_huggingface_resnet
)

from probabilistic_l2d import (
    create_train_state_batch,
    train,
    evaluate
)


@hydra.main(version_base=None, config_path="./conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    """main procedure
    """
    # region VALIDATE ARGS
    # validate if the input files are the same as the number of experts
    assert len(cfg.dataset.train_files) == len(cfg.dataset.train_complete_files)
    assert len(cfg.dataset.train_files) == len(cfg.dataset.test_files)
    # endregion

    # region JAX ENVIRONMENT
    jax.config.update('jax_disable_jit', cfg.jax.disable_jit)
    jax.config.update('jax_platforms', cfg.jax.platform)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)

    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
    )
    # endregion

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
    # endregion

    # region MODELS
    # a functools.partial wrapper of resnet
    base_model = hydra.utils.instantiate(config=cfg.model)

    # parameter of gating function
    gating_state = initialise_huggingface_resnet(
        model=base_model(
            num_classes=len(cfg.dataset.train_files) + 1,  # add a classifier
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

    # parameter of annotators
    init_resnet_fn = partial(
        initialise_huggingface_resnet,
        model=base_model(
            num_classes=cfg.dataset.num_classes,
            input_shape=(1,) + tuple(cfg.dataset.crop_size) + (dset_train[0]['image'].shape[-1],),
            dtype=jnp.bfloat16
        ),
        sample=jnp.expand_dims(a=dset_train[0]['image'] / 255, axis=0),
        num_training_samples=len(dset_train),
        lr=cfg.training.expert_lr,
        batch_size=cfg.training.batch_size,
        num_epochs=cfg.training.num_epochs,
        max_norm=cfg.hparams.clipped_norm
    )
    init_resnet_fn_batch = jax.vmap(fun=init_resnet_fn, in_axes=0, out_axes=0)
    keys = jax.vmap(fun=jax.random.key, in_axes=0, out_axes=0)(
        jnp.array(object=[random.randint(a=0, b=1_000) for _ in range(len(cfg.dataset.train_files) + 1)])
    )
    theta_state = init_resnet_fn_batch(key=keys)
    theta_state = create_train_state_batch(state=theta_state)

    del keys
    del init_resnet_fn
    del init_resnet_fn_batch

    # options to store models
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=100,
        max_to_keep=1,
        step_format_fixed_length=3,
        enable_async_checkpointing=True
    )

    # load pretrained classifier
    state = initialise_huggingface_resnet(
        model=base_model(
            num_classes=cfg.dataset.num_classes,
            input_shape=(1,) + tuple(cfg.dataset.crop_size) + (dset_train[0]['image'].shape[-1],),
            dtype=jnp.bfloat16
        ),
        sample=jnp.expand_dims(a=dset_train[0]['image'] / 255, axis=0),
        num_training_samples=len(dset_train),
        lr=cfg.training.gating_lr,
        batch_size=cfg.training.batch_size,
        num_epochs=cfg.training.num_epochs,
        key=jax.random.key(seed=random.randint(a=0, b=10_000))
    )
    with ocp.CheckpointManager(
        directory='/sda2/cuong_code/classification/logdir/Classification/4cf654e9a8404d3ea6ac78bb37bb9b85/',
        # item_names=('gating_state', 'theta_state'),
        # directory='/sda2/cuong_code/pl2d/logdir/PL2D/79c84177cc2b40aea67004e165e1c62a',
        options=ckpt_options
    ) as ckpt_mngr:
        # checkpoint = ckpt_mngr.restore(
        #     step=1000,
        #     args=ocp.args.Composite(
        #         gating_state=ocp.args.StandardRestore(
        #             item=gating_state
        #         ),
        #         theta_state=ocp.args.StandardRestore(
        #             item=theta_state
        #         )
        #     )
        # )

        # # gating_state = checkpoint.gating_state
        # theta_state = checkpoint.theta_state

        # del checkpoint
        state = ckpt_mngr.restore(step=1000, args=ocp.args.StandardRestore())

    temp = jax.tree.map(lambda x, y: x + jnp.zeros_like(a=y), state['params'], theta_state.params)
    theta_state = theta_state.replace(params=temp)

    temp = jax.tree.map(lambda x, y: x + jnp.zeros_like(a=y), state['batch_stats'], theta_state.batch_stats)
    theta_state = theta_state.replace(batch_stats=temp)

    del temp
    del state
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
            item_names=('gating_state', 'theta_state'),
            options=ckpt_options
        ) as ckpt_mngr:

            if cfg.experiment.run_id is None:
                start_epoch_id = 0

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
            else:
                start_epoch_id = ckpt_mngr.latest_step()

                checkpoint = ckpt_mngr.restore(
                    step=start_epoch_id,
                    args=ocp.args.Composite(
                        gating_state=ocp.args.StandardRestore(
                            item=gating_state
                        ),
                        theta_state=ocp.args.StandardRestore(
                            item=theta_state
                        )
                    )
                )

                gating_state = checkpoint.gating_state
                theta_state = checkpoint.theta_state

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
                    dataset=dset_train,
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
                        gating_state=ocp.args.StandardSave(gating_state),
                        theta_state=ocp.args.StandardSave(theta_state)
                    )
                )

                accuracy, expert_accuracies, coverages, p_z, ece, conf_mat = evaluate(
                    dataset=dset_test,
                    gating_state=gating_state,
                    theta_state=theta_state,
                    cfg=cfg
                )

                metric_dict = {
                    'loss': loss,
                    'accuracy': accuracy,
                    'coverage': coverages[len(cfg.dataset.train_files)] / coverages.total(),
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
