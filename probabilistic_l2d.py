from functools import partial
import logging

import jax
import jax.numpy as jnp

from flax import nnx

import optax

from tensorflow_probability.substrates import jax as tfp

from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative

import grain.python as grain

from omegaconf import DictConfig

from tqdm import tqdm

from utils import confusion_matrix
from transformations import (
    Resize,
    RandomCrop,
    RandomHorizontalFlip,
    ToFloat,
    Normalize
)


@jax.jit
def get_unnorm_log_q_z_tilde(q_uncon: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    """calculate the un-normalised q(z)

    Args:
        q_uncon: the posterior of z without any constraints
        params: a tree-like or dictionary containing the Lagrange multipliers
            - upper:
            - lower:

    Return:
        log_q: the logarithm of the un-normalised posterior
    """
    # calculate log_q_tilde
    log_q_den = params['upper'] - params['lower'] + 1
    log_q = jnp.log(q_uncon) - log_q_den  # (batch, num_experts + 1)

    return log_q


@jax.jit
def constrained_posterior(
    q_z_uncon: jax.Array,
    epsilon_upper: jax.Array,
    epsilon_lower: jax.Array
) -> jax.Array:
    """calculate the work-load balancing posterior

    Args:
        log_q_z_unconstrained: the unconstrained posterior of z
        epsilon_upper: the hyperparameter for upper constraint
        epsilon_lower: the hyperparameter for lower constraint

    Returns:
        log_q_z: the constrained posterior of z
    """
    def duality_Lagrangian(params: dict[str, jax.Array]) -> jax.Array:
        """the duality of Lagrangian to find the Lagrange multiplier

        Args:
            lmbd: a dictionary containing the Lagrange multiplier with the following keys:
                - upper: corresponds to epsilon upper  # (num_experts + 1,)
                - lower: epsilon lower  # (num_experts + 1,)
                - ij: epsilon ij  # (batch, num_experts + 1)

        Returns:
            lagrangian:
        """
        log_q_tilde = get_unnorm_log_q_z_tilde(q_uncon=q_z_uncon, params=params)

        # calculate Lagrangian
        lgr = jax.nn.logsumexp(a=log_q_tilde, axis=-1)
        lgr = jnp.mean(a=lgr, axis=0)

        lgr = lgr + jnp.sum(a=params['upper'] * epsilon_upper, axis=0)
        lgr = lgr - jnp.sum(a=params['lower'] * epsilon_lower, axis=0)

        return lgr

    init_params = dict(
        upper=jnp.zeros_like(a=epsilon_upper, dtype=jnp.float32),
        lower=jnp.zeros_like(a=epsilon_lower, dtype=jnp.float32)
    )

    pg = ProjectedGradient(fun=duality_Lagrangian, projection=projection_non_negative)
    res = pg.run(init_params=init_params)

    log_q_z = get_unnorm_log_q_z_tilde(q_uncon=q_z_uncon, params=res.params)

    # normalisation
    log_q_z -= jax.nn.logsumexp(a=log_q_z, axis=-1, keepdims=True)
    q_z = jnp.exp(log_q_z)

    return q_z


@partial(jax.jit, static_argnames=('num_classes', 'num_experts', 'num_iterations'))
def unconstrained_posteriors(
    log_p_z_x: jax.Array,  # (batch, num_experts + 1)
    log_p_t_x: jax.Array,  # (batch, num_experts + 1, num_classes)
    y: jax.Array,  # (batch,)
    t: jax.Array,  # (batch, num_experts)
    num_classes: int,
    num_experts: int,
    num_iterations: int
) -> tuple[jax.Array, jax.Array]:
    """
    """
    missing_vec = jnp.array(object=(t == -1), dtype=jnp.int32)  # (batch, num_experts)

    # convert annotations to one-hot vectors
    t_one_hot = jax.nn.one_hot(x=t, num_classes=num_classes)  # (batch, num_experts, num_classes)
    t_one_hot = optax.smooth_labels(labels=t_one_hot, alpha=1e-3)  # (batch, num_experts, num_classes)

    # append the classifier's prediction into the experts' annotations
    missing_vec = jnp.concatenate(
        arrays=(missing_vec, jnp.expand_dims(a=jnp.zeros_like(a=y, dtype=jnp.int32), axis=-1)),
        axis=1
    )  # (batch, num_experts + 1,)
    p_t_x_clf = jax.nn.softmax(x=log_p_t_x[:, -1, :], axis=-1)  # (batch, num_classes)
    t_one_hot = jnp.concatenate(
        arrays=(t_one_hot, p_t_x_clf[:, None, :]),
        axis=1
    )  # (batch, num_experts + 1, num_classes)

    # ln Pr(y | x, t) - with all values of t
    t_all_one_hot = jnp.eye(N=num_classes)  # (num_classes, num_classes)
    t_all_one_hot = optax.smooth_labels(labels=t_all_one_hot, alpha=1e-3)
    distr_y_given_t = tfp.distributions.Categorical(probs=t_all_one_hot)
    log_p_y_t = jax.vmap(fun=distr_y_given_t.log_prob, in_axes=0)(y)  # (batch, num_classes,)

    # region POSTERIORS
    # 1. initialise posteriors
    q_t = t_one_hot  # (batch, num_experts + 1, num_classes)
    q_z = jnp.ones_like(a=log_p_z_x, dtype=jnp.float32) / (num_experts + 1)  # (batch, num_experts + 1)

    # 2.
    # create the tensor (num_experts + 1, num_experts, num_classes) for q_t
    q_t_ids = jnp.eye(N=num_experts + 1, dtype=jnp.int32)
    _, q_t_ids = jnp.nonzero(a=1 - q_t_ids, size=num_experts * (num_experts + 1))
    q_t_ids = jnp.reshape(a=q_t_ids, shape=(num_experts + 1, num_experts))
    q_t_ids = jnp.tile(A=q_t_ids, reps=(y.size, 1, 1))

    q_t_one_row_out_fn = jax.vmap(
        fun=lambda q_t, q_t_id: q_t[q_t_id],
        in_axes=(0, 0)
    )

    def fixed_point_iterations(i: int, q: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """
        """
        q_new: dict[str, jax.Array] = {}

        log_q_z = log_p_z_x + jnp.sum(a=q['t'] * log_p_y_t[:, None, :], axis=-1) - 1  # (batch, num_experts + 1,)
        log_q_z -= jax.nn.logsumexp(a=log_q_z, axis=-1, keepdims=True)  # normalise

        q_new['z'] = jax.nn.softmax(x=log_q_z, axis=-1)  # (batch, num_experts + 1,)
        q_t_one_out = q_t_one_row_out_fn(q['t'], q_t_ids)  # (batch, num_experts + 1, num_experts, num_classes)
        log_q_t = log_p_t_x + jnp.sum(a=q_new['z'][:, :, None, None] * q_t_one_out * log_p_y_t[:, None, None, :], axis=-2) - 1  # (batch, num_experts + 1, num_classes)
        log_q_t -= jax.nn.logsumexp(a=log_q_t, axis=-1, keepdims=True)
        q_new['t'] = jax.nn.softmax(x=log_q_t, axis=-1)

        q_new['t'] = q_new['t'] * jnp.expand_dims(a=missing_vec, axis=-1) \
            + (1 - jnp.expand_dims(a=missing_vec, axis=-1)) * t_one_hot  # (batch, num_experts + 1, num_classes)

        return q_new

    q_new = jax.lax.fori_loop(
        lower=0,
        upper=num_iterations,
        body_fun=fixed_point_iterations,
        init_val={'z': q_z, 't': q_t}
    )

    return q_new['z'], q_new['t']


@nnx.jit
@nnx.vmap(in_axes=(0, None), out_axes=0)
def vmap_forward(model: nnx.Module, x: jax.Array) -> jax.Array:
    """perform a parallel forward pass of an ensemble
    """
    out = model(x)
    return out


@partial(nnx.jit, static_argnames=('cfg',))
def variational_free_energy(
    gating_model: nnx.Module,
    theta_model: nnx.Module,
    x: jax.Array,
    y: jax.Array,
    t: jax.Array,
    cfg: DictConfig
) -> tuple[jax.Array, jax.Array]:
    """
    """
    # region AUXILIARY
    # mask classifier
    masks = jnp.zeros(shape=(y.size, len(cfg.dataset.train_files)), dtype=jnp.int32)  # (batch, num_experts)
    masks = jnp.concatenate(
        arrays=(masks, jnp.expand_dims(a=jnp.ones_like(a=y, dtype=jnp.int32), axis=-1)),
        axis=-1
    )  # (batch, num_experts + 1)
    masks = jnp.expand_dims(a=masks, axis=-1)  # (batch, num_experts + 1, 1)

    annotations = jnp.concatenate(arrays=(t, y[:, None]), axis=-1)  # (batch, num_experts + 1)
    annotations_one_hot = jax.nn.one_hot(
        x=annotations,
        num_classes=cfg.dataset.num_classes
    )  # (batch, num_experts + 1, num_classes)
    # endregion

    logits_z_x = gating_model(x)
    log_p_z_x = jax.nn.log_softmax(x=logits_z_x, axis=-1)  # (batch, num_experts + 1)

    logits_t_x = vmap_forward(model=theta_model, x=x)
    logits_t_x = jnp.swapaxes(a=logits_t_x, axis1=0, axis2=1)  # (batch, num_experts + 1, num_classes)
    log_p_t_x = jax.nn.log_softmax(x=logits_t_x, axis=-1)  # (num_experts + 1, batch, num_classes)

    # region E STEP
    q_z, q_t = jax.lax.stop_gradient(
        x=unconstrained_posteriors(
            log_p_z_x=log_p_z_x,
            log_p_t_x=log_p_t_x,
            y=y,
            t=t,
            num_classes=cfg.dataset.num_classes,
            num_experts=len(cfg.dataset.train_files),
            num_iterations=cfg.training.num_fixed_point_iterations
        )
    )

    # initialise the parameters of lower and upper constraints
    epsilon_lower = jnp.array(
        object=cfg.hparams.epsilon_lower,
        dtype=jnp.float32
    )  # (num_experts + 1,)
    epsilon_upper = jnp.array(
        object=cfg.hparams.epsilon_upper,
        dtype=jnp.float32
    )  # (num_epxerts + 1,)

    q_z_con = constrained_posterior(
        q_z_uncon=q_z,
        epsilon_upper=epsilon_upper,
        epsilon_lower=epsilon_lower,
    )
    # endregion

    # q_t for classifier is the ground truth labels
    q_t = q_t * (1 - masks) + masks * annotations_one_hot
    loss_t = optax.losses.softmax_cross_entropy(logits=logits_t_x, labels=q_t)
    loss_t = jnp.sum(a=loss_t, axis=-1)  # (batch,)
    loss_t = jnp.mean(a=loss_t, axis=0)

    loss_z = optax.losses.softmax_cross_entropy(logits=logits_z_x, labels=q_z_con)
    loss_z = jnp.mean(a=loss_z)

    loss = loss_t + loss_z

    return loss, log_p_z_x


@partial(nnx.jit, static_argnames=('cfg',), donate_argnames=('gating_state', 'theta_state'))
def expectation_maximisation(
    x: jax.Array,
    y: jax.Array,
    t: jax.Array,
    gating_state: nnx.Optimizer,
    theta_state: nnx.Optimizer,
    cfg: DictConfig
) -> tuple[nnx.Optimizer, nnx.Optimizer, jax.Array, jax.Array]:
    """
    """
    grad_value_fn = nnx.value_and_grad(
        f=variational_free_energy,
        argnums=range(2),
        has_aux=True
    )
    (loss, log_p_z_x), (grads_gating, grads_theta) = grad_value_fn(
        gating_state.model,
        theta_state.model,
        x,
        y,
        t,
        cfg
    )

    # update parameters from gradients
    gating_state.update(grads=grads_gating)
    theta_state.update(grads=grads_theta)
    # endregion

    # return grads_gating, batch_stats_gating, grads_theta, batch_stats_theta, loss
    return gating_state, theta_state, loss, log_p_z_x


def train(
    data_source: grain.RandomAccessDataSource,
    gating_state: nnx.Optimizer,
    theta_state: nnx.Optimizer,
    cfg: DictConfig
) -> tuple[nnx.Optimizer, nnx.Optimizer, jax.Array, jax.Array]:
    """the main training procedure
    """
    # region DATA LOADER
    index_sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=1,
        shuffle=True,
        shard_options=grain.NoSharding(),
        seed=gating_state.step.value.item()  # set the random seed
    )

    transformations = []
    if cfg.hparams.resize is not None:
        transformations.append(Resize(resize_shape=cfg.hparams.resize))
    if cfg.hparams.crop_size is not None:
        transformations.append(RandomCrop(crop_size=cfg.hparams.crop_size))
    if cfg.hparams.prob_random_h_flip is not None:
        transformations.append(RandomHorizontalFlip(p=cfg.hparams.prob_random_h_flip))
    transformations.append(ToFloat())
    if cfg.hparams.mean is not None and cfg.hparams.std is not None:
        transformations.append(Normalize(mean=cfg.hparams.mean, std=cfg.hparams.std))
    transformations.append(
        grain.Batch(
            batch_size=cfg.training.batch_size,
            drop_remainder=True
        )
    )

    data_loader = grain.DataLoader(
        data_source=data_source,
        sampler=index_sampler,
        operations=transformations,
        worker_count=cfg.data_loading.num_workers,
        shard_options=grain.NoSharding(),
        read_options=grain.ReadOptions(
            num_threads=cfg.data_loading.num_threads,
            prefetch_buffer_size=cfg.data_loading.prefetch_size
        )
    )
    # endregion

    # metric to track the training loss
    loss_accum = nnx.metrics.Average()
    p_z_x_accum = [nnx.metrics.Average() for _ in range(len(cfg.dataset.train_files) + 1)]

    # set train mode
    gating_state.model.train()
    theta_state.model.train()

    for samples in tqdm(
        iterable=data_loader,
        desc='train',
        total=len(data_source) // cfg.training.batch_size,
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)  # input samples
        y = jnp.asarray(a=samples['ground_truth'], dtype=jnp.int32)  # true labels  (batch,)
        t = jnp.asarray(a=samples['label'], dtype=jnp.int32)  # annotated labels (batch, num_experts)

        gating_state, theta_state, loss, log_p_z_x = expectation_maximisation(
            x=x,
            y=y,
            t=t,
            gating_state=gating_state,
            theta_state=theta_state,
            cfg=cfg
        )

        if jnp.isnan(loss):
            raise ValueError('Training loss is NaN.')

        # tracking
        loss_accum.update(values=loss)

        for i in range(len(cfg.dataset.train_files) + 1):
            p_z_x_accum[i].update(values=jnp.exp(log_p_z_x[:, i]))

    return (
        gating_state,
        theta_state,
        loss_accum.compute(),
        [p_z.compute() for p_z in p_z_x_accum]
    )


@nnx.jit
def gating_prediction_step(x: jax.Array, model: nnx.Module) -> jax.Array:
    """
    """
    logits = model(x)
    return logits


@nnx.jit
def expert_prediction_step(x: jax.Array, model: nnx.Optimizer) -> jax.Array:
    logits = vmap_forward(model=model, x=x)  # (num_experts, batch_size, num_classes)

    return logits


def evaluate(
    data_source: grain.RandomAccessDataSource,
    gating_state: nnx.Optimizer,
    theta_state: nnx.Optimizer,
    cfg: DictConfig
) -> tuple[float, list[float], jax.Array, list[float], float, jax.Array]:
    """evaluate performance on a dataset

    Args:
        data_source:
        gating_state:
        theta_state:
        cfg: Hydra/Omega configuration dictionary

    Returns:
        accuracy: prediction accuracy of l2d
        expert_accuracies: list of expert accuracy
        coverage: the ratio of number of samples predicted by each expert
        p_z: average output of the gating model, Pr(z | x, gamma)
        ece: expected calibration error
        conf_mat: confusion matrix for the gating model w.r.t. the classifier. In other
            words, the gating model is considered as a binary classifier: human or ML
            classifier. The prediction is:
                (i) 0 if the ML classifier is incorrect
                (ii) 1 if the ML classifier is correct
    """
    # region DATA LOADER
    index_sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=1,
        shuffle=False,
        shard_options=grain.NoSharding(),
        seed=gating_state.step.value.item()  # set the random seed
    )

    transformations = []
    if cfg.hparams.resize is not None:
        transformations.append(Resize(resize_shape=cfg.hparams.resize))
    if cfg.hparams.crop_size is not None:
        transformations.append(RandomCrop(crop_size=cfg.hparams.crop_size))
    transformations.append(ToFloat())
    if cfg.hparams.mean is not None and cfg.hparams.std is not None:
        transformations.append(Normalize(mean=cfg.hparams.mean, std=cfg.hparams.std))
    transformations.append(
        grain.Batch(
            batch_size=cfg.training.batch_size,
            drop_remainder=False
        )
    )

    data_loader = grain.DataLoader(
        data_source=data_source,
        sampler=index_sampler,
        operations=transformations,
        worker_count=cfg.data_loading.num_workers,
        shard_options=grain.NoSharding(),
        read_options=grain.ReadOptions(
            num_threads=cfg.data_loading.num_threads,
            prefetch_buffer_size=cfg.data_loading.prefetch_size
        )
    )
    # endregion

    p_z_accum = [nnx.metrics.Average() for _ in range(len(cfg.dataset.test_files) + 1)]
    accuracy_accum = nnx.metrics.Accuracy()
    expert_accuracies = [nnx.metrics.Accuracy() for _ in range(len(cfg.dataset.test_files) + 1)]
    coverage = nnx.metrics.Average()
    preds_accum = []  # store prediction to calculate ECE
    labels_true = []

    # confusion matrix
    conf_pred = []  # store 0: clf is not selected, while 1: clf is selected
    conf_gt = []  # store 0: clf is incorrect, or 1: clf is correct

    # set evaluation mode
    gating_state.model.eval()
    theta_state.model.eval()

    for samples in tqdm(
        iterable=data_loader,
        desc='evaluate',
        total=len(data_source)//cfg.training.batch_size + 1,
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)  # input samples
        y = jnp.asarray(a=samples['ground_truth'], dtype=jnp.int32)  # true labels (batch_size,)
        t = jnp.asarray(a=samples['label'], dtype=jnp.int32)  # annotated labels (batch_size, num_experts)

        annotations = jnp.concatenate(arrays=(t, y[:, None]), axis=-1)

        t = jax.nn.one_hot(x=t, num_classes=cfg.dataset.num_classes)  # (batch_size, num_experts, num_classes)

        # Pr(z | x, gamma)
        logits_p_z = gating_prediction_step(x=x, model=gating_state.model)  # (batch_size, num_experts)
        if jnp.isnan(logits_p_z).any():
            logging.error(msg=gating_state.params)
            logging.error(msg=gating_state.batch_stats)
            raise ValueError('NaN detected in the output of gating function.')
        log_p_z = jax.nn.log_softmax(x=logits_p_z, axis=-1)

        for i in range(len(cfg.dataset.train_files) + 1):
            p_z_accum[i].update(values=jnp.exp(log_p_z[:, i]))

        selected_expert_ids = jnp.argmax(a=logits_p_z, axis=-1)  # (batch_size,)

        coverage.update(values=(selected_expert_ids == len(cfg.dataset.test_files)) * 1)

        # accuracy
        logits_p_t = expert_prediction_step(x=x, model=theta_state.model)
        logits_clf = logits_p_t[-1, :, :]  # (batch_size, num_classes)
        human_and_model_predictions = jnp.concatenate(arrays=(t, logits_clf[:, None, :]), axis=1)
        queried_predictions = human_and_model_predictions[jnp.arange(len(x)), selected_expert_ids, :]
        accuracy_accum.update(logits=queried_predictions, labels=y)

        # expert accuracies
        for i in range(len(expert_accuracies)):
            expert_accuracies[i].update(logits=logits_p_t[i], labels=annotations[:, i])

        preds_accum.append(queried_predictions)
        labels_true.append(y)

        # confusion matrix
        clf_selected = (selected_expert_ids == (len(cfg.dataset.train_files))) * 1
        clf_predict_labels = jnp.argmax(a=logits_clf, axis=-1)
        clf_correct_predictions = (clf_predict_labels == y) * 1
        conf_pred.append(clf_selected)
        conf_gt.append(clf_correct_predictions)

    # calculate ECE
    preds_accum = jnp.concatenate(arrays=preds_accum, axis=0)
    labels_true = jnp.concatenate(arrays=labels_true, axis=0)
    ece = tfp.stats.expected_calibration_error(
        num_bins=10,
        logits=preds_accum,
        labels_true=labels_true
    )

    # calculate confusion matrix
    conf_mat = confusion_matrix(
        predictions=jnp.concatenate(arrays=conf_pred, axis=0),
        labels=jnp.concatenate(arrays=conf_gt, axis=0),
        num_classes=2
    )

    return (
        accuracy_accum.compute(),
        [expert_accuracy.compute() for expert_accuracy in expert_accuracies],
        coverage.compute(),
        [p_z.compute() for p_z in p_z_accum],
        ece,
        conf_mat
    )
