from functools import partial
import logging

import jax
import jax.numpy as jnp

from flax.core import FrozenDict
from flax.nnx import metrics

import optax

from tensorflow_probability.substrates import jax as tfp

from chex import Array, Scalar

from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative

from omegaconf import DictConfig

from mlx import data as dx

from tqdm import tqdm

from utils import (
    TrainState,
    prepare_dataset,
    confusion_matrix
)


def create_train_state_batch(state: TrainState) -> TrainState:
    """
    """
    # region define vmap apply function
    def apply_function(params: FrozenDict, batch_stats: FrozenDict, x: Array, train: bool) -> tuple[Array, FrozenDict]:
        """
        """
        logits, batch_stats = state.apply_fn(
            variables={'params': params, 'batch_stats': batch_stats},
            x=x,
            train=train,
            mutable=['batch_stats']
        )

        return logits, batch_stats

    apply_fn_batch = jax.vmap(
        fun=apply_function,
        in_axes=(
            jax.tree.map(f=lambda x: 0, tree=state.params),
            jax.tree.map(f=lambda x: 0, tree=state.batch_stats),
            None,
            None
        )
    )

    states = TrainState.create(
        apply_fn=apply_fn_batch,
        params=state.params,
        batch_stats=state.batch_stats,
        tx=state.tx
    )

    return states


# @partial(jax.jit, device=jax.devices()[-1])
@jax.jit
def constrained_posterior(
    log_q_z_unconstrained: Array,  # (batch, num_experts + 1)
    varepsilon: Array,  # (num_experts,)
    epsilon: Array  # (num_experts,)
) -> Array:
    """calculate the work-load balancing posterior

    Args:
        log_q_z_unconstrained: the unconstrained posterior of z
        varepsilon: the hyperparameter for upper constraint
        epsilon: the hyperparameter for lower constraint

    Returns:
        log_q_z: the constrained posterior of z
    """
    def duality_Lagrangian(lmbd_params: dict[str, Array]) -> Scalar:
        """the duality of Lagrangian to find the Lagrange multiplier

        Args:
            lmbd_params: a dictionary containing the Lagrange multiplier for the lower
                and upper cases

        Returns:
            lagrangian:
        """
        lagrangian = jax.nn.logsumexp(
            a=log_q_z_unconstrained - lmbd_params['lmbd'] + lmbd_params['lmbd_lower'] - 1,
            axis=(0, 1)
        )  # scalar
        lagrangian = lagrangian + jnp.sum(a=lmbd_params['lmbd'] * varepsilon, axis=0)
        lagrangian = lagrangian - jnp.sum(a=lmbd_params['lmbd_lower'] * epsilon, axis=0)

        return lagrangian

    pg = ProjectedGradient(
        fun=duality_Lagrangian,
        projection=projection_non_negative,
        implicit_diff=False,
        tol=1e-4
    )
    pg_sol = pg.run(init_params=dict(
        lmbd=jnp.ones_like(a=varepsilon),
        lmbd_lower=jnp.ones_like(a=epsilon)
        )
    )
    lmbd_params = pg_sol.params  # (num_experts,)  <== need to check

    log_q_z = log_q_z_unconstrained - lmbd_params['lmbd'] + lmbd_params['lmbd_lower'] - 1

    # normalisation
    log_q_z -= jax.nn.logsumexp(a=log_q_z, axis=-1, keepdims=True)

    return log_q_z


@partial(
    jax.jit,
    static_argnames=('num_classes', 'num_experts', 'num_iterations'),
    # device=jax.devices()[-1]
)
def unconstrained_posteriors(
    log_p_z_x: Array,  # (batch, num_experts + 1)
    log_p_t_x: Array,  # (batch, num_experts + 1, num_classes)
    y: Array,  # (batch,)
    t: Array,  # (batch, num_experts)
    num_classes: int,
    num_experts: int,
    num_iterations: int
) -> tuple[Array, Array]:
    """
    """
    missing_vec = jnp.array(object=(t == -1), dtype=jnp.int32)  # (batch, num_experts)

    # convert annotations to one-hot vectors
    t_one_hot = jax.nn.one_hot(
        x=t * (1 - missing_vec),  # temporarily convert missing into label 0
        num_classes=num_classes
    )  # (batch, num_experts, num_classes)
    t_one_hot = optax.smooth_labels(labels=t_one_hot, alpha=1e-3)  # (batch, num_experts, num_classes)

    # append the classifier into the experts
    missing_vec = jnp.concatenate(
        arrays=(missing_vec, jnp.expand_dims(a=jnp.zeros_like(a=y, dtype=jnp.int32), axis=-1)),
        axis=1
    )  # (batch, num_experts + 1,)
    p_t_x_clf = jnp.exp(log_p_t_x[:, -1, :])  # (batch, num_classes)
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

    def fixed_point_iterations(i: int, q: dict[str, Array]) -> dict[str, Array]:
        """
        """
        q_new: dict[str, Array] = {}

        log_q_z = log_p_z_x + jnp.sum(a=q['t'] * log_p_y_t[:, None, :], axis=-1) - 1  # (batch, num_experts + 1,)
        log_q_z -= jax.nn.logsumexp(a=log_q_z, axis=-1, keepdims=True)  # normalise

        q_new['z'] = jnp.exp(log_q_z)  # (batch, num_experts + 1,)
        q_t_one_out = q_t_one_row_out_fn(q['t'], q_t_ids)  # (batch, num_experts + 1, num_experts, num_classes)
        log_q_t = log_p_t_x + jnp.sum(a=q_new['z'][:, :, None, None] * q_t_one_out * log_p_y_t[:, None, None, :], axis=-2) - 1  # (batch, num_experts + 1, num_classes)
        log_q_t -= jax.nn.logsumexp(a=log_q_t, axis=-1, keepdims=True)
        q_new['t'] = jnp.exp(log_q_t)

        q_new['t'] = q_new['t'] * jnp.expand_dims(a=missing_vec, axis=-1) \
            + (1 - jnp.expand_dims(a=missing_vec, axis=-1)) * t_one_hot  # (batch, num_experts + 1, num_classes)

        return q_new

    q_new = jax.lax.fori_loop(
        lower=0,
        upper=num_iterations,
        body_fun=fixed_point_iterations,
        init_val={'z': q_z, 't': q_t}
    )

    # # fixed-point iteration
    # for _ in range(num_iterations):
    #     log_q_z = log_p_z_x + jnp.sum(a=q_t * log_p_y_t[:, None, :], axis=-1) - 1  # (batch, num_experts + 1,)
    #     log_q_z -= jax.nn.logsumexp(a=log_q_z, axis=-1, keepdims=True)  # normalise

    #     q_z_new = jnp.exp(log_q_z)  # (batch, num_experts + 1,)
    #     q_t_one_out = q_t_one_row_out_fn(q_t, q_t_ids)  # (batch, num_experts + 1, num_experts, num_classes)
    #     log_q_t = log_p_t_x + jnp.sum(a=q_z[:, :, None, None] * q_t_one_out * log_p_y_t[:, None, None, :], axis=-2) - 1  # (batch, num_experts + 1, num_classes)
    #     log_q_t -= jax.nn.logsumexp(a=log_q_t, axis=-1, keepdims=True)
    #     q_t = jnp.exp(log_q_t)

    #     q_t = q_t * jnp.expand_dims(a=missing_vec, axis=-1) \
    #         + (1 - jnp.expand_dims(a=missing_vec, axis=-1)) * t_one_hot  # (batch, num_experts + 1, num_classes)
    #     q_z = q_z_new
    # # endregion

    return q_new['z'], q_new['t']


@partial(
    jax.jit,
    static_argnames=('cfg',),
    donate_argnames=('gating_state', 'theta_state')
    # device=jax.devices()[-1]
)
def expectation_maximisation(
    x: Array,
    y: Array,
    t: Array,
    gating_state: TrainState,
    theta_state: TrainState,
    cfg: DictConfig
) -> tuple[TrainState, TrainState, Scalar, Array]:
    """
    """
    # mask classifier
    masks = jnp.zeros(shape=(y.size, len(cfg.dataset.train_files)), dtype=jnp.int32)  # (batch, num_experts)
    masks = jnp.concatenate(
        arrays=(masks, jnp.expand_dims(a=jnp.ones_like(a=y, dtype=jnp.int32), axis=-1)),
        axis=-1
    )  # (batch, num_experts + 1)
    masks = jnp.expand_dims(a=masks, axis=-1)  # (batch, num_experts + 1, 1)

    annotations = jnp.concatenate(
        arrays=(t, y[:, None]),
        axis=-1
    )  # (batch, num_experts + 1)
    missing_mat = (annotations == -1) * 1  # (batch, num_experts + 1)
    annotations_one_hot = jax.nn.one_hot(
        x=annotations * (1 - missing_mat),
        num_classes=cfg.dataset.num_classes
    )  # (batch, num_experts + 1, num_classes)

    # set prior parameter
    dir_params = jnp.array(
        object=cfg.hparams.Dirichlet_concentration,
        dtype=jnp.float32
    )  # (num_experts,)

    def variational_free_energy(
        gating_params: FrozenDict,
        theta_params: FrozenDict
    ) -> tuple[Scalar, tuple[FrozenDict, FrozenDict, Array]]:
        """
        """
        logits_z_x, batch_stats_gating = gating_state.apply_fn(
            variables={'params': gating_params, 'batch_stats': gating_state.batch_stats},
            x=x,
            train=True,
            mutable=['batch_stats']
        )
        log_p_z_x = jax.nn.log_softmax(x=logits_z_x, axis=-1)  # (batch, num_experts + 1)

        logits_t_x, batch_stats_theta = theta_state.apply_fn(
            theta_params,
            theta_state.batch_stats,
            x,
            True
        )
        log_p_t_x = jax.nn.log_softmax(x=logits_t_x, axis=-1)  # (num_experts + 1, batch, num_classes)
        log_p_t_x = jnp.swapaxes(a=log_p_t_x, axis1=0, axis2=1)  # (batch, num_experts + 1, num_classes)

        # expectation
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
        log_q_z = constrained_posterior(
            log_q_z_unconstrained=jnp.log(q_z),
            varepsilon=jnp.array(object=cfg.hparams.epsilon_upper, dtype=jnp.float32),
            epsilon=jnp.array(object=cfg.hparams.epsilon_lower, dtype=jnp.float32)
        )

        # q_t for classifier is the ground truth labels
        q_t = q_t * (1 - masks) + masks * annotations_one_hot
        loss_t = -jnp.sum(a=q_t * log_p_t_x, axis=(-1, -2))  # (batch,)

        loss_z = -jnp.sum(a=jnp.exp(log_q_z) * log_p_z_x, axis=-1)  # (batch,)
        loss = jnp.mean(a=loss_t + loss_z, axis=0)

        # prior
        log_avg_p_z_x = jnp.mean(a=log_p_z_x, axis=0)
        loss_prior = -jnp.sum((dir_params - 1) * log_avg_p_z_x, axis=0)

        loss = loss + cfg.training.batch_size / cfg.dataset.length * loss_prior

        return loss, (batch_stats_gating, batch_stats_theta, log_p_z_x)

    grad_value_fn = jax.value_and_grad(
        fun=variational_free_energy,
        argnums=range(2),
        has_aux=True
    )
    (loss, (batch_stats_gating, batch_stats_theta, log_p_z_x)), (grads_gating, grads_theta) = grad_value_fn(
        gating_state.params,
        theta_state.params
    )

    # update parameters from gradients
    gating_state = gating_state.apply_gradients(grads=grads_gating)
    theta_state = theta_state.apply_gradients(grads=grads_theta)

    # update batch statistics
    gating_state = gating_state.replace(batch_stats=batch_stats_gating['batch_stats'])
    theta_state = theta_state.replace(batch_stats=batch_stats_theta['batch_stats'])
    # endregion

    # return grads_gating, batch_stats_gating, grads_theta, batch_stats_theta, loss
    return gating_state, theta_state, loss, log_p_z_x


def train(
    dataset: dx._c.Buffer,
    gating_state: TrainState,
    theta_state: TrainState,
    cfg: DictConfig
) -> tuple[TrainState, TrainState, Scalar, Array]:
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
    p_z_x_accum = [metrics.Average() for _ in range(len(cfg.dataset.train_files) + 1)]

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


# @partial(jax.jit, device=jax.devices()[-1])
@jax.jit
def gating_prediction_step(x: Array, state: TrainState) -> Array:
    """
    """
    logits, _ = state.apply_fn(
        variables={'params': state.params, 'batch_stats': state.batch_stats},
        x=x,
        train=False,  # prediction
        mutable=['batch_stats']
    )
    return logits


# @partial(jax.jit, device=jax.devices()[-1])
@jax.jit
def expert_prediction_step(x: Array, theta_state: TrainState) -> Array:
    logits, _ = theta_state.apply_fn(
        theta_state.params,
        theta_state.batch_stats,
        x,
        False
    )  # (num_experts, batch_size, num_classes)

    return logits


def evaluate(
    dataset: dx._c.Buffer,
    gating_state: TrainState,
    theta_state: TrainState,
    cfg: DictConfig
) -> tuple[Scalar, Scalar, list[Scalar], Scalar, Array, Array]:
    """evaluate performance on a dataset

    Args:
        dataset:
        gating_state:
        theta_state:
        cfg: Hydra/Omega configuration dictionary

    Returns:
        accuracy: prediction accuracy of l2d
        coverage:
        p_z: average output of the gating model, Pr(z | x, gamma)
        ece: expected calibration error
        conf_mat: confusion matrix for the gating model w.r.t. the classifier. In other
            words, the gating model is considered as a binary classifier: human or ML
            classifier. The prediction is:
                (i) 0 if the ML classifier is incorrect
                (ii) 1 if the ML classifier is correct
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

    p_z_accum = [metrics.Average() for _ in range(len(cfg.dataset.test_files) + 1)]
    accuracy_accum = metrics.Accuracy()
    expert_accuracies = [metrics.Accuracy() for _ in range(len(cfg.dataset.test_files) + 1)]
    coverage = metrics.Average()
    preds_accum = []  # store prediction to calculate ECE
    labels_true = []

    # confusion matrix
    conf_pred = []  # store 0: clf is not selected, while 1: clf is selected
    conf_gt = []  # store 0: clf is incorrect, or 1: clf is correct

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
        y = jnp.asarray(a=samples['ground_truth'], dtype=jnp.int32)  # true labels (batch_size,)
        t = jnp.asarray(a=samples['label'], dtype=jnp.int32)  # annotated labels (batch_size, num_experts)

        annotations = jnp.concatenate(arrays=(t, y[:, None]), axis=-1)

        t = jax.nn.one_hot(x=t, num_classes=cfg.dataset.num_classes)  # (batch_size, num_experts, num_classes)

        # Pr(z | x, gamma)
        logits_p_z = gating_prediction_step(x=x, state=gating_state)  # (batch_size, num_experts)
        if jnp.isnan(logits_p_z).any():
            logging.error(msg=gating_state.params)
            logging.error(msg=gating_state.batch_stats)
            raise ValueError('NaN detected in the output of gating function.')
        log_p_z = jax.nn.log_softmax(x=logits_p_z, axis=-1)

        for i in range(len(cfg.dataset.train_files) + 1):
            p_z_accum[i].update(values=jnp.exp(log_p_z[:, i]))

        selected_expert_ids = jnp.argmax(a=logits_p_z, axis=-1)  # (batch_size,)

        # coverage
        coverage_flag = (selected_expert_ids == len(cfg.dataset.test_files)) * 1
        coverage.update(values=coverage_flag)

        # accuracy
        logits_p_t = expert_prediction_step(x=x, theta_state=theta_state)
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
