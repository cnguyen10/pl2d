import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from functools import partial

from typing import Any, Sequence, Optional
import chex


class BasicBlock(nn.Module):
    in_planes: int
    planes: int
    stride: int = 1
    expansion: int = 1
    train: Optional[bool] = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> Any:
        # train = nn.merge_param(name='train', a=self.train, b=train)

        if self.stride != 1 or self.in_planes != (self.expansion * self.planes):
            shortcut = nn.Conv(
                features=self.expansion * self.planes,
                kernel_size=(1, 1),
                strides=self.stride,
                use_bias=False,
                dtype=self.dtype
            )(inputs=x)
        else:
            shortcut = x

        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=self.stride,
            padding=1,
            use_bias=False,
            dtype=self.dtype
        )(inputs=x)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.relu(x=out)

        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            use_bias=False,
            dtype=self.dtype
        )(inputs=out)
        out = nn.BatchNorm(use_running_average=not train)(x=out)

        out = out + shortcut

        out = nn.relu(x=out)

        return out


class PreActBlock(nn.Module):
    """Pre-activation version of the Basic Block"""
    in_planes: int
    planes: int
    stride: int = 1
    expansion: int = 1
    train: Optional[bool] = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> Any:
        # train = nn.merge_param(name='train', a=self.train, b=train)

        out = nn.BatchNorm(use_running_average=not train)(x=x)
        out = nn.relu(x=out)

        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=self.stride,
            padding=1,
            use_bias=False,
            dtype=self.dtype
        )(inputs=x)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.relu(x=out)

        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            use_bias=False,
            dtype=self.dtype
        )(inputs=out)

        if self.stride != 1 or self.in_planes != (self.expansion * self.planes):
            shortcut = nn.BatchNorm(use_running_average=not train)(x=x)
            shortcut = nn.relu(x=shortcut)
            shortcut = nn.Conv(
                features=self.expansion * self.planes,
                kernel_size=(1, 1),
                strides=self.stride,
                use_bias=False,
                dtype=self.dtype
            )(inputs=shortcut)
        else:
            shortcut = x

        out = out + shortcut

        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    in_planes: int  # number of input channels
    planes: int  # number of output channels
    stride: int
    expansion: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> chex.Array:
        # train = nn.merge_param(name='train', a=self.train, b=train)

        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            shortcut = nn.Conv(
                features=self.expansion * self.planes,
                kernel_size=(1, 1),
                strides=self.stride,
                use_bias=False,
                dtype=self.dtype
            )(inputs=x)
        else:
            shortcut = x

        out = nn.Conv(
            features=self.planes,
            kernel_size=(1, 1),
            strides=1,
            padding=0,
            use_bias=False,
            dtype=self.dtype
        )(inputs=x)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.relu(x=out)

        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=self.stride,
            padding=1,
            use_bias=False,
            dtype=self.dtype
        )(inputs=out)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.relu(x=out)

        out = nn.Conv(
            features=self.expansion * self.planes,
            kernel_size=(1, 1),
            use_bias=False,
            dtype=self.dtype
        )(inputs=out)
        out = nn.BatchNorm(use_running_average=not train)(x=out)

        out = out + shortcut

        out = nn.relu(x=out)

        return out


class PreActResNetFeature(nn.Module):
    """Extract features from a Resnet
    """
    block: PreActBlock | PreActBottleneck
    num_blocks: Sequence[int]
    in_planes: int = 64
    train: bool | None = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> chex.Array:
        # train = nn.merge_param(name='train', a=self.train, b=train)

        out = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            use_bias=False,
            dtype=self.dtype
        )(inputs=x)

        in_planes, layer1 = make_layer(
            block=self.block,
            in_planes=self.in_planes,
            planes=64,
            num_blocks=self.num_blocks[0],
            stride=1,
            train=train,
            dtype=self.dtype
        )
        in_planes, layer2 = make_layer(
            block=self.block,
            in_planes=in_planes,
            planes=128,
            num_blocks=self.num_blocks[1],
            stride=2,
            train=train,
            dtype=self.dtype
        )
        in_planes, layer3 = make_layer(
            block=self.block,
            in_planes=in_planes,
            planes=256,
            num_blocks=self.num_blocks[2],
            stride=2,
            train=train,
            dtype=self.dtype
        )
        _, layer4 = make_layer(
            block=self.block,
            in_planes=in_planes,
            planes=512,
            num_blocks=self.num_blocks[3],
            stride=2,
            train=train,
            dtype=self.dtype
        )

        out = layer1(out)
        out = layer2(out)
        out = layer3(out)
        out = layer4(out)

        out = out.mean(axis=(1, 2))

        return out


class Classifier(nn.Module):
    num_classes: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        out = nn.Dense(features=self.num_classes, dtype=self.dtype)(inputs=x)
        return out


class PreActResNet(nn.Module):
    block: PreActBlock | PreActBottleneck
    num_blocks: Sequence[int]
    num_classes: int = 10
    in_planes: int = 64
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.features = PreActResNetFeature(
            block=self.block,
            num_blocks=self.num_blocks,
            in_planes=self.in_planes,
            dtype=self.dtype
        )
        self.classifier = Classifier(
            num_classes=self.num_classes,
            dtype=self.dtype
        )
        return None
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        rngs = {"params": rng}

        random_params = self.init(rngs, pixel_values)

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> chex.Array:
        out = self.features(x=x, train=train)
        out = self.classifier(x=out)
        return out


def ResNet(num_blocks: Sequence[int], num_classes: int, dtype: jnp.dtype = jnp.float32) -> PreActResNet:
    return PreActResNet(
        block=PreActBlock,
        num_blocks=num_blocks,
        num_classes=num_classes,
        dtype=dtype
    )


def ResNet10(num_classes: int, dtype: jnp.dtype = jnp.float32) -> PreActResNet:
    return ResNet(num_blocks=(1, 1, 1, 1), num_classes=num_classes, dtype=dtype)


def ResNet18(num_classes: int, input_shape: tuple[int, int, int, int] = (1, 32, 32, 3), dtype: jnp.dtype = jnp.float32) -> PreActResNet:
    return ResNet(num_blocks=(2, 2, 2, 2), num_classes=num_classes, dtype=dtype)


def make_layer(
    block: PreActBlock | PreActBottleneck,
    in_planes: int,
    planes: int,
    num_blocks: Sequence[int],
    stride: int,
    train: bool,
    dtype: jnp.dtype = jnp.float32
) -> tuple[int, nn.Module]:
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(
            partial(
                block,
                in_planes=in_planes,
                planes=planes,
                stride=stride,
                train=train,
                dtype=dtype
            )()
        )
        in_planes = planes * block.expansion

    return in_planes, nn.Sequential(layers=layers)