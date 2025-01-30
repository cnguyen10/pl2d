import jax
import jax.numpy as jnp
from flax import nnx

from typing import Sequence


class BasicBlock(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            in_planes: int,
            planes: int,
            stride: int = 1,
            expansion: int = 1,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()
        self.expansion = expansion

        if stride != 1 or in_planes != (expansion * planes):
            self.shortcut = nnx.Conv(
                in_features=in_planes,
                out_features=expansion * self.planes,
                kernel_size=(1, 1),
                strides=stride,
                use_bias=False,
                dtype=dtype,
                rngs=rngs
            )
        else:
            self.shortcut = lambda x: x
        
        self.conv1 = nnx.Conv(
            in_features=in_planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=stride,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=planes, rngs=rngs)
        
        self.conv2 = nnx.Conv(
            in_features=planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(num_features=planes, rngs=rngs)


    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.conv1(inputs=x)
        out = self.bn1(x=out)
        out = nnx.relu(x=out)

        out = self.conv2(inputs=out)
        out = self.bn2(x=out)
        out = out + self.shortcut(x)
        out = nnx.relu(x=out)

        return out


class PreActBlock(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            in_planes: int,
            planes: int,
            stride: int = 1,
            expansion: int = 1,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()
        self.expansion = expansion

        self.bn1 = nnx.BatchNorm(num_features=in_planes, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=in_planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=stride,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )

        self.bn2 = nnx.BatchNorm(num_features=planes, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )

        if stride != 1 or in_planes != (expansion * planes):
            self.shortcut = nnx.Conv(
                in_features=in_planes,
                out_features=expansion * planes,
                kernel_size=(1, 1),
                strides=stride,
                use_bias=False,
                dtype=dtype,
                rngs=rngs
            )
        else:
            self.shortcut = lambda x: x


    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.bn1(x=x)
        out = nnx.relu(x=out)

        shortcut = self.shortcut(out)

        out = self.conv1(inputs=out)

        out = self.bn2(x=out)
        out = nnx.relu(x=out)
        out = self.conv2(inputs=out)

        return out + shortcut


class PreActBottleneck(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            in_planes: int,  # number of input channels
            planes: int,  # number of output channels
            stride: int,
            expansion: int = 4,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()
        self.expansion = expansion

        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = nnx.Conv(
                in_features=in_planes,
                out_features=expansion * planes,
                kernel_size=(1, 1),
                strides=stride,
                use_bias=False,
                dtype=dtype,
                rngs=rngs
            )
        else:
            self.shortcut = lambda x: x
        
        self.conv1 = nnx.Conv(
            in_features=in_planes,
            out_features=planes,
            kernel_size=(1, 1),
            strides=1,
            padding=0,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=planes, rngs=rngs)

        self.conv2 = nnx.Conv(
            in_features=planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=stride,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(num_features=planes, rngs=rngs)

        self.conv3 = nnx.Conv(
            in_features=planes,
            out_features=expansion * planes,
            kernel_size=(1, 1),
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn3 = nnx.BatchNorm(num_features=expansion * planes, rngs=rngs)


    def __call__(self, x: jax.Array) -> jax.Array:
        shortcut = self.shortcut(x)

        out = self.conv1(inputs=x)
        out = self.bn1(x=out)
        out = nnx.relu(x=out)

        out = self.conv2(inputs=out)
        out = self.bn2(x=out)
        out = nnx.relu(x=out)

        out = self.conv3(inputs=out)
        out = self.bn3(x=out)
        out = out + shortcut
        out = nnx.relu(x=out)

        return out


class PreActResNet(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            block: PreActBlock | PreActBottleneck,
            num_blocks: Sequence[int],
            in_planes: int = 64,
            num_classes: int = None,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()

        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=64,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=in_planes, rngs=rngs)

        planes = [64, 128, 256, 512]
        self.layers = [None] * len(num_blocks)
        for i in range(len(num_blocks)):
            in_planes, self.layers[i] = make_layer(
                rngs=rngs,
                block=block,
                in_planes=in_planes,
                planes=planes[i],
                num_blocks=num_blocks[i],
                stride=min(i + 1, 2),
                dtype=dtype
            )
        
        if num_classes is None:
            self.clf = lambda x: x
        else:
            self.clf = nnx.Linear(
                in_features=int(512 * in_planes / planes[-1]),
                out_features=num_classes,
                rngs=rngs
            )


    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.conv1(inputs=x)
        out = self.bn1(x=out)
        out = nnx.relu(x=out)

        for i in range(len(self.layers)):
            out = self.layers[i](out)
        
        out = jnp.mean(a=out, axis=(1, 2))

        out = self.clf(out)

        return out


def ResNet18(num_classes: int, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32) -> PreActResNet:
    return PreActResNet(
        rngs=rngs,
        block=PreActBlock,
        num_blocks=(2, 2, 2, 2),
        num_classes=num_classes,
        dtype=dtype
    )


def make_layer(
        block: PreActBlock | PreActBottleneck,
        in_planes: int,
        planes: int,
        num_blocks: Sequence[int],
        stride: int,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32) -> tuple[int, nnx.Module]:
    """
    """
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        block_layer = block(
            rngs=rngs,
            in_planes=in_planes,
            planes=planes,
            stride=stride,
            dtype=dtype
        )
        layers.append(block_layer)
        in_planes = planes * block_layer.expansion

    return in_planes, nnx.Sequential(*layers)
