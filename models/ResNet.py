import jax
import jax.numpy as jnp
from flax import nnx

from collections.abc import Sequence, Callable


class BasicBlock(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            dropout_rate: float,
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
            kernel_size=(3, 3),
            strides=stride,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(
            num_features=planes,
            dtype=dtype,
            rngs=rngs
        )
        
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
        self.bn2 = nnx.BatchNorm(
            num_features=planes,
            dtype=dtype,
            rngs=rngs
        )

        self.dropout = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)


    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.conv1(inputs=x)
        out = self.bn1(x=out)
        out = nnx.relu(x=out)

        out = self.dropout(out)

        out = self.conv2(inputs=out)
        out = self.bn2(x=out)

        out = out + self.shortcut(x)
        out = nnx.relu(x=out)

        return out


class Bottleneck(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            dropout_rate: float,
            in_planes: int,
            planes: int,
            stride: int = 1,
            expansion: int = 4,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()
        self.expansion = expansion

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

        self.conv1 = nnx.Conv(
            in_features=in_planes,
            out_features=planes,
            kernel_size=(1, 1),
            strides=stride,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=planes, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=self.bn1.num_features,
            out_features=self.bn1.num_features,
            kernel_size=(3, 3),
            strides=stride,
            dtype=dtype,
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(
            num_features=planes,
            dtype=dtype,
            rngs=rngs
        )
        self.conv3 = nnx.Conv(
            in_features=planes,
            out_features=planes * self.expansion,
            kernel_size=(1, 1),
            dtype=dtype,
            rngs=rngs
        )
        self.bn3 = nnx.BatchNorm(
            num_features=planes * self.expansion,
            dtype=dtype,
            rngs=rngs
        )

        self.dropout = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)

    def forward(self, x: jax.Array) -> jax.Array:
        out = self.conv1(x)
        out = self.bn1(out)
        out = nnx.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nnx.relu(out)

        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.shortcut(x)

        out = nnx.relu(out + shortcut)

        return out


class PreActBlock(nnx.Module):
    """Implement the pre-activated residual neural networks in the paper:
    'Identity Mappings in Deep Residual Networks' (ECCV 2016)
    """
    def __init__(
            self,
            rngs: nnx.Rngs,
            dropout_rate: float,
            in_planes: int,
            planes: int,
            stride: int = 1,
            expansion: int = 1,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()
        self.expansion = expansion

        self.bn1 = nnx.BatchNorm(
            num_features=in_planes,
            dtype=dtype,
            rngs=rngs
        )
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

        self.bn2 = nnx.BatchNorm(
            num_features=planes,
            dtype=dtype,
            rngs=rngs
        )
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
        
        self.dropout = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)


    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.bn1(x=x)
        out = nnx.relu(x=out)

        shortcut = self.shortcut(x)

        out = self.conv1(inputs=out)

        out = self.bn2(x=out)
        out = nnx.relu(x=out)

        out = self.dropout(out)

        out = self.conv2(inputs=out)

        return out + shortcut


class PreActBottleneck(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            dropout_rate: float,
            in_planes: int,  # number of input channels
            planes: int,  # number of output channels
            stride: int,
            expansion: int = 4,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()
        self.expansion = expansion

        if stride != 1 or in_planes != self.expansion * planes:
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
        
        self.bn1 = nnx.BatchNorm(num_features=in_planes, rngs=rngs)

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

        self.bn2 = nnx.BatchNorm(num_features=planes, rngs=rngs)

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

        self.bn3 = nnx.BatchNorm(num_features=planes, rngs=rngs)

        self.conv3 = nnx.Conv(
            in_features=planes,
            out_features=expansion * planes,
            kernel_size=(1, 1),
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )

        self.dropout = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)


    def __call__(self, x: jax.Array) -> jax.Array:
        shortcut = self.shortcut(x)

        out = self.bn1(x=x)
        out = nnx.relu(x=out)

        out = self.conv1(inputs=out)

        out = self.bn2(x=out)
        out = nnx.relu(x=out)

        out = self.conv2(inputs=out)

        out = self.bn3(x=out)
        out = nnx.relu(x=out)

        out = self.dropout(out)

        out = self.conv3(inputs=out)

        out = out + shortcut

        return out


class ResNet(nnx.Module):
    def __init__(
            self,
            in_channels: int,
            conv1_kernel_size: tuple[int, int] | int,
            rngs: nnx.Rngs,
            dropout_rate: float,
            block: Callable[..., BasicBlock | Bottleneck],
            num_blocks: Sequence[int],
            in_planes: int = 64,
            num_classes: int | None = None,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()

        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=64,
            kernel_size=conv1_kernel_size,
            strides=2,
            padding=3,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=in_planes, dtype=dtype, rngs=rngs)

        planes = [64, 128, 256, 512]
        self.layers = []
        for i in range(len(num_blocks)):
            in_planes, layer_temp = make_layer(
                block=block,
                in_planes=in_planes,
                planes=planes[i],
                num_blocks=num_blocks[i],
                stride=min(i + 1, 2),
                rngs=rngs,
                dropout_rate=dropout_rate,
                dtype=dtype
            )

            self.layers.append(layer_temp)

        if num_classes is None:
            self.clf = lambda x: x
        else:
            self.clf = nnx.Linear(
                in_features=int(512 * in_planes / planes[-1]),
                out_features=num_classes,
                dtype=dtype,
                rngs=rngs
            )
        
        # dropout of the first conv and the last fully-connected layers
        self.dropout_conv = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)
        self.dropout_mlp = nnx.Dropout(rate=dropout_rate, broadcast_dims=(-1,), rngs=rngs)


    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.conv1(inputs=x)
        out = self.bn1(x=out)
        out = nnx.relu(x=out)
        out = nnx.max_pool(inputs=out, window_shape=(3, 3), strides=(2, 2), padding='VALID')

        out = self.dropout_conv(out)

        for i in range(len(self.layers)):
            out = self.layers[i](out)

        out = jnp.mean(a=out, axis=(1, 2))

        out = self.dropout_mlp(out)

        out = self.clf(out)

        return out


class PreActResNet(nnx.Module):
    def __init__(
            self,
            in_channels: int,
            conv1_kernel_size: tuple[int, int] | int,
            rngs: nnx.Rngs,
            dropout_rate: float,
            block: Callable[..., PreActBlock | PreActBottleneck],
            num_blocks: Sequence[int],
            in_planes: int = 64,
            num_classes: int | None = None,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()

        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=64,
            kernel_size=conv1_kernel_size,
            strides=1,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=in_planes, dtype=dtype, rngs=rngs)

        planes = [64, 128, 256, 512]
        self.layers = []
        for i in range(len(num_blocks)):
            in_planes, layer_temp = make_layer(
                block=block,
                in_planes=in_planes,
                planes=planes[i],
                num_blocks=num_blocks[i],
                stride=min(i + 1, 2),
                rngs=rngs,
                dropout_rate=dropout_rate,
                dtype=dtype
            )
            self.layers.append(layer_temp)

        # dropout of the first conv and the last fully-connected layers
        self.dropout_conv = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)
        self.dropout_mlp = nnx.Dropout(rate=dropout_rate, broadcast_dims=(-1,), rngs=rngs)
        
        if num_classes is None:
            self.clf = lambda x: x
        else:
            self.clf = nnx.Linear(
                in_features=int(512 * in_planes / planes[-1]),
                out_features=num_classes,
                dtype=dtype,
                rngs=rngs
            )


    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.conv1(inputs=x)

        out = self.dropout_conv(out)

        for i in range(len(self.layers)):
            out = self.layers[i](out)

        out = jnp.mean(a=out, axis=(1, 2))

        out = self.dropout_mlp(out)

        out = self.clf(out)

        return out


def resnet18(
        num_classes: int,
        in_channels: int,
        conv1_kernel_size: tuple[int, int] | int,
        rngs: nnx.Rngs,
        dropout_rate: float,
        dtype: jnp.dtype = jnp.float32) -> ResNet:
    return ResNet(
        in_channels=in_channels,
        conv1_kernel_size=conv1_kernel_size,
        rngs=rngs,
        dropout_rate=dropout_rate,
        block=BasicBlock,
        num_blocks=(2, 2, 2, 2),
        num_classes=num_classes,
        dtype=dtype
    )


def resnet18_preact(
        num_classes: int,
        in_channels: int,
        conv1_kernel_size: tuple[int, int] | int,
        rngs: nnx.Rngs,
        dropout_rate: float,
        dtype: jnp.dtype = jnp.float32) -> PreActResNet:
    return PreActResNet(
        in_channels=in_channels,
        conv1_kernel_size=conv1_kernel_size,
        rngs=rngs,
        dropout_rate=dropout_rate,
        block=PreActBlock,
        num_blocks=(2, 2, 2, 2),
        num_classes=num_classes,
        dtype=dtype
    )


def make_layer(
        block: Callable[..., BasicBlock | Bottleneck | PreActBlock | PreActBottleneck],
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        rngs: nnx.Rngs,
        dropout_rate: float,
        dtype: jnp.dtype = jnp.float32) -> tuple[int, nnx.Module]:
    """
    """
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        block_layer = block(
            rngs=rngs,
            dropout_rate=dropout_rate,
            in_planes=in_planes,
            planes=planes,
            stride=stride,
            dtype=dtype
        )
        layers.append(block_layer)
        in_planes = planes * block_layer.expansion

    return in_planes, nnx.Sequential(*layers)