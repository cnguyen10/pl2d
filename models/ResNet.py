from dataclasses import dataclass
from transformers import FlaxResNetForImageClassification, ResNetConfig

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from chex import Array, PRNGKey


@dataclass
class ResNet:
    depths: tuple[int, int, int, int]
    layer_type: str
    num_classes: int
    input_shape: tuple[int, int, int, int] = (1, 224, 224, 3)
    dtype: jnp.dtype = jnp.float32

    def __post_init__(self) -> None:
        configuration = ResNetConfig(
            num_channels=self.input_shape[-1],
            depths=self.depths,
            layer_type=self.layer_type,
            num_labels=self.num_classes
        )

        self.resnet = FlaxResNetForImageClassification(
            config=configuration,
            input_shape=self.input_shape,
            dtype=self.dtype
        )

        return None

    def apply(self, variables: dict, x: Array, train: bool, mutable: list[str] = ['batch_stats']) -> tuple[Array, FrozenDict]:
        # convert x from channel-last to channel-first
        x = jnp.swapaxes(a=x, axis1=1, axis2=-1)

        outputs = self.resnet(
            pixel_values=x,
            params=variables,
            train=train,
            return_dict=False
        )

        logits = jax.tree.leaves(tree=outputs[0])[0]

        if train:
            batch_stats = outputs[1]
        else:
            batch_stats = dict(batch_stats=variables['batch_stats'])

        return logits, batch_stats
    
    def init_weights(self, rng: PRNGKey, input_shape: tuple[int, int, int, int] = None) -> FrozenDict:
        """initialise parameters, both weights and batch statistics
        """
        if input_shape is None:
            input_shape = self.resnet.input_shape
        
        params = self.resnet.init_weights(rng=rng, input_shape=input_shape)

        return params


def resnet10(num_classes: int, input_shape: tuple[int, int, int, int] = (1, 224, 224, 3), dtype: jnp.dtype = jnp.float32) -> ResNet:
    depths = (1, 1, 1, 1)
    layer_type = 'basic'

    return ResNet(depths=depths, layer_type=layer_type, num_classes=num_classes, input_shape=input_shape, dtype=dtype)


def resnet18(num_classes: int, input_shape: tuple[int, int, int, int] = (1, 224, 224, 3), dtype: jnp.dtype = jnp.float32) -> ResNet:
    depths = (2, 2, 2, 2)
    layer_type = 'basic'

    return ResNet(depths=depths, layer_type=layer_type, num_classes=num_classes, input_shape=input_shape, dtype=dtype)


def resnet34(num_classes: int, input_shape: tuple[int, int, int, int] = (1, 224, 224, 3), dtype: jnp.dtype = jnp.float32) -> ResNet:
    depths = (3, 4, 6, 3)
    layer_type = 'basic'

    return ResNet(depths=depths, layer_type=layer_type, num_classes=num_classes, input_shape=input_shape, dtype=dtype)


def resnet50(num_classes: int, input_shape: tuple[int, int, int, int] = (1, 224, 224, 3), dtype: jnp.dtype = jnp.float32) -> ResNet:
    depths = (3, 4, 6, 3)
    layer_type = 'bottleneck'

    return ResNet(depths=depths, layer_type=layer_type, num_classes=num_classes, input_shape=input_shape, dtype=dtype)


def resnet101(num_classes: int, input_shape: tuple[int, int, int, int] = (1, 224, 224, 3), dtype: jnp.dtype = jnp.float32) -> ResNet:
    depths = (3, 4, 23, 3)
    layer_type = 'bottleneck'

    return ResNet(depths=depths, layer_type=layer_type, num_classes=num_classes, input_shape=input_shape, dtype=dtype)