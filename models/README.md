## Models

This folder contains the implementation of Pre-act-resnet and standard Resnet.

### Pre-act-resnet

The Pre-act-resnet architecture is implemented in `PreActResNet.py`. This model is designed for small size images (e.g., Cifar), where its first convolutional filter size is 3-by-3. The implementation is inpsired from the PyTorch version one, but adapted into Flax.

### Resnet

The standard Resnet architecture is implemented in `ResNet.py`. In particular, it uses the pre-defined Resnet in HuggingFace's `transformers[flax]` library.