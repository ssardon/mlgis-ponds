"""
ML model definitions for avocado pond detection (mostly UNETs).
Preserves all original model architecture from masterml.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import keras_cv

def build_unet(input_shape, config):
    """
    UNet with batch normalization for sparse object detection.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Encoder
    c1 = layers.Conv2D(8, 3, padding='same', use_bias=False)(x)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.LeakyReLU(alpha=0.1)(c1)
    c1 = layers.Conv2D(8, 3, padding='same', use_bias=False)(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.LeakyReLU(alpha=0.1)(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(16, 3, padding='same', use_bias=False)(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.LeakyReLU(alpha=0.1)(c2)
    c2 = layers.Conv2D(16, 3, padding='same', use_bias=False)(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.LeakyReLU(alpha=0.1)(c2)
    p2 = layers.MaxPooling2D()(c2)

    # Bridge
    b = layers.Conv2D(32, 3, padding='same', use_bias=False)(p2)
    b = layers.BatchNormalization()(b)
    b = layers.LeakyReLU(alpha=0.1)(b)
    b = layers.Conv2D(32, 3, padding='same', use_bias=False)(b)
    b = layers.BatchNormalization()(b)
    b = layers.LeakyReLU(alpha=0.1)(b)
    b = layers.Dropout(0.3)(b)

    # Decoder
    u2 = layers.UpSampling2D()(b)
    u2 = layers.concatenate([u2, c2])
    c3 = layers.Conv2D(16, 3, padding='same', use_bias=False)(u2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.LeakyReLU(alpha=0.1)(c3)
    c3 = layers.Conv2D(16, 3, padding='same', use_bias=False)(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.LeakyReLU(alpha=0.1)(c3)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.concatenate([u1, c1])
    c4 = layers.Conv2D(8, 3, padding='same', use_bias=False)(u1)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.LeakyReLU(alpha=0.1)(c4)
    c4 = layers.Conv2D(8, 3, padding='same', use_bias=False)(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.LeakyReLU(alpha=0.1)(c4)

    p = 0.05
    output_bias = tf.keras.initializers.Constant(np.log(p/(1-p)))
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid',
                            bias_initializer=output_bias)(c4)
    return models.Model(inputs, outputs)


def build_improved_segformer(input_shape, paths, config):
    """SegFormer improved with wrapper approach - uses working build_segformer with resizing."""

    # 1) Build the known-good SegFormer that expects 224x224x3
    base = build_segformer((224, 224, 3), paths, config=config)  # <- your existing builder

    # 2) Front-end: channel mix + resize up to 224
    inp = tf.keras.Input(shape=input_shape)
    x = inp
    if input_shape[-1] != 3:
        x = layers.Conv2D(3, 1, padding="same", name="channel_mixer")(x)
    if input_shape[0] != 224 or input_shape[1] != 224:
        x = layers.Resizing(224, 224, interpolation="bilinear", name="resize_to_224")(x)

    # 3) Run base model and resize back to patch size
    logits_224 = base(x)
    out = layers.Resizing(input_shape[0], input_shape[1],
                          interpolation="bilinear",
                          name="resize_back")(logits_224)
    return tf.keras.Model(inp, out, name="segformer_improved")


def build_segformer(input_shape, paths, config):
    """
    SegFormer (B0) for binary segmentation with flexible input size/channels.

    Uses 2-class output to avoid degenerate 1-class softmax, then extracts
    foreground probability for binary segmentation.

    Args:
        input_shape: Input tensor shape
        paths: Platform-specific paths
        config: Configuration dict with seg_head settings
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import keras_cv

    H, W, C = input_shape
    if (H % 32) or (W % 32):
        raise ValueError(f"SegFormer prefers H and W divisible by 32; got {H}x{W}")

    inp = layers.Input(shape=input_shape, name="segformer_input")

    x = inp
    # If not RGB, adapt to 3ch to use pretrained weights
    if C != 3:
        x = layers.Conv2D(3, 1, use_bias=False, padding="same", name="to_rgb3")(x)
        x = layers.BatchNormalization(name="to_rgb3_bn")(x)

    # Your data is [-1,1]; presets expect [0,255]
    x = layers.Rescaling(scale=127.5, offset=127.5, name="to_255")(x)

    # --- Build base backbone without head ---
    base = keras_cv.models.segmentation.SegFormer.from_preset(
        "segformer_b0", num_classes=2
    )
    # Get preset input size (usually 224)
    try:
        h0, w0 = base.input_shape[1:3]
    except Exception:
        h0, w0 = 224, 224

    # Resize to backbone's expected input size
    x_up = layers.Resizing(h0, w0, interpolation="bilinear", name="up_to_preset")(x)
    features = base(x_up)  # Get features from backbone

    # Replace the 2-class softmax head with binary sigmoid head
    # First resize features back if needed
    if features.shape[1] != h0 or features.shape[2] != w0:
        features = layers.Resizing(h0, w0, interpolation="bilinear", name="features_resize")(features)

    # Get bias initialization for imbalanced data
    seg_head_config = config.get('seg_head', {}) if config else {}
    init_logit_bias = seg_head_config.get('init_logit_bias', None)

    if init_logit_bias is not None:
        # Use custom bias initialization
        bias_initializer = tf.keras.initializers.Constant(init_logit_bias)
        print(f"  Using prior-probability bias initialization: {init_logit_bias:.3f}")
    else:
        # Default bias initialization
        bias_initializer = 'zeros'

    # Binary segmentation head with sigmoid and proper bias initialization
    pred_224 = layers.Conv2D(
        1, 1,
        activation='sigmoid',
        bias_initializer=bias_initializer,
        name='binary_head'
    )(features)

    # Resize output back to patch size
    pred = layers.Resizing(H, W, interpolation="bilinear", name="back_to_patch")(pred_224)

    return models.Model(inp, pred, name="segformer_b0_binary")


def build_enhanced_unet(input_shape):
    """Build an enhanced U-Net for segmentation when SegFormer fails."""

    inputs = layers.Input(shape=input_shape)

    # Encoder
    # Block 1
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    c1 = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(c1)

    # Block 2
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    c2 = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(c2)

    # Block 3
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    c3 = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(c3)

    # Block 4
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    c4 = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(c4)

    # Bottleneck
    x = layers.Conv2D(1024, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(1024, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    # Decoder
    # Block 4
    x = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(x)
    x = layers.concatenate([x, c4])
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 3
    x = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)
    x = layers.concatenate([x, c3])
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 2
    x = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x)
    x = layers.concatenate([x, c2])
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 1
    x = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)
    x = layers.concatenate([x, c1])
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Output with bias initialization for class imbalance
    p = 0.05
    output_bias = tf.keras.initializers.Constant(np.log(p / (1 - p)))
    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same',
                           bias_initializer=output_bias)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_resnet50_unet(input_shape, config):
    # This function remains unchanged, but is included for completeness.
    inputs = layers.Input(shape=input_shape)
    if input_shape[2] < 3:
        # If fewer than 3 channels, repeat to get 3 channels
        x = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(inputs)
    elif input_shape[2] > 3:
        # If more than 3 channels, use Conv2D to project down to 3 channels
        # This allows the model to learn optimal channel combination
        x = layers.Conv2D(3, 1, padding='same', use_bias=False, name='channel_adapter')(inputs)
    else:
        x = inputs
    x = layers.Lambda(lambda z: (z + 1.0) / 2.0, name='scale_to_0_1')(x)
    base_model = ResNet50(input_tensor=x, include_top=False, weights='imagenet')

    # Freeze ENTIRE ResNet50 to prevent overfitting - pure feature extractor
    for layer in base_model.layers:
        layer.trainable = False

    s1 = base_model.get_layer('conv1_relu').output
    s2 = base_model.get_layer('conv2_block3_out').output
    s3 = base_model.get_layer('conv3_block4_out').output
    s4 = base_model.get_layer('conv4_block6_out').output
    bridge = base_model.get_layer('conv5_block3_out').output
    u4 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(bridge), s4])
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    u3 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(c4), s3])
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u3)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    u2 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(c3), s2])
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    u1 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(c2), s1])
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    u0 = layers.UpSampling2D(size=(2, 2))(c1)
    c0 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u0)
    c0 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c0)
    p = 0.05
    output_bias = tf.keras.initializers.Constant(np.log(p / (1 - p)))
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', bias_initializer=output_bias)(c0)
    return models.Model(inputs=inputs, outputs=outputs), base_model

def build_unet_heavy(input_shape, config, arch_type='unet_heavy'):
    """
    Heavy U-Net with configurable depth and base filters.
    Default: base=16, depth=3 for patch_size=128
    Can scale to depth=4 with patch_size=256
    """
    inputs = layers.Input(shape=input_shape)

    # Get hyperparameters from config or use defaults based on architecture type
    if arch_type == 'unet_tiny':
        base_filters = 8   # Ultra-lightweight
        depth = 2          # Shallow network
        arch_name = "UNet Tiny"
    elif arch_type == 'unet_medium':
        base_filters = 16  # Moderate capacity
        depth = 3          # Medium depth
        arch_name = "UNet Medium"
    elif arch_type == 'unet_heavy':
        base_filters = 64  # Heavy capacity
        depth = 4          # Deep network
        arch_name = "UNet Heavy"
    elif arch_type == 'unet_ultra_heavy':
        base_filters = 512  # Ultra heavy capacity
        depth = 4          # Deep network
        arch_name = "UNet Ultra Heavy"
    else:
        base_filters = 64  # Default to heavy
        depth = 4
        arch_name = "UNet Heavy (default)"

    if config and 'GLOBAL' in config and 'ARCHITECTURE_HPARAMS' in config['GLOBAL']:
        hparams = config['GLOBAL']['ARCHITECTURE_HPARAMS'].get(arch_type, {})
        base_filters = hparams.get('base_filters', base_filters)
        depth = hparams.get('depth', depth)

    print(f"  {arch_name}: base_filters={base_filters}, depth={depth}")

    # Encoder path
    encoder_outputs = []
    x = inputs

    for level in range(depth):
        filters = base_filters * (2 ** level)

        # Double convolution block
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        # Add dropout for deeper levels
        if level >= depth - 2:
            x = layers.Dropout(0.2)(x)

        # Save for skip connection
        encoder_outputs.append(x)

        # Downsample (except at bottom)
        if level < depth - 1:
            x = layers.MaxPooling2D(2)(x)

    # Bottleneck - concatenate with deepest encoder output first
    # The bottleneck is at the same resolution as the last encoder output
    x = layers.concatenate([x, encoder_outputs[depth - 1]])

    bottleneck_filters = base_filters * (2 ** (depth - 1)) * 2
    x = layers.Conv2D(bottleneck_filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(bottleneck_filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.3)(x)

    # Decoder path - iterate from second deepest to shallowest
    # We already used encoder_outputs[depth-1] in the bottleneck
    for i in range(depth - 1):
        level = depth - 2 - i  # Go from depth-2 down to 0
        filters = base_filters * (2 ** level)

        # Upsample
        x = layers.UpSampling2D(2)(x)

        # Concatenate with skip connection from corresponding encoder level
        x = layers.concatenate([x, encoder_outputs[level]])

        # Double convolution block
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        # Add dropout for regularization
        if level >= depth - 2:
            x = layers.Dropout(0.1)(x)

    # Output layer with bias initialization for class imbalance
    p = 0.01  # Expected positive ratio for ponds
    output_bias = tf.keras.initializers.Constant(np.log(p / (1 - p)))
    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same',
                           bias_initializer=output_bias)(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # Count parameters
    total_params = model.count_params()
    print(f"  Total parameters: {total_params:,}")

    return model


def get_model(config, paths, architecture):
    """
    Get model instance based on configuration.
    """
    task = config['task']  # Direct access - crash if missing
    # Bands are now in TASKS section
    task_config = config['TASKS'][task]
    num_channels = len(task_config['bands'])

    input_shape = (task_config['patch_size'], task_config['patch_size'], num_channels)

    arch = architecture
    print(f"Building model: {arch}...")

    if arch == 'unet':
        model = build_unet(input_shape, config)
    elif arch == 'unet_tiny':
        model = build_unet_heavy(input_shape, config, 'unet_tiny')
    elif arch == 'unet_medium':
        model = build_unet_heavy(input_shape, config, 'unet_medium')
    elif arch == 'unet_heavy':
        model = build_unet_heavy(input_shape, config, 'unet_heavy')
    elif arch == 'unet_ultra_heavy':
        model = build_unet_heavy(input_shape, config, 'unet_ultra_heavy')
    elif arch == 'resnet50_unet':
        model, _ = build_resnet50_unet(input_shape, config)  # Ignore backbone
    elif arch == 'segformer_b0':
        model = build_segformer(input_shape, paths, config)
    elif arch == 'segformer_improved':
        model = build_improved_segformer(input_shape, paths, config)
    else:
        raise ValueError(f"Unknown model architecture: {arch}")

    print(f"Model parameters: {model.count_params():,}")
    return model
