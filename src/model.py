import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="LeafDocNet")
class AttentiveTransition(tf.keras.layers.Layer):
    """DenseNet attentive transition described in LeafDoc-Net.

    Channel weights are generated from global average pooled features,
    applied to the DenseNet feature map, refined by a 1x1 convolution,
    then spatially reduced with 2x2 max pooling.
    """

    def __init__(self, attention_filters: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.attention_filters = attention_filters
        self.channel_dense = tf.keras.layers.Dense(attention_filters)
        self.channel_sigmoid = tf.keras.layers.Activation("sigmoid")
        self.feature_conv = tf.keras.layers.Conv2D(
            attention_filters,
            kernel_size=1,
            padding="same",
        )
        self.pool = tf.keras.layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding="valid",
        )

    def build(self, input_shape):
        channels = int(input_shape[-1])
        if channels != self.attention_filters:
            raise ValueError(
                "DenseNet attentive transition requires "
                f"{self.attention_filters} input channels, received {channels}."
            )
        self.channel_dense.build((None, channels))
        self.feature_conv.build((None, None, None, channels))
        super().build(input_shape)

    def call(self, inputs, training=None):
        del training
        attention = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        attention = self.channel_dense(attention)
        attention = self.channel_sigmoid(attention)
        attention = tf.keras.layers.Reshape((1, 1, self.attention_filters))(attention)
        attended = inputs * attention
        refined = self.feature_conv(attended)
        return self.pool(refined)

    def get_config(self):
        config = super().get_config()
        config.update({"attention_filters": self.attention_filters})
        return config


@tf.keras.utils.register_keras_serializable(package="LeafDocNet")
class MobileAttention(tf.keras.layers.Layer):
    """Spatial channel attention for the MobileNetV2 branch.

    A paper-style 1x1 convolution produces one sigmoid weight per MobileNetV2
    feature channel. The weights are multiplied by the original feature maps.
    """

    def __init__(self, filters: int = 1280, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.attention_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            padding="same",
        )
        self.attention_sigmoid = tf.keras.layers.Activation("sigmoid")

    def build(self, input_shape):
        channels = int(input_shape[-1])
        if channels != self.filters:
            raise ValueError(
                "MobileNet attention requires "
                f"{self.filters} input channels, received {channels}."
            )
        self.attention_conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        del training
        weights = self.attention_conv(inputs)
        weights = self.attention_sigmoid(weights)
        return inputs * weights

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


def create_mobilenetv2_model(
    num_classes: int,
    image_size: tuple[int, int],
    weights: str | None = "imagenet",
    dropout_rate: float = 0.25,
) -> tf.keras.Model:
    """Create the transfer-learning model.

    MobileNetV2 supplies ImageNet features; only the custom head is trained
    during the initial phase. The input layer receives raw 0-255 pixels and
    MobileNetV2's preprocessing function is applied once inside the graph.
    """
    if num_classes < 2:
        raise ValueError("At least two classes are required.")

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(image_size[0], image_size[1], 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="plant_disease_mobilenetv2")


def enable_fine_tuning(
    model: tf.keras.Model,
    trainable_last_layers: int = 20,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """Unfreeze selected upper base layers using a small learning rate.

    BatchNormalization layers remain frozen because updating their running
    statistics with a small fine-tuning dataset can harm pretrained features.
    """
    base_model = next(
        (layer for layer in model.layers if isinstance(layer, tf.keras.Model)),
        None,
    )
    if base_model is None:
        raise ValueError("Could not locate the MobileNetV2 base model.")

    base_model.trainable = True
    for layer in base_model.layers[:-trainable_last_layers]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_leafdoc_net_model(
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    weights: str | None = "imagenet",
    freeze_backbones: bool = True,
    dropout_rate: float = 0.25,
) -> tf.keras.Model:
    """Create the two-backbone LeafDoc-Net-inspired hybrid architecture.

    Assumptions for details not fully specified by the paper:
    * The DenseNet 128-layer uses Swish plus BatchNormalization, consistent
      with the stated 1024/512 dense blocks.
    * The DenseNet attentive transition refines features with a 1024-channel
      1x1 convolution and 2x2 max pooling.
    * The MobileNet branch generates sigmoid attention over all 1280 output
      channels, then applies global average pooling.
    * Dropout is not a replacement for paper attention; it is limited to the
      dense classifier only.
    """
    if num_classes < 2:
        raise ValueError("At least two classes are required.")
    if image_size[0] != 224 or image_size[1] != 224:
        raise ValueError("LeafDoc-Net uses the documented 224x224 input.")

    inputs = tf.keras.Input(
        shape=(image_size[0], image_size[1], 3),
        name="leaf_image",
    )

    densenet = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=(224, 224, 3),
        name="densenet121_base",
    )
    densenet.trainable = not freeze_backbones

    mobilenet = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(224, 224, 3),
        name="mobilenetv2_base",
    )
    mobilenet.trainable = not freeze_backbones

    dense_input = tf.keras.applications.densenet.preprocess_input(inputs)
    dense_features = densenet(dense_input, training=None)
    dense_features = AttentiveTransition(
        attention_filters=1024,
        name="densenet_attentive_transition",
    )(dense_features)
    dense_features = tf.keras.layers.GlobalAveragePooling2D(
        name="densenet_global_average_pool"
    )(dense_features)

    mobile_input = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    mobile_features = mobilenet(mobile_input, training=None)
    mobile_features = MobileAttention(
        filters=1280,
        name="mobilenet_sigmoid_attention",
    )(mobile_features)
    mobile_features = tf.keras.layers.GlobalAveragePooling2D(
        name="mobilenet_global_average_pool"
    )(mobile_features)

    fused = tf.keras.layers.Concatenate(
        name="leafdoc_feature_fusion"
    )([dense_features, mobile_features])

    x = tf.keras.layers.Dense(1024, name="dense_1024")(fused)
    x = tf.keras.layers.Activation("swish", name="swish_1024")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm_1024")(x)
    x = tf.keras.layers.Dense(512, name="dense_512")(x)
    x = tf.keras.layers.Activation("swish", name="swish_512")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm_512")(x)
    x = tf.keras.layers.Dense(128, name="dense_128")(x)
    x = tf.keras.layers.Activation("swish", name="swish_128")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm_128")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="leafdoc_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="leafdoc_net",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def enable_leafdoc_fine_tuning(
    model: tf.keras.Model,
    densenet_last_layers: int = 30,
    mobilenet_last_layers: int = 20,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """Fine-tune upper backbone layers while keeping BatchNormalization frozen."""
    for base_name in ("densenet121_base", "mobilenetv2_base"):
        base_model = next(
            (
                layer
                for layer in model.layers
                if isinstance(layer, tf.keras.Model)
                and layer.name == base_name
            ),
            None,
        )
        if base_model is None:
            raise ValueError(f"Could not locate the {base_name} backbone.")

        last_layers = (
            densenet_last_layers
            if base_name == "densenet121_base"
            else mobilenet_last_layers
        )
        base_model.trainable = True
        for layer in base_model.layers[:-last_layers]:
            layer.trainable = False
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_leafdoc_model_report(model: tf.keras.Model) -> dict:
    """Return model parameter and trainability metadata."""
    total = int(model.count_params())
    trainable = int(
        sum(tf.keras.backend.count_params(variable) for variable in model.trainable_weights)
    )
    non_trainable = total - trainable
    return {
        "total_parameters": int(model.count_params()),
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
    }


def create_mobilenetv2_baseline(
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """MobileNetV2-only comparison baseline."""
    inputs = tf.keras.Input(shape=(*image_size, 3), name="baseline_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(*image_size, 3),
        name="mobilenetv2_baseline",
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_baseline")


def create_densenet121_baseline(
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """DenseNet121-only comparison baseline."""
    inputs = tf.keras.Input(shape=(*image_size, 3), name="baseline_image")
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=(*image_size, 3),
        name="densenet121_baseline",
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="densenet121_baseline")
