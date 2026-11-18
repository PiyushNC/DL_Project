import tensorflow as tf


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
