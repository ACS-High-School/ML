import tensorflow as tf


class MLMODEL:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(input_shape=[12]),
            tf.keras.layers.Dense(128, activation='relu', use_bias=True),
            tf.keras.layers.Dense(64, activation='leaky_relu', use_bias=True),
            tf.keras.layers.Dense(32, activation='relu', use_bias=True),
            tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True)
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(),
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
        )

    def getModel(self):
        return self.model