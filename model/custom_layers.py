"""
Custom layers for the CMOSE Experiment 3 concentration detection model.

Model:
CMOSE Temporal Convolution + BiLSTM + Attention + I3D
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate


@tf.keras.utils.register_keras_serializable(package="CMOSE")
class TemporalAttention(tf.keras.layers.Layer):
    """
    Learns attention weights over the temporal BiLSTM outputs.
    """

    def __init__(self, attention_units=128, **kwargs):
        super().__init__(**kwargs)

        self.attention_units = attention_units

        self.W = Dense(
            attention_units,
            activation="tanh"
        )

        self.V = Dense(
            1,
            activation=None
        )

    def call(self, inputs):
        score = self.V(
            self.W(inputs)
        )

        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        context = tf.reduce_sum(
            attention_weights * inputs,
            axis=1
        )

        return context

    def get_config(self):
        config = super().get_config()

        config.update({
            "attention_units": self.attention_units
        })

        return config


@tf.keras.utils.register_keras_serializable(package="CMOSE")
class GatedFusion(tf.keras.layers.Layer):
    """
    Combines OpenFace behavioural features and I3D visual features
    using a learned sigmoid gate.
    """

    def __init__(self, units=256, **kwargs):
        super().__init__(**kwargs)

        self.units = units

        self.openface_projection = Dense(
            units,
            activation="swish"
        )

        self.i3d_projection = Dense(
            units,
            activation="swish"
        )

        self.gate = Dense(
            units,
            activation="sigmoid"
        )

    def call(self, inputs):
        openface_features, i3d_features = inputs

        openface_features = (
            self.openface_projection(
                openface_features
            )
        )

        i3d_features = (
            self.i3d_projection(
                i3d_features
            )
        )

        combined = Concatenate()(
            [
                openface_features,
                i3d_features
            ]
        )

        gate = self.gate(combined)

        fused = (
            gate * openface_features
            + (1.0 - gate) * i3d_features
        )

        return fused

    def get_config(self):
        config = super().get_config()

        config.update({
            "units": self.units
        })

        return config
