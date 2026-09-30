"""
CMOSE Experiment 3:
Temporal Convolution + BiLSTM + Attention + I3D + Gated Fusion.

This architecture classifies student concentration into:
0 - Distracted
1 - Partially Focused
2 - Focused
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    Concatenate,
    LayerNormalization,
    Dropout,
    Bidirectional,
    LSTM,
)
from tensorflow.keras.models import Model

from model.custom_layers import TemporalAttention, GatedFusion


def build_cmose_experiment3_model():
    """
    Build the multimodal CMOSE Experiment 3 concentration detection model.

    Inputs:
        OpenFace behavioural sequence: shape (120, 25)
        I3D visual embedding: shape (1024,)

    Returns:
        TensorFlow/Keras model with a 3-class softmax output.
    """

    # ==========================================================
    # OPENFACE BEHAVIOURAL BRANCH
    # ==========================================================

    openface_input = Input(
        shape=(120, 25),
        name="openface_input"
    )

    x = LayerNormalization(
        name="openface_input_norm"
    )(openface_input)

    # Multi-scale temporal convolution
    conv_3 = Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="swish",
        name="temporal_conv_3"
    )(x)

    conv_5 = Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="swish",
        name="temporal_conv_5"
    )(x)

    conv_7 = Conv1D(
        filters=64,
        kernel_size=7,
        padding="same",
        activation="swish",
        name="temporal_conv_7"
    )(x)

    x = Concatenate(
        name="multiscale_temporal_features"
    )([
        conv_3,
        conv_5,
        conv_7
    ])

    x = LayerNormalization(
        name="temporal_conv_norm"
    )(x)

    x = Dropout(
        0.15,
        name="temporal_conv_dropout"
    )(x)

    x = Dense(
        160,
        activation="swish",
        name="temporal_projection"
    )(x)

    # First BiLSTM
    x = Bidirectional(
        LSTM(
            128,
            return_sequences=True,
            dropout=0.20,
            recurrent_dropout=0.0
        ),
        name="bilstm_1"
    )(x)

    x = LayerNormalization(
        name="bilstm_1_norm"
    )(x)

    # Second BiLSTM
    x = Bidirectional(
        LSTM(
            96,
            return_sequences=True,
            dropout=0.20,
            recurrent_dropout=0.0
        ),
        name="bilstm_2"
    )(x)

    x = LayerNormalization(
        name="bilstm_2_norm"
    )(x)

    # Temporal attention
    openface_features = TemporalAttention(
        attention_units=128,
        name="openface_temporal_attention"
    )(x)

    openface_features = Dense(
        192,
        activation="swish",
        name="openface_dense"
    )(openface_features)

    openface_features = Dropout(
        0.25,
        name="openface_dropout"
    )(openface_features)

    # ==========================================================
    # I3D VISUAL BRANCH
    # ==========================================================

    i3d_input = Input(
        shape=(1024,),
        name="i3d_input"
    )

    i = LayerNormalization(
        name="i3d_input_norm"
    )(i3d_input)

    i = Dense(
        512,
        activation="swish",
        name="i3d_dense_1"
    )(i)

    i = Dropout(
        0.25,
        name="i3d_dropout_1"
    )(i)

    i = Dense(
        256,
        activation="swish",
        name="i3d_dense_2"
    )(i)

    i = LayerNormalization(
        name="i3d_dense_2_norm"
    )(i)

    i = Dropout(
        0.20,
        name="i3d_dropout_2"
    )(i)

    i3d_features = Dense(
        192,
        activation="swish",
        name="i3d_features"
    )(i)

    # ==========================================================
    # GATED MULTIMODAL FUSION AND CLASSIFICATION
    # ==========================================================

    fused = GatedFusion(
        units=256,
        name="gated_multimodal_fusion"
    )([
        openface_features,
        i3d_features
    ])

    fused = LayerNormalization(
        name="fusion_norm"
    )(fused)

    fused = Dense(
        256,
        activation="swish",
        name="fusion_dense_1"
    )(fused)

    fused = Dropout(
        0.35,
        name="fusion_dropout_1"
    )(fused)

    fused = Dense(
        128,
        activation="swish",
        name="fusion_dense_2"
    )(fused)

    fused = Dropout(
        0.25,
        name="fusion_dropout_2"
    )(fused)

    output = Dense(
        3,
        activation="softmax",
        name="concentration_output"
    )(fused)

    model = Model(
        inputs=[
            openface_input,
            i3d_input
        ],
        outputs=output,
        name="CMOSE_TemporalConv_BiLSTM_GatedFusion"
    )

    return model


if __name__ == "__main__":
    model = build_cmose_experiment3_model()
    model.summary()
