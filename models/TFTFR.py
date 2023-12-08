import tensorflow as tf
import numpy as np

class TFTFR:
    def __init__(self, n_classes=1):
        # Get model hyperparameters
        self.n_classes = n_classes

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        """
        Creates positional encoding matrix.
        """
        
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def feed_forward(self, inputs):
        """
        Creates feed forward block for transformer encoder.
        """
        seq = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation='relu'),
            tf.keras.layers.Dense(inputs.shape[-1]),
            tf.keras.layers.Dropout(self.dropout)
        ])

        add = tf.keras.layers.Add()
        layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        x = add([inputs, seq(inputs)])
        x = layer_norm(x)

        return x

    def transformer_encoder(self, inputs):
        # Multi-Head Attention Block
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)(x, x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        res = x + inputs

        # Feed Forward Block
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = self.feed_forward(x)
        res = x + res

        return res

    def build_model(self,
                    input_shape,
                    head_size=128,
                    num_heads=4,
                    ff_dim=2,
                    num_transformer_blocks=4,
                    mpl_units=[256],
                    dropout=0.25,
                    mlp_dropout=0.4,
                    ):

        """
        Creates final model transformer encoder block.
        """
        self.dropout = dropout
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mpl_units = mpl_units
        self.mlp_dropout = mlp_dropout

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        # Positional Encoding
        pos_encoding = self.positional_encoding(input_shape[0], input_shape[1])
        x = x + pos_encoding


        for _ in range(num_transformer_blocks):
          x = self.transformer_encoder(x)

        x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)

        for dim in mpl_units:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)
            x = tf.keras.layers.Dropout(self.mlp_dropout)(x)

        outputs = tf.keras.layers.Dense(self.n_classes)(x)

        return tf.keras.Model(inputs, outputs)