import tensorflow as tf

class TFTransformer:
    def __init__(self, n_classes=1):
        # Get model hyperparameters
        self.n_classes = n_classes

    def build_model(self,
                    input_shape,
                    num_heads=4,
                    d_model=128,
                    num_layers=4,
                    dff=512,
                    dropout=0.1):
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_layers = num_layers
        self.dff = dff
        self.dropout = dropout

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        # Positional encoding is typically added to the input
        # Alternatively, you can use an Embedding layer for positional encodings
        positional_encoding = self._get_positional_encoding(input_shape, self.d_model)
        x = x + positional_encoding

        for _ in range(self.num_layers):
            # Multi-Head Self-Attention Layer
            x = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)([x, x])
            x = tf.keras.layers.Dropout(self.dropout)(x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)

            # Feed Forward Neural Network
            ffn_output = tf.keras.layers.Dense(self.dff, activation='relu')(x)
            ffn_output = tf.keras.layers.Dense(self.d_model)(ffn_output)
            ffn_output = tf.keras.layers.Dropout(self.dropout)(ffn_output)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output + x)

        # Output layer
        outputs = tf.keras.layers.Dense(self.n_classes)(x)

        return tf.keras.Model(inputs, outputs)

    def _get_positional_encoding(self, input_shape, d_model):
        position = tf.range(0, input_shape[1], dtype=tf.float32)
        position = position / tf.math.pow(10000, 2 * tf.range(0, d_model, 2, dtype=tf.float32) / d_model)
        sine_wave = tf.math.sin(position)
        cosine_wave = tf.math.cos(position)
        positional_encoding = tf.concat([sine_wave, cosine_wave], axis=-1)
        return tf.expand_dims(positional_encoding, axis=0)
