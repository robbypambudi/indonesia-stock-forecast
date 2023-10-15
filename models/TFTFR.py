import tensorflow as tf

class TFTFR:
    def __init__(self, n_classes=1):
        # Get model hyperparameters
        self.n_classes = n_classes

    # def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0, epsilon=1e-6, attention_axes=None, kernel_size=1):
    #     x = tf.keras.layers.LayerNormalization(epsilon=epsilon)(inputs)
    #     x = tf.keras.layers.MultiHeadAttention(
    #         key_dim=head_size, num_heads=num_heads, dropout=dropout,
    #         attention_axes=attention_axes)(x, x)
    #     x = tf.keras.layers.Dropout(dropout)(x) 
    #     res = x + inputs

    #     # Feed Forward Part
    #     x = tf.keras.layers.LayerNormalization(epsilon=epsilon)(res)
    #     x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=kernel_size, activation='relu')(x)
    #     x = tf.keras.layers.Dropout(dropout)(x)
    #     x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=kernel_size)(x)
    #     return x + res

    def build_model(self,
                    input_shape,
                    head_size,
                    num_heads,
                    ff_dim,
                    num_transformer_blocks,
                    mpl_units,
                    dropout=0,
                    mlp_dropout=0,
                    epsilon=1e-6,
                    attention_axes=None,
                    kernel_size=1):

        """
        Creates final model transformer encoder block.
        """
        self.dropout = dropout

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        for _ in range(num_transformer_blocks):
            x = tf.keras.layers.LayerNormalization(epsilon=epsilon)(inputs)
            x = tf.keras.layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout,
                attention_axes=attention_axes)(x, x)
            x = tf.keras.layers.Dropout(dropout)(x) 
            res = x + inputs

            # Feed Forward Part
            x = tf.keras.layers.LayerNormalization(epsilon=epsilon)(res)
            x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=kernel_size, activation='relu')(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=kernel_size)(x)
            x = x + res        
        
        x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)

        for dim in mpl_units:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        
        outputs = tf.keras.layers.Dense(self.n_classes, activation='sigmoid')(x)

        return tf.keras.Model(inputs, outputs)