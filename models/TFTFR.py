import tensorflow as tf

class TFTFR:
    def __init__(self, n_classes=1):
        # Get model hyperparameters
        self.n_classes = n_classes

    def transformer_encoder(self, inputs):
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        print(x)
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout
            )(x, x)
        x = tf.keras.layers.Dropout(self.dropout)(x) 
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

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
        self.dropout = dropout
        self.head_size =head_size
        self.num_heads=num_heads,
        self.ff_dim = ff_dim,
        self.num_transformer_blocks=num_transformer_blocks,
        self.mpl_units=mpl_units,
        self.dropout=dropout,
        self.mlp_dropout=mlp_dropout

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        print(x)
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x)     
        
        x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)

        for dim in mpl_units:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)
            x = tf.keras.layers.Dropout(self.mlp_dropout)(x)
        
        outputs = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)

        return tf.keras.Model(inputs, outputs)