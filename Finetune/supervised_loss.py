import tensorflow.compat.v2 as tf

def supervised_loss(labels, logits):
    """Compute mean supervised loss over local batch."""
    losses = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)

    return tf.reduce_mean(losses)