import tensorflow as tf

# Load the existing model
model = tf.keras.models.load_model("D:/ai_vision/TeachableMachine/keras_model.h5")

# Save the model in the SavedModel format
model.save("D:/ai_vision/TeachableMachine/saved_model")
