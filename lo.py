import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

from keras.preprocessing import image
# Load the trained model
model = load_model('intel_images_classification.h5')  # Load your trained model

# Load and preprocess the image for prediction
img_path = '22.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image data

# Make a prediction
predictions = model.predict(img_array)

# Get the class with the highest probability
predicted_class = np.argmax(predictions)

# Print the predicted class and its probability
print("Predicted class:", predicted_class)
print("Predicted probability:", predictions[0][predicted_class])
