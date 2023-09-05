
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

input_shape = (224, 224)  
num_classes = 67  



class indoor_class:
    def __init__(self,filename):
        self.filename =filename
    
    def preprocess_image(self,image_path):
        img = load_img(image_path, target_size=input_shape)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image pixels between 0 and 1
        return img_array
   


    def predictiondogcat(self):
        
        saved_model_path = "intel_images_classification.h5"
        model = load_model(saved_model_path)
        # model.summary()
        imagename = self.filename

        sample_image = load_img(imagename, target_size=(224, 224))
        sample_image_array = img_to_array(sample_image)
        sample_image_array = preprocess_input(sample_image_array)
        sample_image_array = np.expand_dims(sample_image_array, axis=0)
        predicted_class_probabilities = model.predict(sample_image_array)
        print(predicted_class_probabilities)
        
        predicted_class_index = np.argmax(predicted_class_probabilities)

        # List of class names (replace with your actual class names)
        class_names = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']

        # Get the predicted class label
        predicted_class_label = class_names[predicted_class_index]
        max_prob=round(np.max(predicted_class_probabilities),2)
        print(f'Predicted class probabilities: {max_prob}')
        print(f'Predicted class label: {predicted_class_label}')


        # single_image = self.preprocess_image(imagename)
        # prediction = model.predict(single_image)
        # prediction_scores = predicted_class_probabilities[0][0]
        


        # predicted_class_index = np.argmax(prediction)
        # class_mapping = {}
        # with open("class_mapping.txt", "r") as file:
        #     for line in file:
        #         idx, label = line.strip().split()
        #         class_mapping[int(idx)] = label

        # predicted_class_label = class_mapping[predicted_class_index]

        # print(f"Predicted Class Label: {predicted_class_label}")
        # print(prediction_scores)
        result={"prediction_scores":str(max_prob*100),
                "predicted_class_label":predicted_class_label
                }

        return result







       


