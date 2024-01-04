import tensorflow as tf
import numpy as np
import sys
from PIL import Image

model_path="models/ktp.tflite"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def classify(file):
    try:
        input_data = np.array(Image.open(file))
        input_data = np.array(file) / 255.0  # Normalize to [0, 1]
        input_data = input_data.reshape((1, 150, 150, 3)) 
        input_data = input_data.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        threshold = 0.5
        predicted_class = 0 if predictions[0] > threshold else 1
        kelas = ['0', '1']
        predicted_label = kelas[predicted_class]
        print("Predicted Label: {}, TensorValue: {}".format(predicted_label, predictions))
        print(predictions)
        return predicted_label
    except Exception as e:
        print(e)

if __name__ == "__main__":
    classify(sys.argv[1])
