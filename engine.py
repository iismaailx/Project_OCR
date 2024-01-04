import cv2
import base64
import time
import ocr
import threading
import numpy as np
import tensorflow as tf
from json import dumps
from mqttservice import MQTTClient
from picamera2 import Picamera2
from PIL import Image
from gpiozero import CPUTemperature
#engine

class ImageProcesing:
    def __init__(self, model_path):
        self.cam = Picamera2()
        self.File = "images/image.jpg"
        self.resolution = (1920, 1080)
        self.format = "RGB888"
        self.gambar = None
        self.json_file = None
        self.config = self.cam.create_still_configuration(main={"size":(self.resolution), "format":self.format},
                                                          raw={"size":self.cam.sensor_resolution})
        self.cam.configure(self.config)
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def capture(self):
        self.cam.start()
        time.sleep(1)
        self.gambar = self.cam.capture_image("main")
        print("get capture!")
        # self.gambar.save(self.File) 
        time.sleep(1)

    def convert(self):
        if self.gambar is not None:
            print("get convert!")
            image = np.array(self.gambar)
            resized_image = cv2.resize(image, (640, 480))
            resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            _, img_encoded = cv2.imencode('.jpg', resized_image_rgb)
            image_base64 = base64.b64encode(img_encoded).decode()
            return image_base64
        else:
            print("gambar is None")
            
    def readocr(self):
        if self.gambar is not None:
            data_ktp = ocr.main(self.gambar)
            print(f"ini ktp {data_ktp}")
            if data_ktp is not None:
                return data_ktp
            
    def classify(self):
        if self.gambar is not None:
            image = self.gambar.resize((150, 150))
            input_data = np.array(image) / 255.0  # Normalize to [0, 1]
            input_data = input_data.reshape((1, 150, 150, 3)) 
            input_data = input_data.astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            threshold = 0.5
            predicted_class = 0 if predictions[0] > threshold else 1
            kelas = ['0', '1']
            predicted_label = kelas[predicted_class]
            print("Predicted Label: {}, TensorValue: {}".format(predicted_label, predictions))
            return predicted_label
        
def monitordevice(client):
    temp = 0
    while True:
        cpu = CPUTemperature()
        temp = (cpu.temperature)
        client.publish("nutech/ocr/tempCPU", str(temp))
        time.sleep(60)
          
def main(client, image):
    def process_message(client, userdata, message):
        dataKTP = None
        dataBase64 = None
        id = 0
        response={
            "id":"","kode":"", "status":"", "message":"", "ktp":"",
            "length":"", "time":"","database64":""
        }
        start_time = time.time()
        payload = message.payload.decode('utf-8')
        print(f"Received message: {payload}")
        
        if payload == "1":
            print("Processing command...")
            try:
                image.capture()
                dataBase64 = image.convert()
                predict = image.classify()
                if predict == "1":
                    dataKTP = image.readocr()
                    response["kode"] = "200"
                    response["status"] = "success"
                    response["message"] = predict
                    response["dataktp"] = dataKTP
                else:
                    response["kode"] = "200"
                    response["status"] = "not success"
                    response["message"] = predict             
            except Exception as e:
                response["kode"] = "400"
                response["status"] = "error"
                response["message"] = str(e)
                
        end_time = time.time()
        elapsed_time = end_time - start_time
        response["id"] = id
        response["length"] = str(len(dataBase64))
        response["time"] = str(elapsed_time)
        response["database64"] = dataBase64
        send_response = dumps(response)
        client.publish("nutech/ocr/data", send_response, qos=2)
        print("Processing complete.....")
        print("nutech/ocr/data, success")
        print(f"time to process: {elapsed_time}\n")
        id += 1
              
    client.client.on_message = process_message
    client.connect()
    client.subscribe("nutech/ocr/command", qos=2)

    try:
        client.client.loop_forever()
    except KeyboardInterrupt:
        client.disconnect()

if __name__ == "__main__":
    client = MQTTClient('ocr.local', 1883)
    model_path="/home/rnd/Development/OCR_RASPI/models/ktp.tflite"
    image = ImageProcesing(model_path)
    main_thread = threading.Thread(target=main, args=(client, image))
    main_thread.start()
    monitor_thread = threading.Thread(target=monitordevice, args=(client,))
    monitor_thread.start()

