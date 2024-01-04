import cv2
import base64
import time
import threading
import numpy as np
import tensorflow as tf
# import ocr as OCR  #==> using pyc format file.
import imp
from json import dumps, loads
from picamera2 import Picamera2
from PIL import Image
from io import BytesIO
from defisheye import Defisheye
from gpiozero import CPUTemperature


MQTT = imp.load_compiled("mqttservice", "/home/rnd/Development/FINAL/run/mqttservice.cpython-39.pyc")
OCR = imp.load_compiled("ocr", "/home/rnd/Development/FINAL/run/ocr.cpython-39.pyc")

class ImageCameraProcesing:
    def __init__(self, model_path):
        self.cam = Picamera2()
        self.File = "/home/rnd/Development/FINAL/images/image.jpg"
        self.resolution = (3280, 2464)
        self.format = "RGB888"
        self.gambar = None
        self.imageocr = None
        self.fishEye = "/home/rnd/Development/FINAL/fisheye/result_fishEye.jpg"
        self.zoom = "/home/rnd/Development/FINAL/images/cropped_image.jpg"
        self.config = self.cam.create_still_configuration(main={"size":(self.resolution), "format":self.format},
                                                          raw={"size":self.cam.sensor_resolution})
        self.cam.configure(self.config)
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.dtype = 'linear'
        self.format = 'fullframe'
        self.fov = 100
        self.pfov = 90
        
    def capture(self):
        self.cam.start()
        time.sleep(1)
        self.gambar = self.cam.capture_image("main")
        print("get capture!")
        # self.gambar.save("images/imagenotrotate.jpg")
        # self.gambar = self.gambar.rotate(90, expand=False)
        self.gambar.save(self.File) 
        time.sleep(1/100)
        
        
    def convertfishEye(self):
        obj = Defisheye(self.File, dtype=self.dtype, format=self.format, fov=self.fov, pfov=self.pfov)
        obj.convert(outfile=self.fishEye)
        time.sleep(1/10000)
        # print("done fish!")
        
    def zoom_in(self):
        new_image = cv2.imread(self.fishEye)
        width, height, _ = new_image.shape
        top_crop = int(height * 0.15)  # 10% dari bagian atas
        bottom_crop = height - top_crop  # 10% dari bagian bawah
        left_crop = int(width * 0.25)  # 20% dari sisi kiri
        right_crop = width - left_crop  # 20% dari sisi kanan
        # Pemotongan gambar atas dan bawah
        cropped_image = new_image[top_crop:bottom_crop, left_crop:right_crop]
        rotated_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(self.zoom, rotated_image)
        # print("get zoom!")

    def convert(self):
        image = cv2.imread(self.zoom)
        _, img_encoded = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(img_encoded).decode()
        return image_base64
            
    def readocr(self):
        data_ktp = OCR.main(self.zoom)
        if data_ktp is not None:
            return data_ktp
            
    def classify(self):
        if self.gambar is not None:
            image = cv2.imread(self.zoom)
            image = cv2.resize(image, (150, 150))
            input_data = np.array(image) / 255.0  # Normalize to [0, 1]
            input_data = input_data.reshape((1, 150, 150, 3)) 
            input_data = input_data.astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            threshold = 0.5
            predicted_class = 0 if predictions[0] > threshold else 1
            kelas = [0, 1]
            predicted_label = kelas[predicted_class]
            print("Predicted Label: {}, TensorValue: {}".format(predicted_label, predictions))
            return predicted_label

class ExtractFromUpload:
    def __init__(self, model_path) -> None:
        self.upload = None
        self.get_image = None
        self.path = "/home/rnd/Development/FINAL/images/from_upload.jpg"
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
            
    def ExtractConvertoImage(self, file):
        _base64data = base64.b64decode(file)
        _stream = BytesIO(_base64data)
        self.get_image = Image.open(_stream)
        self.get_image.save(self.path)
        
    def ExtractClasifyImage(self): #not use for classify
        if self.get_image is not None:
            # print(self.image.shape)
            image = cv2.imread(self.path)
            image = cv2.resize(image, (150, 150))  # Perbaikan pada baris ini, mengubah dari image(150, 150) menjadi cv2.resize(image, (150, 150))
            input_data = np.array(image) / 255.0  # Normalize to [0, 1]
            input_data = input_data.reshape((1, 150, 150, 3))
            input_data = input_data.astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            threshold = 1
            predicted_class = 0 if predictions[0] >= threshold else 1 # perbandingannya diubah
            kelas = [0, 1]
            predicted_label = kelas[predicted_class]
            print("Predicted Label: {}, TensorValue: {}".format(predicted_label, predictions))
            return predicted_label
    
    def ExtractReadOCR(self):
        if self.get_image is not None:
            data_ktp = OCR.main(self.path)
            if data_ktp is not None:
                return data_ktp   
                   
def main(client, camera, upload):
    def process_message(client, userdata, message):
        data_ktp = ""
        dataBase64 = ""
        id = 0
        response={
            "id":"-", "kode":"-", "status":"-", "message":"-",
            "length":"-", "time":"-","database64":"-"
        }
        start_time = time.time()
        
        
        try:     
            if message.topic == "nutech/ocr/command":
                command = message.payload.decode('utf-8')
                print(f"Received message: {command}")
                if command == "1":
                    print("Processing Command...")
                    camera.capture()
                    camera.convertfishEye()
                    camera.zoom_in()
                    dataBase64 = camera.convert()
                    predict = camera.classify()
                    if predict == 1:
                        data_ktp = camera.readocr()
                        response["kode"] = "200"
                        response["status"] = "Successfully"
                        response["id"] = str(id)
                        id += 1
                    else:
                        # data_ktp = camera.readocr()
                        response["kode"] = "400"
                        response["status"] = "Invalid ID card picture, please try again"
                        
                response["database64"] = dataBase64
                response["length"] = str(len(dataBase64))
                                     
            if message.topic == "nutech/ocr/upload":
                _file = message.payload.decode('utf-8')
                print("Processing Upload...")
                json_data = loads(_file)
                base64_string = json_data.get('data', '')
                # print(base64_string)
                upload.ExtractConvertoImage(base64_string)
                data_ktp = upload.ExtractReadOCR()
                response["kode"] = "200"
                response["status"] = "Successfully"
                response["id"] = str(id)
                id += 1  
                response["database64"] = "none"
                response["length"] = "none"
                              
        except Exception as e:
            response["kode"] = "400"
            response["status"] = "There Something Wrong on System"
            response["message"] = str(e)
            print(f"this error : {str(e)}")
                 
        end_time = time.time()
        elapsed_time = end_time - start_time      
        response["time"] = "{:.2f}s".format(elapsed_time)
        
        if data_ktp != None:
            ktp_response = dumps(data_ktp)
            client.publish("nutech/ocr/ktp", ktp_response, qos=2)
            print("nutech/ocr/ktp, success")
        
        if response != None:
            send_response = dumps(response)
            client.publish("nutech/ocr/data", send_response, qos=2)
            print("nutech/ocr/data, success")

        print(f"time to process: {elapsed_time}\n")
        print("Processing complete.....")
        print("-"*30)
                
    client.client.on_message = process_message
    client.connect()
    client.subscribe("nutech/ocr/command", qos=2)
    client.subscribe("nutech/ocr/upload", qos=2)

    try:
        client.client.loop_forever()
    except KeyboardInterrupt:
        client.disconnect()

def monitordevice(client):
    temp = 0
    while True:
        cpu = CPUTemperature()
        temp = (cpu.temperature)
        client.publish("nutech/ocr/tempCPU", str(temp))
        time.sleep(15)
        
if __name__ == "__main__":
    client = MQTT.MQTTClient('ocr.local', 1883)
    model_path="/home/rnd/Development/FINAL/models/ktp.tflite"
    camera = ImageCameraProcesing(model_path)
    upload = ExtractFromUpload(model_path)
    main_thread = threading.Thread(target=main, args=(client, camera, upload))
    main_thread.start()
    monitor_thread = threading.Thread(target=monitordevice, args=(client,))
    monitor_thread.start()

