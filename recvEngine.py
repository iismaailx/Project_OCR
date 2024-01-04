import cv2
import base64
import time
import numpy as np
from json import dumps
from mqttservice import MQTTClient
from PIL import Image
import threading

def monitordevice(client):
    while True:
        command = input("Masukan command : ")
        client.publish("nutech/ocr/command", command, qos=2)
        print("Command : {} terkirim!".format(command))

def main(client):
    def process_message(client, userdata, message):
        payload = message.payload.decode('utf-8')
        print(f"received message: {payload}")
        try:
            if message.topic == "nutech/ocr/data":
                status_payload = payload
                print(status_payload)
            if message.topic == "nutech/ocr/tempCPU":
                dataTemp_payload = payload
                print(dataTemp_payload)
        except Exception as e:
            print("This Error Message : {}".format(str(e)))
                    
    client.client.on_message = process_message
    client.connect()
    client.subscribe("nutech/ocr/data", qos=2)
    client.subscribe("nutech/ocr/tempCPU", qos=2)
    
    try:
        client.client.loop_forever()
    except KeyboardInterrupt:
        client.disconnect()

if __name__ == "__main__":
    client = MQTTClient('ocr.local', 1883)
    main_thread = threading.Thread(target=main, args=(client,))
    main_thread.start()
    monitor_thread = threading.Thread(target=monitordevice, args=(client,))
    monitor_thread.start()
