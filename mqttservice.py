import paho.mqtt.client as mqtt

class MQTTClient:
    def __init__(self, broker_address, port=1883):
        self.client = mqtt.Client()
        self.broker_address = broker_address
        self.port = port
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_publish = self.on_publish
        self.client.on_subscribe = self.on_subscribe
        self.message_payload = None

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker")
        else:
            print("Connection failed with error code " + str(rc))

    def on_message(self, client, userdata, message):
        payload = message.payload.decode('utf-8')  # Mengambil payload pesan dan mengkonversi ke string
        print(f"Received message '{payload}' on topic '{message.topic}'")
        self.message_payload = payload

    def on_publish(self, client, userdata, mid):
        print("Message published (mid: " + str(mid) + ")")

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribed to topic with QoS: " + str(granted_qos))

    def connect(self):
        self.client.connect(self.broker_address, self.port)

    def publish(self, topic, message, qos=0):
        self.client.publish(topic, message, qos=qos)
        print("{topic} succes".format(topic=topic))  
        
    def subscribe(self, topic, qos=0):
        self.client.subscribe(topic, qos=qos)
        print("Subscribed to topic '" + topic + "'")

    def unsubscribe(self, topic):
        self.client.unsubscribe(topic)
        print("Unsubscribed from topic '" + topic + "'")
        
    def getPayload(self):
        return self.message_payload
    
    def disconnect(self):
        self.client.disconnect()
        self.client.loop_stop()
        print("Disconnected from MQTT broker")

