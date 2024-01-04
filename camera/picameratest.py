# from picamera2 import Picamera2
# picam2 = Picamera2()
# picam2.start_and_capture_file("images/test.jpg")
# from picamera2 import Picamera2, Preview
# import time
# import logging
# picam2 = Picamera2(verbose_console=0)
# camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
# picam2.configure(camera_config)
# # picam2.start_preview(Preview.QTGL)
# # picam2.start()
# # time.sleep(2)
# picam2.start_and_capture_file("images/test.jpg")
# Picamera2.set_logging(Picamera2.ERROR)

# from picamera2 import Picamera2
# import time
# picam2 = Picamera2()
# picam2.start()
# time.sleep(1)
# array = picam2.capture_array("main")
# print(type(array))
# print(array)

from picamera2 import Picamera2
import time
while True:
    command = input("masukan command: ")
    picam2 = Picamera2()
    if command == "1":
        picam2.start()
        time.sleep(1)
        image = picam2.capture_image("main")
        print(type(image))
    if command == "2":
        break