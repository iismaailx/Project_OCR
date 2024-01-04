import cv2
from ultralytics import YOLO

# Load model kustom yang telah dilatih
model = YOLO('models/yolov8n.pt')  # Ganti 'custom_model.pt' dengan nama file model Anda

# Baca gambar untuk deteksi objek
img = cv2.imread('images/R.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Lakukan prediksi menggunakan model
results = model.predict(img)

# Tampilkan hasil deteksi objek pada gambar
for r in results:
    for box in r.boxes:
        b = box.xyxy[0]
        c = int(box.cls)
        label = f"Class: {c} - Confidence: {float(box.conf):.2f}"
        print(label)
        # Tambahkan logika untuk menandai atau menggambar bounding box dan label pada gambar di sini
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        cv2.putText(img, label, (int(b[0]), int(b[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Tampilkan gambar dengan hasil deteksi
cv2.imshow("YOLO Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
