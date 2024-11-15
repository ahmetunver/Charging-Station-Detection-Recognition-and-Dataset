import cv2
import supervision as sv
from ultralytics import YOLO

#google collabstan oluşturduğumuz best.pt [en iyi ağırlık dosyasını] ekledik.
model = YOLO('C:/Users/asus/OneDrive/Desktop/Pyhton/pycharm_derlemeleri/real time object detection and recognition/best.pt')

# Kamerayı başlattık. [0 varsayılan pc kamerası]
cap = cv2.VideoCapture(0)

# Kamera açılmadıysa uyarı mesajı yazdırdık.
if not cap.isOpened():
    print("Kameradan görüntü gelmiyor...")

# Görüntü üzerinde sınırlayıcı kutular ve etiketler eklemek için annotator [sınırlayıcı çerçeve] oluştur
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# while döngüsünde Sonsuz döngü ile her frame i ele aldık.
while True:
    # Kameradan görüntü okuma kısmı
    ret, frame = cap.read()

    # Eğer görüntü okunamazsa döngüyü durdur kısmı
    if not ret:
        break

    frame = cv2.flip(frame,1)   #ayna görünümü kaldırdık

    # Modeli çalıştırarak görüntüdeki nesneleri tespit et
    results = model(frame)[0]
    
    # Tespit edilen nesneleri Supervision kütüphanesinin Detections modulünü kullanarak detections değişkenine atatık.
    detections = sv.Detections.from_ultralytics(results)

    # görüntüde yakaladığımız objeyi çerçeve içine aldık
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    
    # Görüntü üzerine görüntü labelını [görüntü adını] yazdırdık.
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
   
    # son görüntüyü ekrana ver
    cv2.imshow("Webcam", annotated_image)

    # Esc tuşuna basıldığında döngüyü kapat
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Esc tuşuna basıldı.. Kapatılıyor..")
        break

cap.release()
cv2.destroyAllWindows()