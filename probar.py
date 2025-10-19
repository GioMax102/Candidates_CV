import cv2
import numpy as np
from ultralytics import YOLO

def main():
    v = int(input("Version: "))
    ruta = f"/home/mariogiovanni//Documents/yolov11_candidates/best_v{v}.pt"
    model = YOLO(ruta)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return
    
    COLOR_CLASES = {
        0: (255, 0, 0),    # Rojo 
        1: (0, 255, 0),    # Verde 
        2: (0, 0, 255),    # Azul 
        3: (255, 255, 0),  # Cian 
        4: (255, 0, 255),  # Magenta 
        5: (0, 255, 255),  # Amarillo 
        6: (128, 0, 128),  # Púrpura
        7: (255, 165, 0),  # Naranja
        8: (128, 128, 0),  # Oliva 
        9: (0, 128, 128)   # Verde azulado 
    }
    
    confidence_threshold = 0.5

    print("YOLOv11 Segmentación - Presiona 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        results = model(frame)
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data
                boxes = result.boxes.data
                
                for i, mask in enumerate(masks):
                    # Verificar si hay caja correspondiente y umbral de confianza
                    if len(boxes) > i:
                        box = boxes[i]
                        x1, y1, x2, y2, conf, class_id = box[:6]
                        
                        if conf < confidence_threshold:
                            continue
                            
                        class_id_int = int(class_id)
                        color = COLOR_CLASES.get(class_id_int, (0, 0, 0))
                        
                        mask = mask.cpu().numpy()
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        
                        # mascara
                        _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY) # threshold!
                        binary_mask = binary_mask.astype(np.uint8)
                        
                        # encontrar contornos
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # dibujar figura segmentación
                        cv2.drawContours(frame, contours, -1, color, 2)
                        
                        # dibujar bounding box (bbox)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # mostrar etiqueta con clase y confianza
                        class_name = model.names[class_id_int]
                        label = f"{class_name} {conf:.2f}"
                        
                        # fondo para el texto
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1) - label_size[1] - 10),
                                    (int(x1) + label_size[0], int(y1)),
                                    color, -1)
                        
                        # Texto
                        cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Mostrar información en pantalla
        cv2.putText(frame, f"Confidence threshold: {confidence_threshold}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('YOLOv11 Segmentacion', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            # Aumentar umbral de confianza
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"Umbral de confianza aumentado a: {confidence_threshold}")
        elif key == ord('-'):
            # Disminuir umbral de confianza
            confidence_threshold = max(0.05, confidence_threshold - 0.05)
            print(f"Umbral de confianza disminuido a: {confidence_threshold}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  