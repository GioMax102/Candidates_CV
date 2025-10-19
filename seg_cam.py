import cv2
import numpy as np
from ultralytics import YOLO

def main_yolov8_real_segmentation():
    model = YOLO('/home/mariogiovanni//Documents/yolov11_candidates/best_v2.pt')  
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    # Colores por clase
    colores_clases = {
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = model(frame)
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data
                boxes = result.boxes
                
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    conf = box.conf.item()
                    class_id = int(box.cls.item())

                    if conf < confidence_threshold:
                        continue

                    color = colores_clases.get(class_id, (0, 0, 0))
                    
                    mask_np = mask.cpu().numpy()
                    mask_np = cv2.resize(mask_np, 
                                       (frame.shape[1], frame.shape[0]))
                    

                    mask_uint8 = (mask_np * 255).astype(np.uint8)
                    

                    contours, _ = cv2.findContours(mask_uint8, 
                                                 cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_SIMPLE)
                    

                    for contour in contours:

                        epsilon = 0.01 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        cv2.drawContours(frame, [approx], -1, color, 2)
                    
             
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                    class_name = model.names[class_id]
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('YOLO Segmentacion Real', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_yolov8_real_segmentation()