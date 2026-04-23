from ultralytics import YOLO
import cv2

class PerceptionModule:
    def __init__(self, model_path='yolo11n.pt'):
        """
        Initialize YOLOv11 model.
        """
        # Load a pretrained YOLOv11 model
        try:
            self.model = YOLO(model_path) 
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def detect_vehicles(self, frame):
        """
        Detect vehicles in a frame and return the count and detections.
        """
        if self.model is None:
            return 0, []

        results = self.model(frame, verbose=False)
        
        vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck (COCO class IDs)
        vehicle_count = 0
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in vehicle_classes:
                    vehicle_count += 1
                    detections.append(box.xyxy[0].cpu().numpy()) # Bounding box coordinates

        return vehicle_count, detections

if __name__ == "__main__":
    # Test stub
    perception = PerceptionModule()
    print("Perception module initialized.")
