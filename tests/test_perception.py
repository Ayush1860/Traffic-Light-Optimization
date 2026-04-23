import sys
import os
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from models.perception import PerceptionModule
    SKIP_TEST = False
except ImportError:
    print("Skipping Perception Test: Dependencies not installed.")
    SKIP_TEST = True

def test_perception():
    if SKIP_TEST: return

    video_path = "d:/Traffic-AI-Project/data/sample_traffic.mp4"
    if not os.path.exists(video_path):
        print("Video file not found.")
        return

    perception = PerceptionModule()
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    if ret:
        count, detections = perception.detect_vehicles(frame)
        print(f"Frame 1: Detected {count} vehicles.")
        print(f"Detections: {detections}")
    else:
        print("Failed to read video frame.")
    
    cap.release()

if __name__ == "__main__":
    test_perception()
