# check all cameras are working
import cv2

# show the camera feed for each camera ID
for cam_id in range(10,20):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Camera ID {cam_id} is not available.")
        continue
    
    print(f"Camera ID {cam_id} is working. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read from Camera ID {cam_id}.")
            break
        
        cv2.imshow(f"Camera {cam_id}", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
