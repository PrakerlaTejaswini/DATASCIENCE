from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # nano model (fast)

# Read image
image_path = r"C:\Users\LENOVO\Documents\COMPUTER VISION\YOLO\image 1.jpg"
img = cv2.imread(image_path)

# Run detection
results = model(img)

# Draw bounding boxes
annotated_frame = results[0].plot()

# Show output
cv2.imshow("YOLOv8 Image Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite("output_image.jpg", annotated_frame)
