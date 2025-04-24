import cv2
import os
import time

# SETTINGS
label = 'ok'  # Change this to the current letter (A, B, C, ...)
num_images = 50
delay_between = 1  # seconds
save_path = f'dataset/{label}'

# Create directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
print(f"Starting capture for letter '{label}'. Press 'q' to quit early.")

img_count = 0

while img_count < num_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show the webcam feed
    cv2.imshow('Capture - Press q to quit', frame)

    # Save image every X seconds
    img_name = os.path.join(save_path, f"{label}_{img_count+1}.jpg")
    cv2.imwrite(img_name, frame)
    print(f"[{img_count+1}/{num_images}] Saved: {img_name}")
    img_count += 1

    # Wait before capturing next
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Capture manually stopped.")
        break
    time.sleep(delay_between)

# Release and cleanup
cap.release()
cv2.destroyAllWindows()
print("Done.")
