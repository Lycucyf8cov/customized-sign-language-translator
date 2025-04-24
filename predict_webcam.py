import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# Load model
model = load_model("sign_lang_model.h5")

# Get class labels
class_names = sorted([d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))])
le = LabelEncoder()
le.fit(class_names)

IMG_SIZE = 224

# Webcam start
cap = cv2.VideoCapture(0)
print("Press SPACE to predict | Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw box
    cv2.rectangle(frame, (100, 100), (324, 324), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand inside box", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Sign Language Prediction", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    elif key == 32:  # SPACE to capture
        roi = frame[100:324, 100:324]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=0) / 255.0

        pred = model.predict(img)
        pred_label = le.inverse_transform([np.argmax(pred)])
        confidence = np.max(pred)

        print(f"Prediction: {pred_label[0]} ({confidence*100:.2f}%)")
        cv2.putText(frame, f"Predicted: {pred_label[0]}", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Prediction", frame)
        cv2.waitKey(2000)

cap.release()
cv2.destroyAllWindows()
