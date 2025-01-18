from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

# Initialize FastAPI
app = FastAPI()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Helper function to classify skin tone
def classify_skin_tone(hue):
    if hue < 15:
        return "Fair"
    elif 15 <= hue < 25:
        return "Medium"
    else:
        return "Dark"

@app.get("/")
def root():
    return {"message": "Welcome to the Enhanced Facial Analysis API!"}

@app.post("/analyze-face/")
async def analyze_face(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")

        # Convert BGR to RGB for processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape  # Original image dimensions

        # ---------------------------
        # MediaPipe Face Detection
        # ---------------------------
        skin_tones = []
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            detection_results = face_detection.process(rgb_image)

            if detection_results.detections:
                for detection in detection_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    # Calculate bounding box dimensions
                    x, y, w_bbox, h_bbox = (
                        int(bboxC.xmin * width),
                        int(bboxC.ymin * height),
                        int(bboxC.width * width),
                        int(bboxC.height * height),
                    )
                    # Extract the face region for skin tone analysis
                    face_region = rgb_image[y : y + h_bbox, x : x + w_bbox]
                    if face_region.size > 0:
                        hsv_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
                        avg_hue = np.mean(hsv_face[:, :, 0])
                        skin_tones.append(classify_skin_tone(avg_hue))
                    else:
                        skin_tones.append("Unknown")

        if not skin_tones:
            return {"message": "No faces detected in the image."}

        # ---------------------------
        # DeepFace Analysis
        # ---------------------------
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, rgb_image)

        # Perform analysis with DeepFace
        deepface_result = DeepFace.analyze(
            img_path=temp_image_path,
            actions=["age", "gender", "emotion"],
            enforce_detection=False,
        )

        # Delete the temporary file
        import os
        os.remove(temp_image_path)

        # Combine MediaPipe and DeepFace results
        combined_results = {
            "faces": [
                {
                    "skin_tone": skin_tones[idx] if idx < len(skin_tones) else "Unknown",
                    "age": deepface_result["age"],
                    "gender": deepface_result["gender"],
                    "dominant_emotion": deepface_result["dominant_emotion"],
                }
                for idx in range(len(skin_tones))
            ]
        }

        return JSONResponse(content=combined_results)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

