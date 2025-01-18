from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

mp_face_detection = mp.solutions.face_detection

def classify_skin_tone(hue):
    if hue < 15:
        return "Fair"
    elif 15 <= hue < 25:
        return "Medium"
    else:
        return "Dark"

def analyze_deepface(image: np.ndarray):
    return DeepFace.analyze(
        img_path=image,
        actions=["age", "gender", "emotion"],
        enforce_detection=False,
    )

@app.get("/")
def root():
    return {"message": "Welcome to the Enhanced Facial Analysis API!"}

@app.post("/analyze-face/")
async def analyze_face(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        skin_tones = []
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            detection_results = face_detection.process(rgb_image)

            if detection_results.detections:
                for detection in detection_results.detections:
                    bboxC = detection.location_data.relative_bounding_box

                    x, y, w_bbox, h_bbox = (
                        max(0, int(bboxC.xmin * width)),
                        max(0, int(bboxC.ymin * height)),
                        max(1, int(bboxC.width * width)),
                        max(1, int(bboxC.height * height)),
                    )
                    face_region = rgb_image[y : y + h_bbox, x : x + w_bbox]
                    if face_region.size > 0:
                        hsv_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
                        avg_hue = np.mean(hsv_face[:, :, 0])
                        skin_tones.append(classify_skin_tone(avg_hue))
                    else:
                        skin_tones.append("Unknown")

        if not skin_tones:
            return {"message": "No faces detected in the image."}

        with ThreadPoolExecutor() as executor:
            future = executor.submit(analyze_deepface, rgb_image)
            deepface_result = future.result()

        combined_results = {
            "faces": [
                {
                    "skin_tone": skin_tones[idx] if idx < len(skin_tones) else "Unknown",
                    "age": deepface_result.get("age", "Unknown"),
                    "gender": deepface_result.get("gender", "Unknown"),
                    "dominant_emotion": deepface_result.get("dominant_emotion", "Unknown"),
                }
                for idx in range(len(skin_tones))
            ]
        }

        return JSONResponse(content=combined_results)

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


