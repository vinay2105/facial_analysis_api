from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import mediapipe as mp
import requests

app = FastAPI()

mp_face_detection = mp.solutions.face_detection

class ImageURL(BaseModel):
    url: str

def classify_skin_tone(hue):
    if hue < 15:
        return "Fair"
    elif 15 <= hue < 25:
        return "Medium"
    else:
        return "Dark"

@app.get("/")
def root():
    return {"message": "Welcome to the Optimized Skin Tone Analysis API!"}

@app.post("/analyze-skin-tone/")
async def analyze_skin_tone(data: ImageURL):
    try:
        response = requests.get(data.url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch the image from the URL.")
        
        np_image = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")
        
        resized_image = cv2.resize(image, (640, 480))
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        height, width, _ = resized_image.shape

        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            detection_results = face_detection.process(rgb_image)
            if detection_results.detections:
                detection = detection_results.detections[0]
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
                    skin_tone = classify_skin_tone(avg_hue)
                else:
                    skin_tone = "Unknown"
                return {"skin_tone": skin_tone}
        
        return {"message": "No faces detected in the image."}
    except HTTPException as e:
        raise e
    except requests.exceptions.RequestException:
        return {"error": "Error fetching the image. Check the URL and try again."}
    except Exception as e:
        return {"error": str(e)}



