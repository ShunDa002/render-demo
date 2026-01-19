# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import tempfile
import shutil
import uuid
import os
import uvicorn

from classifier import classify_video
from pose_service import analyze_video

app = FastAPI(title="Badminton Gesture Analysis")

os.makedirs("uploads", exist_ok=True)
MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB


# -----------------------
# Shot Classification Endpoint
# -----------------------
@app.post("/classify_shot")
async def classify_shot(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 30MB)")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    shot_type = classify_video(tmp_path)
    os.unlink(tmp_path)

    return JSONResponse({"shot_type": shot_type})


# -----------------------
# Pose Comparison Endpoint
# -----------------------
@app.post("/compare_pose")
async def compare_pose(file: UploadFile = File(...), shot_type: str = Form(...)):
    filename = f"uploads/{uuid.uuid4()}.mp4"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_video(filename, shot_type)
    os.remove(filename)

    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
