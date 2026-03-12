import base64
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from detectar_video_json import NOMBRES_CLASES, procesar_frame


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = PROJECT_ROOT / "videos"
PROCESSED_DIR = PROJECT_ROOT / "process_videos"
MODELOS_DIR = PROJECT_ROOT / "Modelos"


def _ensure_directories() -> None:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _get_device():
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def _load_models(device):
    from ultralytics import YOLO

    modelo_coarse_path = MODELOS_DIR / "best_s.pt"
    modelo_fine_path = MODELOS_DIR / "best_l.pt"

    if not modelo_coarse_path.is_file():
        raise RuntimeError(f"No se encontró el modelo coarse: {modelo_coarse_path}")
    if not modelo_fine_path.is_file():
        raise RuntimeError(f"No se encontró el modelo fine: {modelo_fine_path}")

    modelo_coarse = YOLO(str(modelo_coarse_path))
    modelo_coarse.to(device)
    modelo_fine = YOLO(str(modelo_fine_path))
    modelo_fine.to(device)

    return modelo_coarse, modelo_fine


def _default_config(device):
    return {
        "imgsz_coarse": 416,
        "conf_coarse": 0.10,
        "imgsz_fine": 640,
        "conf_fine": 0.30,
        "grid_rows": 2,
        "grid_cols": 3,
        "overlap": 100,
        "iou_nms": 0.45,
        "debug_tiles": False,
        "device": device,
    }


async def lifespan(app: FastAPI):
    _ensure_directories()
    device = _get_device()
    modelo_coarse, modelo_fine = _load_models(device)
    app.state.device = device
    app.state.modelo_coarse = modelo_coarse
    app.state.modelo_fine = modelo_fine
    app.state.config = _default_config(device)
    try:
        yield
    finally:
        # Los objetos YOLO no requieren liberación explícita,
        # pero si en el futuro se añaden recursos, se pueden limpiar aquí.
        pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="El archivo de video debe tener un nombre.")

    ext = os.path.splitext(file.filename)[1] or ".mp4"
    uid = uuid4().hex

    input_path = VIDEOS_DIR / f"{uid}_{file.filename}"
    output_path = PROCESSED_DIR / f"{uid}_procesado{ext}"

    # Guardar el archivo subido
    try:
        with input_path.open("wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo guardar el video de entrada: {e}")

    # Procesar el video usando el pipeline ya existente
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="No se pudo abrir el video de entrada.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    datos_json = {
        "video": file.filename,
        "total_frames": total_frames,
        "total_detecciones": 0,
        "frames": {},
    }
    total_dets = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_out, scores, clases = procesar_frame(
                frame,
                app.state.modelo_coarse,
                app.state.modelo_fine,
                app.state.config,
            )
            writer.write(frame_out)

            conteo_frame: dict[str, int] = defaultdict(int)
            for cls in clases:
                nombre = NOMBRES_CLASES[cls] if cls < len(NOMBRES_CLASES) else str(cls)
                conteo_frame[nombre] += 1

            n_dets = len(clases)
            total_dets += n_dets
            datos_json["frames"][str(frame_idx)] = {"total": n_dets, **dict(conteo_frame)}
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    if not output_path.is_file():
        raise HTTPException(status_code=500, detail="No se generó el video de salida.")

    # Re-encodar con H.264 CRF 18 para buena calidad y compatibilidad web
    output_h264 = output_path.with_stem(output_path.stem + "_h264")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(output_path),
                "-vcodec", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-movflags", "+faststart",
                str(output_h264),
            ],
            check=True,
            capture_output=True,
        )
        final_path = output_h264
    except (subprocess.CalledProcessError, FileNotFoundError):
        final_path = output_path

    datos_json["total_frames"] = frame_idx
    datos_json["total_detecciones"] = total_dets

    with final_path.open("rb") as vf:
        video_b64 = base64.b64encode(vf.read()).decode("utf-8")

    datos_json["video_base64"] = video_b64
    return JSONResponse(content=datos_json)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Video processing API is running"}

