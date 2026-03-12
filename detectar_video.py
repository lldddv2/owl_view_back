"""
detectar_video.py
-----------------
Aplica el pipeline Coarse-to-Fine (Eagle Eye) a un video y genera
un nuevo video con las detecciones dibujadas.

Uso:
    python detectar_video.py <video_entrada> [opciones]

Ejemplos:
    python detectar_video.py "videos de prueba/uav0000086_00000_v.mp4"
    python detectar_video.py mi_video.mp4 --output resultado.mp4
    python detectar_video.py mi_video.mp4 --conf-fine 0.4 --debug-tiles
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from torchvision.ops import nms as torchvision_nms
from ultralytics import YOLO

# ── Clases VisDrone ───────────────────────────────────────────────────────────
NOMBRES_CLASES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

COLORES_CLASE = [
    (0, 255, 0),     # pedestrian
    (0, 220, 255),   # people
    (255, 100, 0),   # bicycle
    (0, 0, 255),     # car
    (255, 0, 200),   # van
    (128, 0, 255),   # truck
    (0, 180, 255),   # tricycle
    (255, 180, 0),   # awning-tricycle
    (0, 255, 200),   # bus
    (180, 255, 0),   # motor
]


# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════

def generar_tiles(frame_h, frame_w, grid_rows, grid_cols, overlap):
    tiles = []
    h_step = frame_h // grid_rows
    w_step = frame_w // grid_cols
    for r in range(grid_rows):
        for c in range(grid_cols):
            x1 = c * w_step
            y1 = r * h_step
            x2 = frame_w if c == grid_cols - 1 else (c + 1) * w_step
            y2 = frame_h if r == grid_rows - 1 else (r + 1) * h_step
            x1 = max(0, x1 - overlap)
            y1 = max(0, y1 - overlap)
            x2 = min(frame_w, x2 + overlap)
            y2 = min(frame_h, y2 + overlap)
            tiles.append((x1, y1, x2, y2))
    return tiles


def filtrar_tiles_activos(tiles, cajas_coarse):
    if len(cajas_coarse) == 0:
        return []
    activos = []
    for (tx1, ty1, tx2, ty2) in tiles:
        for (bx1, by1, bx2, by2) in cajas_coarse:
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                activos.append((tx1, ty1, tx2, ty2))
                break
    return activos


def mapear_a_global(cajas_locales, tile_x1, tile_y1):
    cajas = cajas_locales.copy().astype(float)
    cajas[:, 0] += tile_x1
    cajas[:, 1] += tile_y1
    cajas[:, 2] += tile_x1
    cajas[:, 3] += tile_y1
    return cajas


def aplicar_nms_global(all_boxes, all_scores, all_classes, iou_threshold=0.45):
    if not all_boxes:
        return np.array([]), np.array([]), []
    boxes_t   = torch.tensor(all_boxes,  dtype=torch.float32)
    scores_t  = torch.tensor(all_scores, dtype=torch.float32)
    classes_a = np.array(all_classes,    dtype=int)
    keep_indices = []
    for cls_id in np.unique(classes_a):
        mascara    = classes_a == cls_id
        idx        = np.where(mascara)[0]
        keep_local = torchvision_nms(boxes_t[mascara], scores_t[mascara], iou_threshold)
        keep_indices.extend(idx[keep_local.numpy()])
    keep_indices = sorted(keep_indices)
    return (
        boxes_t[keep_indices].numpy(),
        scores_t[keep_indices].numpy(),
        classes_a[keep_indices].tolist()
    )


def dibujar_detecciones(frame, cajas, scores, clases):
    for (x1, y1, x2, y2), score, cls in zip(cajas, scores, clases):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color  = COLORES_CLASE[int(cls) % len(COLORES_CLASE)]
        nombre = NOMBRES_CLASES[int(cls)] if int(cls) < len(NOMBRES_CLASES) else str(cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        etiqueta = f"{nombre} {score:.2f}"
        cv2.putText(frame, etiqueta, (x1, max(y1 - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame


def dibujar_tiles_debug(frame, tiles, tiles_activos):
    overlay = frame.copy()
    for (x1, y1, x2, y2) in tiles:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 80, 80), 1)
    for (x1, y1, x2, y2) in tiles_activos:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), -1)
    return cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def procesar_frame(frame, modelo_coarse, modelo_fine, config):
    frame_h, frame_w = frame.shape[:2]

    # 1. Visión global (Coarse)
    resultado_c = modelo_coarse.predict(
        source=frame, imgsz=config['imgsz_coarse'],
        conf=config['conf_coarse'], verbose=False, device=config['device']
    )[0]
    cajas_coarse = (
        resultado_c.boxes.xyxy.cpu().numpy()
        if len(resultado_c.boxes) > 0
        else np.empty((0, 4), dtype=float)
    )

    # 2. Generar cuadrícula con overlap
    tiles = generar_tiles(frame_h, frame_w,
                          config['grid_rows'], config['grid_cols'],
                          config['overlap'])

    # 3. Filtrar tiles activos
    tiles_activos = filtrar_tiles_activos(tiles, cajas_coarse)

    if not tiles_activos:
        frame_out = frame.copy()
        if config['debug_tiles']:
            frame_out = dibujar_tiles_debug(frame_out, tiles, [])
        return frame_out, 0

    # 4. Visión detallada (Fine) sobre tiles activos
    all_boxes, all_scores, all_classes = [], [], []
    for (tx1, ty1, tx2, ty2) in tiles_activos:
        recorte = frame[ty1:ty2, tx1:tx2]
        resultado_f = modelo_fine.predict(
            source=recorte, imgsz=config['imgsz_fine'],
            conf=config['conf_fine'], verbose=False, device=config['device']
        )[0]
        if len(resultado_f.boxes) == 0:
            continue
        cajas_glob = mapear_a_global(resultado_f.boxes.xyxy.cpu().numpy(), tx1, ty1)
        all_boxes.extend(cajas_glob.tolist())
        all_scores.extend(resultado_f.boxes.conf.cpu().numpy().tolist())
        all_classes.extend(resultado_f.boxes.cls.cpu().numpy().astype(int).tolist())

    # 5. NMS global
    if all_boxes:
        cajas, scores, clases = aplicar_nms_global(
            all_boxes, all_scores, all_classes, config['iou_nms']
        )
    else:
        cajas, scores, clases = np.array([]), np.array([]), []

    # 6. Dibujar
    frame_out = frame.copy()
    if config['debug_tiles']:
        frame_out = dibujar_tiles_debug(frame_out, tiles, tiles_activos)
    if len(cajas) > 0:
        frame_out = dibujar_detecciones(frame_out, cajas, scores, clases)

    return frame_out, len(cajas)


def procesar_video_core(
    input_path,
    output_path,
    modelo_coarse_path,
    modelo_fine_path,
    imgsz_coarse=416,
    conf_coarse=0.10,
    imgsz_fine=640,
    conf_fine=0.30,
    grid_rows=2,
    grid_cols=3,
    overlap=100,
    iou_nms=0.45,
    debug_tiles=False,
):
    """
    Procesa un video usando el pipeline Coarse-to-Fine y devuelve la ruta de salida.
    Lanza excepciones en caso de error en lugar de terminar el proceso.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"El archivo de entrada no existe: {input_path}")

    device = 0 if torch.cuda.is_available() else 'cpu'

    config = {
        'imgsz_coarse': imgsz_coarse,
        'conf_coarse': conf_coarse,
        'imgsz_fine': imgsz_fine,
        'conf_fine': conf_fine,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'overlap': overlap,
        'iou_nms': iou_nms,
        'debug_tiles': debug_tiles,
        'device': device,
    }

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_detecciones.mp4"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

    print(f"\n🎬 Video entrada : {input_path}  ({w}×{h}, {fps:.1f} fps, {total_frames} frames)")
    print(f"💾 Video salida  : {output_path}")
    print(f"🖥️  Device        : {'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    print(f"📐 Grid          : {grid_rows}×{grid_cols} | overlap={overlap}px")
    print(f"🔍 Coarse        : {modelo_coarse_path}  (imgsz={imgsz_coarse}, conf={conf_coarse})")
    print(f"🔬 Fine          : {modelo_fine_path}  (imgsz={imgsz_fine}, conf={conf_fine})\n")

    print("⏳ Cargando modelos...")
    modelo_coarse = YOLO(modelo_coarse_path)
    modelo_coarse.to(device)
    modelo_fine = YOLO(modelo_fine_path)
    modelo_fine.to(device)
    print("✅ Modelos listos\n")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    t_inicio = time.time()
    total_dets = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_out, n_dets = procesar_frame(frame, modelo_coarse, modelo_fine, config)
            writer.write(frame_out)
            total_dets += n_dets
            frame_idx += 1

            if frame_idx % 50 == 0 or frame_idx == total_frames:
                elapsed = time.time() - t_inicio
                fps_real = frame_idx / elapsed if elapsed > 0 else 0
                restante = (total_frames - frame_idx) / fps_real if fps_real > 0 else 0
                pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                print(
                    f"  {frame_idx:5d}/{total_frames} ({pct:5.1f}%) | "
                    f"{fps_real:.1f} fps | ETA: {restante:.0f}s | "
                    f"dets en este frame: {n_dets}",
                    end='\r',
                )
    finally:
        cap.release()
        writer.release()

    elapsed_total = time.time() - t_inicio
    fps_promedio = frame_idx / elapsed_total if elapsed_total > 0 else 0
    print(f"\n\n✅ Listo!")
    print(f"   {frame_idx} frames procesados en {elapsed_total:.1f}s ({fps_promedio:.1f} fps promedio)")
    print(f"   Detecciones totales : {total_dets}")
    print(f"   Video guardado en   : {output_path}")

    return output_path


def procesar_video(args):
    """
    Envoltura para mantener la interfaz de linea de comandos original.
    """
    try:
        procesar_video_core(
            input_path=args.input,
            output_path=args.output,
            modelo_coarse_path=args.modelo_coarse,
            modelo_fine_path=args.modelo_fine,
            imgsz_coarse=args.imgsz_coarse,
            conf_coarse=args.conf_coarse,
            imgsz_fine=args.imgsz_fine,
            conf_fine=args.conf_fine,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            overlap=args.overlap,
            iou_nms=args.iou_nms,
            debug_tiles=args.debug_tiles,
        )
    except Exception as e:
        print(f"❌ Error procesando el video: {e}")
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description='Detecta objetos en un video usando el pipeline Coarse-to-Fine Eagle Eye.'
    )

    # Argumentos obligatorios
    parser.add_argument('input',  help='Video de entrada (.mp4, .avi, ...)')

    # Salida
    parser.add_argument('--output', '-o', default=None,
                        help='Video de salida (por defecto: <entrada>_detecciones.mp4)')

    # Modelos
    parser.add_argument('--modelo-coarse', default=os.path.join(PROJECT_ROOT, 'Modelos', 'best_s.pt'),
                        help='Ruta al modelo Coarse/Small (por defecto: Modelos/best_s.pt)')
    parser.add_argument('--modelo-fine',   default=os.path.join(PROJECT_ROOT, 'Modelos', 'best_l.pt'),
                        help='Ruta al modelo Fine/Large (por defecto: Modelos/best_l.pt)')

    # Parámetros de inferencia
    parser.add_argument('--imgsz-coarse', type=int,   default=416,  help='imgsz para el modelo Coarse (default: 416)')
    parser.add_argument('--conf-coarse',  type=float, default=0.10, help='Confianza mínima Coarse (default: 0.10)')
    parser.add_argument('--imgsz-fine',   type=int,   default=640,  help='imgsz para el modelo Fine (default: 640)')
    parser.add_argument('--conf-fine',    type=float, default=0.30, help='Confianza mínima Fine (default: 0.30)')

    # Cuadrícula
    parser.add_argument('--grid-rows', type=int,   default=2,    help='Filas de la cuadrícula (default: 2)')
    parser.add_argument('--grid-cols', type=int,   default=3,    help='Columnas de la cuadrícula (default: 3)')
    parser.add_argument('--overlap',   type=int,   default=100,  help='Solapamiento en píxeles (default: 100)')
    parser.add_argument('--iou-nms',   type=float, default=0.45, help='Umbral IoU para NMS global (default: 0.45)')

    # Debug
    parser.add_argument('--debug-tiles', action='store_true',
                        help='Dibuja la cuadrícula y tiles activos en el video de salida')

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ El archivo de entrada no existe: {args.input}")
        sys.exit(1)

    procesar_video(args)


if __name__ == '__main__':
    main()
