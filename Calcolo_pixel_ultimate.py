"""
pixel_size_calculator.py
========================
Calcola la dimensione di un pixel in mm e µm a una data distanza oggetto-fotocamera,
usando i parametri di calibrazione estratti da calibparam3.mat.
 
Uso:
    python pixel_size_calculator.py <cartella_immagini>
    python pixel_size_calculator.py <cartella_immagini> --distance 284.6
    python pixel_size_calculator.py <cartella_immagini> --distance 150 --sensor-width 5.6
 
Parametri di calibrazione (estratti da calibparam3.mat con MATLAB R2025b):
    fx = 819.78 px   (lunghezza focale asse X in pixel)
    fy = 818.84 px   (lunghezza focale asse Y in pixel)
    cx = 814.18 px   (punto principale X)
    cy = 594.03 px   (punto principale Y)
    k1 =   0.126     (distorsione radiale 1)
    k2 =   4.340     (distorsione radiale 2)
    Immagine: 1936 x 1216 px
    Unità calibrazione: millimetri
"""
 
import os
import sys
import argparse
import struct
import zlib
import math
 
# ── Dipendenze opzionali ────────────────────────────────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
 
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
 
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
 
# ── Parametri di calibrazione estratti da calibparam3.mat ──────────────────
CALIB = {
    "fx":          819.7782,   # lunghezza focale X [pixel]
    "fy":          818.8350,   # lunghezza focale Y [pixel]
    "cx":          814.1819,   # punto principale X [pixel]
    "cy":          594.0343,   # punto principale Y [pixel]
    "k1":            0.1260,   # distorsione radiale 1
    "k2":            4.3399,   # distorsione radiale 2
    "image_width":  1936,      # [pixel]
    "image_height": 1216,      # [pixel]
    "world_units":  "mm",      # unità di misura della calibrazione
    # Distanza media rilevata durante la calibrazione (vettori di traslazione Z)
    "calib_distance_mm": 284.6,
}
 
# ── Estensioni immagine supportate ─────────────────────────────────────────
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".webp", ".ppm", ".pgm", ".pbm",
}
 
 
# ════════════════════════════════════════════════════════════════════════════
# Calcolo dimensione pixel
# ════════════════════════════════════════════════════════════════════════════
 
def pixel_size_mm(distance_mm: float,
                  fx: float = CALIB["fx"],
                  fy: float = CALIB["fy"],
                  sensor_width_mm: float = None,
                  image_width_px: int = CALIB["image_width"]) -> dict:
    """
    Calcola la dimensione di un pixel nel piano oggetto a distanza Z.
 
    Metodo 1 – dalla sola lunghezza focale in pixel (richiede la distanza):
        GSD_x = Z / fx   [mm/px]
        GSD_y = Z / fy   [mm/px]
    
    Questo è il metodo diretto: la lunghezza focale in pixel = f / p_size,
    quindi p_size = f_mm / fx.  Ma poiché non conosciamo f in mm separatamente,
    usiamo la relazione GSD = Z / f_px che la combina automaticamente.
 
    Metodo 2 – se conosci la larghezza fisica del sensore (mm):
        pixel_pitch = sensor_width_mm / image_width_px
        GSD = Z * pixel_pitch / focal_length_mm
        dove focal_length_mm ≈ fx * pixel_pitch
 
    Returns:
        dict con gsd_x_mm, gsd_y_mm, gsd_x_um, gsd_y_um, ecc.
    """
    # GSD = Ground Sample Distance (mm per pixel nel piano oggetto)
    gsd_x_mm = distance_mm / fx
    gsd_y_mm = distance_mm / fy
 
    result = {
        "distance_mm":   distance_mm,
        "gsd_x_mm":      gsd_x_mm,
        "gsd_y_mm":      gsd_y_mm,
        "gsd_x_um":      gsd_x_mm * 1000,
        "gsd_y_um":      gsd_y_mm * 1000,
        "gsd_mean_mm":   (gsd_x_mm + gsd_y_mm) / 2,
        "gsd_mean_um":   (gsd_x_mm + gsd_y_mm) / 2 * 1000,
        "method":        "focal_length_px",
    }
 
    # Se è fornita la larghezza del sensore, calcola anche il pixel pitch fisico
    if sensor_width_mm is not None:
        pixel_pitch_mm = sensor_width_mm / image_width_px
        focal_length_mm = fx * pixel_pitch_mm
        result["pixel_pitch_mm"]    = pixel_pitch_mm
        result["pixel_pitch_um"]    = pixel_pitch_mm * 1000
        result["focal_length_mm"]   = focal_length_mm
        # GSD di verifica tramite formula alternativa (deve coincidere)
        gsd_check_mm = distance_mm * pixel_pitch_mm / focal_length_mm
        result["gsd_check_mm"]      = gsd_check_mm
        result["method"]            = "focal_length_px + sensor_width"
 
    return result
 
 
def field_of_view(distance_mm: float,
                  fx: float = CALIB["fx"],
                  fy: float = CALIB["fy"],
                  image_width_px: int  = CALIB["image_width"],
                  image_height_px: int = CALIB["image_height"]) -> dict:
    """Campo visivo (FoV) nel piano oggetto a distanza Z."""
    fov_w_mm = image_width_px  * (distance_mm / fx)
    fov_h_mm = image_height_px * (distance_mm / fy)
    # Angolo (in gradi) del campo visivo
    angle_h_deg = 2 * math.degrees(math.atan(image_width_px  / (2 * fx)))
    angle_v_deg = 2 * math.degrees(math.atan(image_height_px / (2 * fy)))
    return {
        "fov_width_mm":   fov_w_mm,
        "fov_height_mm":  fov_h_mm,
        "fov_width_cm":   fov_w_mm  / 10,
        "fov_height_cm":  fov_h_mm  / 10,
        "fov_angle_h_deg": angle_h_deg,
        "fov_angle_v_deg": angle_v_deg,
    }
 
 
# ════════════════════════════════════════════════════════════════════════════
# Lettura immagini
# ════════════════════════════════════════════════════════════════════════════
 
def load_image_info(path: str) -> dict:
    """Legge dimensioni e info base di un'immagine."""
    info = {"path": path, "filename": os.path.basename(path)}
 
    if HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            info.update({"width": w, "height": h, "channels": channels,
                         "dtype": str(img.dtype), "loaded_with": "OpenCV"})
            return info
 
    if HAS_PIL:
        with Image.open(path) as im:
            w, h = im.size
            info.update({"width": w, "height": h, "channels": len(im.getbands()),
                         "mode": im.mode, "loaded_with": "Pillow"})
            return info
 
    # Fallback: solo dimensioni del file
    info["file_size_bytes"] = os.path.getsize(path)
    info["loaded_with"] = "stat only"
    return info
 
 
def find_images(folder: str) -> list:
    """Trova tutte le immagini in una cartella (non ricorsivo)."""
    images = []
    if not os.path.isdir(folder):
        print(f"  [!] Cartella non trovata: {folder}")
        return images
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(os.path.join(folder, fname))
    return images
 
 
# ════════════════════════════════════════════════════════════════════════════
# Stampa risultati
# ════════════════════════════════════════════════════════════════════════════
 
def print_separator(char="─", width=62):
    print(char * width)
 
def print_calibration_summary():
    print_separator("═")
    print("  PARAMETRI DI CALIBRAZIONE  (calibparam3.mat)")
    print_separator("═")
    print(f"  Lunghezza focale:  fx = {CALIB['fx']:.4f} px   fy = {CALIB['fy']:.4f} px")
    print(f"  Punto principale:  cx = {CALIB['cx']:.4f} px   cy = {CALIB['cy']:.4f} px")
    print(f"  Distorsione rad.:  k1 = {CALIB['k1']:.4f}       k2 = {CALIB['k2']:.4f}")
    print(f"  Immagine:          {CALIB['image_width']} × {CALIB['image_height']} px")
    print(f"  Unità calibraz.:   {CALIB['world_units']}")
    print(f"  Distanza calib.:   ≈ {CALIB['calib_distance_mm']:.1f} mm (media vettori Z)")
    print_separator()
 
def print_pixel_size_results(res: dict, fov: dict):
    print_separator("─")
    print(f"  DISTANZA OGGETTO–FOTOCAMERA: {res['distance_mm']:.2f} mm")
    print_separator("─")
    print(f"  GSD asse X:  {res['gsd_x_mm']:.6f} mm/px  =  {res['gsd_x_um']:.3f} µm/px")
    print(f"  GSD asse Y:  {res['gsd_y_mm']:.6f} mm/px  =  {res['gsd_y_um']:.3f} µm/px")
    print(f"  GSD medio:   {res['gsd_mean_mm']:.6f} mm/px  =  {res['gsd_mean_um']:.3f} µm/px")
    if "pixel_pitch_mm" in res:
        print()
        print(f"  Pixel pitch fisico: {res['pixel_pitch_mm']:.6f} mm = {res['pixel_pitch_um']:.3f} µm")
        print(f"  Focale reale (mm):  {res['focal_length_mm']:.3f} mm")
    print()
    print(f"  Campo visivo a {res['distance_mm']:.1f} mm:")
    print(f"    Larghezza: {fov['fov_width_mm']:.2f} mm  ({fov['fov_width_cm']:.2f} cm)   "
          f"angolo: {fov['fov_angle_h_deg']:.1f}°")
    print(f"    Altezza:   {fov['fov_height_mm']:.2f} mm  ({fov['fov_height_cm']:.2f} cm)   "
          f"angolo: {fov['fov_angle_v_deg']:.1f}°")
    print_separator()
 
def print_image_results(img_info: dict, res: dict):
    """Stampa risultati per una singola immagine, con verifica delle dimensioni."""
    w = img_info.get("width")
    h = img_info.get("height")
    print(f"\n  📷  {img_info['filename']}")
    if w and h:
        print(f"      Dimensioni: {w} × {h} px")
        # Avviso se le dimensioni non corrispondono alla calibrazione
        if w != CALIB["image_width"] or h != CALIB["image_height"]:
            print(f"      ⚠️  ATTENZIONE: questa immagine è {w}×{h} px,")
            print(f"         ma la calibrazione è stata fatta su "
                  f"{CALIB['image_width']}×{CALIB['image_height']} px.")
            print(f"         I valori GSD sono validi solo se il setup è identico.")
            # Scalatura approssimata se l'immagine è ridimensionata
            scale_x = w / CALIB["image_width"]
            scale_y = h / CALIB["image_height"]
            if abs(scale_x - scale_y) < 0.01:
                fx_scaled = CALIB["fx"] * scale_x
                fy_scaled = CALIB["fy"] * scale_y
                gsd_x_scaled = res["distance_mm"] / fx_scaled
                gsd_y_scaled = res["distance_mm"] / fy_scaled
                print(f"         → Se l'immagine è solo ridimensionata (scala {scale_x:.3f}×),")
                print(f"           GSD corretto: X = {gsd_x_scaled*1000:.3f} µm/px, "
                      f"Y = {gsd_y_scaled*1000:.3f} µm/px")
    print(f"      GSD medio: {res['gsd_mean_mm']:.6f} mm/px  =  {res['gsd_mean_um']:.3f} µm/px")
 
 
# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcola la dimensione di un pixel (GSD) da parametri di calibrazione.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python pixel_size_calculator.py ./immagini
  python pixel_size_calculator.py ./immagini --distance 200
  python pixel_size_calculator.py ./immagini --distance 150 --sensor-width 5.6
  python pixel_size_calculator.py ./immagini --multi-distance 100 200 300 500
 
Nota:
  GSD (Ground Sample Distance) = mm reali coperti da 1 pixel a distanza Z.
  Formula: GSD = Z / f_px
  dove Z è la distanza in mm e f_px è la lunghezza focale in pixel.
        """,
    )
    parser.add_argument("folder",
                        help="Cartella contenente le immagini da analizzare")
    parser.add_argument("--distance", type=float,
                        default=CALIB["calib_distance_mm"],
                        help=f"Distanza oggetto–fotocamera in mm "
                             f"(default: {CALIB['calib_distance_mm']} mm, "
                             f"dalla calibrazione)")
    parser.add_argument("--sensor-width", type=float, default=None,
                        metavar="MM",
                        help="Larghezza fisica del sensore in mm (opzionale). "
                             "Se fornita, calcola anche pixel pitch e focale in mm.")
    parser.add_argument("--multi-distance", type=float, nargs="+",
                        metavar="MM",
                        help="Calcola il GSD per più distanze contemporaneamente.")
    return parser.parse_args()
 
 
def main():
    args = parse_args()
 
    print()
    print_calibration_summary()
 
    # ── Lista distanze da analizzare ───────────────────────────────────────
    distances = [args.distance]
    if args.multi_distance:
        distances = sorted(set(args.multi_distance))
 
    # ── Calcolo GSD per ogni distanza ──────────────────────────────────────
    for dist in distances:
        res = pixel_size_mm(
            distance_mm=dist,
            sensor_width_mm=args.sensor_width,
        )
        fov = field_of_view(distance_mm=dist)
        print_pixel_size_results(res, fov)
 
    # ── Analisi immagini nella cartella ───────────────────────────────────
    images = find_images(args.folder)
    if not images:
        print(f"\n  Nessuna immagine trovata in: {args.folder}")
        print(f"  Estensioni supportate: {', '.join(sorted(IMAGE_EXTENSIONS))}")
    else:
        print(f"\n  IMMAGINI TROVATE IN: {args.folder}  ({len(images)} file)")
        # Usa la prima distanza fornita per il riepilogo per-immagine
        main_res = pixel_size_mm(
            distance_mm=args.distance,
            sensor_width_mm=args.sensor_width,
        )
        for img_path in images:
            img_info = load_image_info(img_path)
            print_image_results(img_info, main_res)
        print()
 
    # ── Riepilogo tabellare multi-distanza ────────────────────────────────
    if len(distances) > 1:
        print_separator("═")
        print("  TABELLA RIEPILOGATIVA GSD")
        print_separator("═")
        print(f"  {'Distanza (mm)':>14}  {'GSD X (mm/px)':>14}  "
              f"{'GSD X (µm/px)':>14}  {'FoV larghezza (mm)':>18}")
        print_separator()
        for dist in distances:
            r = pixel_size_mm(dist)
            f = field_of_view(dist)
            print(f"  {dist:>14.1f}  {r['gsd_x_mm']:>14.6f}  "
                  f"{r['gsd_x_um']:>14.3f}  {f['fov_width_mm']:>18.2f}")
        print_separator("═")
        print()
 
    # ── Nota metodologica ─────────────────────────────────────────────────
    print("  NOTA METODOLOGICA")
    print_separator()
    print("  La formula GSD = Z / f_px assume che:")
    print("  1. L'oggetto sia nel piano focale a distanza Z.")
    print("  2. L'immagine sia della stessa dimensione usata in calibrazione")
    print(f"     ({CALIB['image_width']} × {CALIB['image_height']} px).")
    print("  3. Non si usi zoom digitale o crop.")
    print()
    print("  Per misure sub-pixel precise, applicare prima la correzione")
    print("  della distorsione (cv2.undistort) con i coefficienti k1, k2.")
    print_separator("═")
    print()
 
 
if __name__ == "__main__":
    main()