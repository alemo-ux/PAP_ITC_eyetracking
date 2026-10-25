"""Video de pantalla con la mirada superpuesta y mapas de calor.

Solo usa archivos ya guardados: pantalla.avi, pantalla_tiempos.csv y mirada.csv.
NO usa ni necesita imágenes de la webcam.

Uso desde PowerShell (o con generar_videos.bat):
    .\\.venv\\Scripts\\python.exe render.py
        -> procesa todas las tareas de resultados/ que tengan pantalla.avi
           y todavía no tengan video de mirada.
    .\\.venv\\Scripts\\python.exe render.py resultados\\P001_...\\tarea_1_...
        -> una tarea (o todas las de una sesión si das la carpeta de sesión).
    Añade --forzar para regenerar aunque ya exista.

Archivos que genera en la carpeta de cada tarea:
    pantalla_mirada.mp4 (o .avi si el códec mp4 no está disponible)
        Video de la pantalla a velocidad real con: punto de mirada suavizado,
        estela del último segundo, mapa de calor "reciente" que se desvanece,
        aviso cuando la cabeza estaba fuera de posición y tiempo de la tarea.
    heatmap_sobre_pantalla.png
        Densidad acumulada de toda la tarea sobre un fotograma de la mitad.
    video_mirada.json
        Parámetros usados y conteos, para documentar el procesamiento.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent

FPS_SALIDA = 10            # igual que la grabación de pantalla
SUAVIZADO_S = .20          # mediana de la mirada en esta ventana
ESTELA_S = 1.0             # trayectoria reciente dibujada
VIDA_MEDIA_CALOR_S = 3.0   # el calor reciente pierde la mitad en este tiempo
SIGMA_CALOR_PX = 45        # dispersión del calor, en píxeles de pantalla
CELDA_PX = 8               # resolución interna del mapa, en píxeles de pantalla
ANCHO_MAXIMO = 1920        # reduce pantallas grandes para ahorrar tiempo y espacio
EXCLUIR_NIVEL_CRITICO = True  # muestras con cabeza en nivel 2 no suman calor

COLORES_NIVEL = {0: (60, 220, 60), 1: (0, 170, 255), 2: (0, 0, 255)}  # BGR
MOTIVOS = {'pitch': 'giro arriba/abajo', 'yaw': 'giro izquierda/derecha',
           'roll': 'inclinacion lateral', 'x': 'desplazamiento lateral',
           'y': 'desplazamiento vertical', 'dist': 'distancia a pantalla',
           'sin_rostro': 'rostro no detectado'}


# ---------------------------------------------------------------------------
# Lectura de datos
# ---------------------------------------------------------------------------
def _float(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def read_gaze(path):
    """Arreglos ordenados por tiempo: t, x_px, y_px, ok, nivel_cabeza, motivo."""
    t, x, y, ok, level, reason = [], [], [], [], [], []
    with Path(path).open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            tt = _float(row.get('tiempo_tarea_s'))
            if math.isnan(tt):
                continue
            xx, yy = _float(row.get('x_pantalla_px')), _float(row.get('y_pantalla_px'))
            t.append(tt); x.append(xx); y.append(yy)
            ok.append(row.get('estado') == 'ok' and not (math.isnan(xx) or math.isnan(yy)))
            lv = _float(row.get('cabeza_nivel'))   # columnas nuevas; CSV viejos no las tienen
            level.append(0 if math.isnan(lv) else int(lv))
            reason.append(row.get('cabeza_motivo') or '')
    order = np.argsort(t, kind='stable')
    return {'t': np.asarray(t, float)[order], 'x': np.asarray(x, float)[order],
            'y': np.asarray(y, float)[order], 'ok': np.asarray(ok, bool)[order],
            'nivel': np.asarray(level, int)[order],
            'motivo': np.asarray(reason, object)[order]}


def read_frame_times(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return [float(r['captura_tarea_s']) for r in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# Mapa de calor
# ---------------------------------------------------------------------------
class Heat:
    """Rejilla de baja resolución en coordenadas de pantalla."""

    def __init__(self, width, height, cell=CELDA_PX, sigma_px=SIGMA_CALOR_PX):
        self.width, self.height, self.cell = width, height, cell
        self.grid = np.zeros((max(1, math.ceil(height / cell)), max(1, math.ceil(width / cell))), np.float32)
        self.sigma = max(.5, sigma_px / cell)
        unit = np.zeros((int(self.sigma * 8) | 1,) * 2, np.float32)
        unit[unit.shape[0] // 2, unit.shape[1] // 2] = 1
        self.unit_peak = float(cv2.GaussianBlur(unit, (0, 0), self.sigma).max())
        self.total = 0

    def add(self, x, y, weight=1.):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[int(y // self.cell), int(x // self.cell)] += weight
            self.total += 1
            return True
        return False

    def decay(self, factor):
        self.grid *= factor

    def normalized(self, floor_samples=0.):
        """0..1. `floor_samples` evita que unas pocas muestras se vean saturadas."""
        blurred = cv2.GaussianBlur(self.grid, (0, 0), self.sigma)
        top = max(float(blurred.max()), self.unit_peak * floor_samples, 1e-9)
        return blurred / top


def blend_heat(frame, norm, alpha_max=.55, cutoff=.06):
    """Mezcla el mapa (0..1, baja resolución) sobre el fotograma BGR."""
    mask = norm > cutoff
    if not mask.any():
        return frame
    h, w = frame.shape[:2]
    gy, gx = norm.shape
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    # Solo se procesa el rectángulo con calor (mucho más rápido en pantallas grandes).
    x0 = max(0, int((cols[0] - 2) * w / gx)); x1 = min(w, int((cols[-1] + 3) * w / gx))
    y0 = max(0, int((rows[0] - 2) * h / gy)); y1 = min(h, int((rows[-1] + 3) * h / gy))
    big = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
    color = cv2.applyColorMap(np.uint8(np.clip(big, 0, 1) * 255), cv2.COLORMAP_TURBO)
    alpha = (np.clip((big - cutoff) / (1 - cutoff), 0, 1) * alpha_max)[..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.uint8(roi * (1 - alpha) + color * alpha)
    return frame


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _text(img, text, org, scale, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _strip(img, y0, y1, color, alpha):
    roi = img[y0:y1]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_overlay(frame, t, point, trail, level, reason, half_life):
    h, w = frame.shape[:2]
    u = w / 1280  # unidad de escala para texto y marcadores
    color = COLORES_NIVEL.get(level, COLORES_NIVEL[2])
    # Estela: segmentos más delgados mientras más antiguos.
    for i in range(1, len(trail)):
        thick = max(1, int(round(1 + 3 * i / len(trail) * u)))
        cv2.line(frame, trail[i - 1], trail[i], color, thick, cv2.LINE_AA)
    r = max(8, int(18 * u))
    status = 'sin mirada valida (parpadeo o rostro no detectado)'
    if point is not None:
        px, py = point
        inside = 0 <= px < w and 0 <= py < h
        cx, cy = int(np.clip(px, r, w - r)), int(np.clip(py, r, h - r))
        cv2.circle(frame, (cx, cy), r, (0, 0, 0), max(3, int(5 * u)), cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, color, max(2, int(3 * u)), cv2.LINE_AA)
        if inside:
            cv2.circle(frame, (cx, cy), max(2, int(3 * u)), color, -1, cv2.LINE_AA)
            status = 'mirada estimada'
        else:
            status = 'mirada fuera de pantalla (se muestra en el borde)'
    if level >= 1:
        bar = int(36 * u)
        _strip(frame, 0, bar, color, .75)
        label = 'ALERTA: cabeza muy fuera de posicion' if level == 2 else 'Aviso: cabeza fuera de posicion'
        if reason in MOTIVOS:
            label += f' ({MOTIVOS[reason]})'
        _text(frame, label + ' - la estimacion es menos confiable', (int(12 * u), int(25 * u)), .62 * u, thickness=max(1, int(2 * u)))
    bar = int(30 * u)
    _strip(frame, h - bar, h, (0, 0, 0), .55)
    _text(frame, f't = {t:7.1f} s  |  {status}  |  calor: ultimos ~{half_life:g} s',
          (int(10 * u), h - int(10 * u)), .5 * u)
    return frame


def open_writer(base, fps, size):
    for ext, fourcc in (('.mp4', 'mp4v'), ('.avi', 'MJPG')):
        path = base.with_suffix(ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
        path.unlink(missing_ok=True)
    raise RuntimeError('No se pudo crear el video de salida (mp4v ni MJPG).')


# ---------------------------------------------------------------------------
# Proceso principal
# ---------------------------------------------------------------------------
def existing_output(folder):
    for ext in ('.mp4', '.avi'):
        if (Path(folder) / f'pantalla_mirada{ext}').exists():
            return Path(folder) / f'pantalla_mirada{ext}'
    return None


def render_task(folder, max_width=ANCHO_MAXIMO, progress=None, force=False,
                half_life=VIDA_MEDIA_CALOR_S, exclude_critical=EXCLUIR_NIVEL_CRITICO):
    """Genera el video con mirada y el heatmap sobre pantalla. Devuelve un resumen."""
    folder = Path(folder)
    video, times_csv, gaze_csv = folder / 'pantalla.avi', folder / 'pantalla_tiempos.csv', folder / 'mirada.csv'
    for p in (video, times_csv, gaze_csv):
        if not p.exists():
            raise FileNotFoundError(f'Falta {p.name} en {folder}')
    previous = existing_output(folder)
    if previous and not force:
        return {'omitido': True, 'archivo': previous.name}

    frame_times = read_frame_times(times_csv)
    gaze = read_gaze(gaze_csv)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir {video.name}')
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counted = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(len(frame_times), counted) if counted > 0 else len(frame_times)
    if n_frames == 0 or vw == 0:
        cap.release()
        raise RuntimeError('El video de pantalla no tiene fotogramas.')

    try:
        monitor = json.loads((folder / 'inicio.json').read_text(encoding='utf-8'))['monitor']
        sw, sh = int(monitor['width']), int(monitor['height'])
    except Exception:
        sw, sh = vw, vh
    scale = min(1., max_width / vw) if max_width else 1.
    ow, oh = int(vw * scale) // 2 * 2, int(vh * scale) // 2 * 2
    writer, out_path = open_writer(folder / 'pantalla_mirada', FPS_SALIDA, (ow, oh))

    recent, total = Heat(sw, sh), Heat(sw, sh)
    usable = gaze['ok'] & ((gaze['nivel'] < 2) if exclude_critical else True)
    t0, t_end = frame_times[0], frame_times[n_frames - 1]
    steps = int(math.floor((t_end - t0) * FPS_SALIDA)) + 1
    middle_step = steps // 2
    background = None
    source, next_index, current = None, 0, None
    gi, prev_t = 0, None
    trail = []
    excluded = 0

    def to_out(x, y):
        return int(round(x * scale)), int(round(y * scale))

    try:
        for k in range(steps):
            t = t0 + k / FPS_SALIDA
            # Fotograma de pantalla más reciente capturado antes de t (repite si la captura se atrasó).
            while next_index < n_frames and frame_times[next_index] <= t + 1e-9:
                ok, source = cap.read()
                if not ok:
                    n_frames = next_index
                    break
                current = source
                next_index += 1
            if current is None:
                continue
            frame = cv2.resize(current, (ow, oh), interpolation=cv2.INTER_AREA) if scale < 1 else current[:oh, :ow].copy()
            if k == middle_step:
                background = frame.copy()

            # Calor reciente: desvanecer y sumar muestras nuevas.
            if prev_t is not None:
                recent.decay(.5 ** ((t - prev_t) / half_life))
            while gi < len(gaze['t']) and gaze['t'][gi] <= t:
                if usable[gi]:
                    recent.add(gaze['x'][gi], gaze['y'][gi])
                    total.add(gaze['x'][gi], gaze['y'][gi])
                elif gaze['ok'][gi]:
                    excluded += 1
                gi += 1
            prev_t = t
            blend_heat(frame, recent.normalized(floor_samples=6))

            # Punto de mirada: mediana de la ventana reciente (no inventa datos en huecos).
            lo = np.searchsorted(gaze['t'], t - SUAVIZADO_S, side='right')
            hi = np.searchsorted(gaze['t'], t, side='right')
            sel = slice(lo, hi)
            valid = gaze['ok'][sel]
            point = None
            if valid.any():
                point = to_out(np.median(gaze['x'][sel][valid]), np.median(gaze['y'][sel][valid]))
                trail.append((t, point))
            trail = [(tt, p) for tt, p in trail if t - tt <= ESTELA_S]
            if point is None:
                trail = []  # no unir puntos a través de parpadeos o pérdidas
            lo_l = np.searchsorted(gaze['t'], t - .3, side='right')
            levels = gaze['nivel'][lo_l:hi]
            level = int(levels.max()) if len(levels) else 0
            reason = ''
            if level:
                idx = lo_l + int(np.argmax(levels))
                reason = gaze['motivo'][idx]
            draw_overlay(frame, t, point, [p for _, p in trail], level, reason, half_life)
            writer.write(frame)
            if progress and k % 10 == 0:
                progress(k / steps)
    finally:
        cap.release()
        writer.release()

    # Heatmap acumulado sobre un fotograma representativo.
    png = folder / 'heatmap_sobre_pantalla.png'
    if background is not None:
        img = background.copy()
        if total.total:
            blend_heat(img, total.normalized(), alpha_max=.6, cutoff=.04)
        u = ow / 1280
        _strip(img, 0, int(52 * u), (0, 0, 0), .6)
        _text(img, f'Densidad acumulada de {total.total} muestras de toda la tarea (no fijaciones).',
              (int(10 * u), int(20 * u)), .5 * u)
        _text(img, f'Fondo: fotograma en t = {middle_step / FPS_SALIDA + t0:.1f} s. Si hubo scroll o cambio de pagina, '
                   'el fondo no corresponde a todo el calor.', (int(10 * u), int(42 * u)), .45 * u)
        cv2.imwrite(str(png), img)

    summary = {'archivo': out_path.name, 'creado_utc': datetime.now(timezone.utc).isoformat(),
               'fps': FPS_SALIDA, 'fotogramas_salida': steps, 'duracion_s': steps / FPS_SALIDA,
               'escala_respecto_a_pantalla': scale, 'tamano_salida': [ow, oh],
               'fotogramas_pantalla_usados': n_frames,
               'muestras_en_calor_acumulado': total.total,
               'muestras_excluidas_por_cabeza_critica': excluded,
               'parametros': {'suavizado_s': SUAVIZADO_S, 'estela_s': ESTELA_S,
                              'vida_media_calor_s': half_life, 'sigma_calor_px': SIGMA_CALOR_PX,
                              'excluir_nivel_critico': exclude_critical},
               'nota': 'Punto = mediana de estimaciones recientes. No son fijaciones. '
                       'Coordenadas de pantalla, no de página. Sin imágenes de webcam.'}
    (folder / 'video_mirada.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if progress:
        progress(1.)
    return summary


def find_tasks(paths):
    for p in paths:
        p = Path(p)
        if (p / 'pantalla.avi').exists():
            yield p
        else:
            yield from sorted(v.parent for v in p.rglob('pantalla.avi'))


def main():
    parser = argparse.ArgumentParser(description='Genera video de pantalla con mirada y mapas de calor.')
    parser.add_argument('carpetas', nargs='*', default=[str(BASE / 'resultados')])
    parser.add_argument('--forzar', action='store_true', help='regenerar aunque ya exista')
    parser.add_argument('--ancho-maximo', type=int, default=ANCHO_MAXIMO)
    args = parser.parse_args()
    tasks = list(find_tasks(args.carpetas))
    if not tasks:
        print('No se encontraron tareas con pantalla.avi.')
    for folder in tasks:
        print(f'\n{folder}')
        try:
            result = render_task(folder, args.ancho_maximo, force=args.forzar,
                                 progress=lambda p: print(f'\r  {p:5.0%}', end='', flush=True))
            print('\n  ya existía, se omite (usa --forzar)' if result.get('omitido') else f"\n  listo: {result['archivo']}")
        except Exception as exc:
            print(f'\n  ERROR: {exc}')


if __name__ == '__main__':
    main()
