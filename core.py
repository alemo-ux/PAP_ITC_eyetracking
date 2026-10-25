"""Estimación experimental de mirada. No clasifica fijaciones ni sacadas."""
import numpy as np

FEATURE_NAMES = ['iris_rx', 'iris_ry', 'iris_lx', 'iris_ly',
                 'nose_x', 'nose_y', 'face_width', 'roll']

def extract_features(lm):
    if len(lm) < 478:
        return None, 'sin_iris'
    p = np.array([[v.x, v.y] for v in lm], dtype=float)
    values = []
    for iris, a, b, top, bottom in [(468,33,133,159,145),(473,362,263,386,374)]:
        axis = p[b]-p[a]
        width = np.linalg.norm(axis)
        if width < 1e-5:
            return None, 'ojo_no_valido'
        u = axis/width
        v = np.array([-u[1],u[0]])
        if np.dot(p[bottom]-p[top],v) < 0:
            v = -v
        opening = abs(np.dot(p[bottom]-p[top],v))/width
        if opening < .12:
            return None, 'parpadeo_ojo_cerrado'
        center = (p[a]+p[b])/2
        values.extend([np.dot(p[iris]-center,u)/width,
                       np.dot(p[iris]-center,v)/width])
    center = (p[33]+p[263])/2
    scale = np.linalg.norm(p[263]-p[33])
    if scale < 1e-5:
        return None, 'rostro_no_valido'
    nose = (p[1]-center)/scale
    values.extend([nose[0],nose[1],scale,
                   np.arctan2(p[263,1]-p[33,1],p[263,0]-p[33,0])])
    result = np.asarray(values)
    return (result,'ok') if np.isfinite(result).all() else (None,'no_finito')

class GazeModel:
    """Regresión ridge lineal con variables estandarizadas; sin recortar salidas."""
    def fit(self, features, targets):
        x = np.asarray(features, float)
        y = np.asarray(targets, float)
        if len(x) < 9 or y.shape != (len(x),2) or not np.isfinite(x).all():
            raise ValueError('Calibración incompleta.')
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0),1e-4)
        z = np.column_stack([np.ones(len(x)),(x-self.mean)/self.std])
        ridge = np.eye(z.shape[1])*.5
        ridge[0,0] = 0
        self.coef = np.linalg.solve(z.T@z+ridge,z.T@y)
        return self
    def predict(self, features):
        f = np.asarray(features,float)
        return np.r_[1.,(f-self.mean)/self.std]@self.coef
    def export(self):
        return {k:getattr(self,k).tolist() for k in ['mean','std','coef']}

def validation_error(predicted, target, width, height):
    delta = (np.asarray(predicted)-np.asarray(target))*[width,height]
    return float(np.linalg.norm(delta))

def heat_grid(points, width, height, cell=16):
    nx,ny = max(1,int(np.ceil(width/cell))),max(1,int(np.ceil(height/cell)))
    grid=np.zeros((ny,nx),dtype=float)
    for x,y in points:
        if 0<=x<width and 0<=y<height:
            grid[min(ny-1,int(y/cell)),min(nx-1,int(x/cell))]+=1
    return grid

# ---------------------------------------------------------------------------
# Posición de la cabeza
# ---------------------------------------------------------------------------
# MediaPipe entrega una matriz 4x4 (rostro canónico -> cámara) con traslación en
# centímetros aproximados. A partir de ella se obtienen tres giros y tres
# desplazamientos. Los valores absolutos dependen del campo de visión supuesto
# por MediaPipe; lo útil es la DIFERENCIA respecto a la posición de calibración.

HEAD_NAMES = ['cabeza_pitch_deg', 'cabeza_yaw_deg', 'cabeza_roll_deg',
              'cabeza_x_cm', 'cabeza_y_cm', 'cabeza_dist_cm']
HEAD_KEYS = ['pitch', 'yaw', 'roll', 'x', 'y', 'dist']

# (aviso, crítico). Giros en grados, desplazamientos en cm, distancia como
# fracción relativa (0.07 = 7 % más cerca o más lejos). Ajústenlos con sus
# propias validaciones: son un punto de partida, no valores validados.
HEAD_LIMITS = {'pitch': (5., 10.), 'yaw': (5., 10.), 'roll': (6., 12.),
               'x': (2.5, 5.), 'y': (2.5, 5.), 'dist': (.07, .15)}

# Cambiar a True si la imagen de la webcam llega en espejo (raro con OpenCV).
CAMARA_ESPEJO = False

HEAD_LABELS = {'pitch': 'inclinación arriba/abajo', 'yaw': 'giro izquierda/derecha',
               'roll': 'inclinación lateral', 'x': 'desplazamiento lateral',
               'y': 'desplazamiento vertical', 'dist': 'distancia a la pantalla',
               'sin_rostro': 'rostro no detectado'}


def rotation_vector_deg(rotation):
    """Vector de rotación (eje * ángulo) en grados: [x=pitch, y=yaw, z=roll]."""
    r = np.asarray(rotation, float)
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    factor = .5 if angle < 1e-6 else angle / (2 * np.sin(angle))
    return np.degrees(w * factor)


def head_pose(matrix):
    """[pitch, yaw, roll, x, y, distancia] o None si la matriz no es válida.

    Convenciones comprobadas con MediaPipe 0.10.21 y cámara SIN espejo:
    yaw > 0 = gira hacia SU izquierda; pitch > 0 = mira hacia abajo;
    x > 0 = se mueve hacia SU izquierda; y > 0 = sube; distancia en cm.
    """
    if matrix is None:
        return None
    m = np.asarray(matrix, float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        return None
    u, _, vt = np.linalg.svd(m[:3, :3])  # quita escala/ruido numérico
    rv = rotation_vector_deg(u @ vt)
    distance = -m[2, 3]
    if distance <= 0:
        return None
    return np.array([rv[0], rv[1], rv[2], m[0, 3], m[1, 3], distance])


def head_deviation(pose, reference):
    d = np.asarray(pose, float) - np.asarray(reference, float)
    out = dict(zip(HEAD_KEYS[:5], d[:5].tolist()))
    out['dist'] = float(pose[5] / reference[5] - 1)
    return out


def head_advice(key, value, mirror=CAMARA_ESPEJO):
    """Instrucción para regresar a la posición de calibración."""
    side = -value if mirror else value
    if key == 'yaw':
        return 'Gira la cabeza un poco hacia tu ' + ('derecha' if side > 0 else 'izquierda')
    if key == 'x':
        return 'Muévete un poco hacia tu ' + ('derecha' if side > 0 else 'izquierda')
    if key == 'pitch':
        return 'Levanta un poco la barbilla' if value > 0 else 'Baja un poco la barbilla'
    if key == 'y':
        return 'Baja un poco la cabeza' if value > 0 else 'Sube un poco la cabeza'
    if key == 'roll':
        return 'Endereza la cabeza, no la inclines de lado'
    if key == 'dist':
        return 'Acércate un poco a la pantalla' if value > 0 else 'Aléjate un poco de la pantalla'
    if key == 'sin_rostro':
        return 'No se detecta tu rostro: regresa frente a la cámara'
    return ''


class HeadMonitor:
    """Compara la cabeza con la referencia de calibración.

    Nivel 0 = bien, 1 = aviso, 2 = crítico. Usa la mediana de los últimos
    `window` cuadros para no reaccionar a ruido, e histéresis para que la alerta
    no parpadee en el límite. Si el rostro se pierde `lost_after` cuadros
    seguidos, el nivel pasa a crítico.
    """
    HYSTERESIS = .8

    def __init__(self, reference, limits=None, window=5, lost_after=15, mirror=CAMARA_ESPEJO):
        self.reference = np.asarray(reference, float)
        self.limits = dict(limits or HEAD_LIMITS)
        self.window, self.lost_after, self.mirror = window, lost_after, mirror
        self.history = []
        self.missing = 0
        self.level, self.key, self.message, self.deviation = 0, '', '', None

    def update(self, pose):
        if pose is None:
            self.missing += 1
            if self.missing >= self.lost_after:
                self.level, self.key = 2, 'sin_rostro'
                self.message, self.deviation = head_advice('sin_rostro', 0), None
            return self
        self.missing = 0
        self.history = (self.history + [np.asarray(pose, float)])[-self.window:]
        dev = head_deviation(np.median(self.history, axis=0), self.reference)
        warn = {k: abs(dev[k]) / self.limits[k][0] for k in HEAD_KEYS}
        crit = max(abs(dev[k]) / self.limits[k][1] for k in HEAD_KEYS)
        worst = max(warn, key=warn.get)
        if crit >= (self.HYSTERESIS if self.level == 2 else 1):
            level = 2
        elif warn[worst] >= (self.HYSTERESIS if self.level >= 1 else 1):
            level = 1
        else:
            level = 0
        self.level, self.deviation = level, dev
        self.key = worst if level else ''
        self.message = head_advice(worst, dev[worst], self.mirror) if level else ''
        return self

    @property
    def ok(self):
        return self.level == 0
