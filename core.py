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
