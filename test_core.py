import unittest
from types import SimpleNamespace
import numpy as np
import csv, json, tempfile
from pathlib import Path
import cv2
from core import (GazeModel, heat_grid, validation_error, extract_features,
                  head_pose, head_deviation, head_advice, HeadMonitor)
import render

def matrix(yaw=0., pitch=0., roll=0., x=0., y=0., dist=50.):
    """Matriz 4x4 como la de MediaPipe (rotaciones en grados, traslación en cm)."""
    a,b,c=np.radians([pitch,yaw,roll])
    rx=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    ry=np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
    rz=np.array([[np.cos(c),-np.sin(c),0],[np.sin(c),np.cos(c),0],[0,0,1]])
    m=np.eye(4);m[:3,:3]=rz@ry@rx;m[:3,3]=[x,y,-dist]
    return m

class CoreTests(unittest.TestCase):
    def test_calibration_predicts_unseen_points(self):
        rng=np.random.default_rng(18)
        x=rng.normal(size=(100,8))
        y=np.column_stack([.5+.12*x[:,0]-.08*x[:,1],.5+.13*x[:,2]])
        model=GazeModel().fit(x,y)
        test=rng.normal(size=8)
        expected=[.5+.12*test[0]-.08*test[1],.5+.13*test[2]]
        np.testing.assert_allclose(model.predict(test),expected,atol=.005)
    def test_heatmap_does_not_clip_invalid_points_to_edges(self):
        grid=heat_grid([(0,0),(99,79),(-1,30),(100,40),(20,80)],100,80)
        self.assertEqual(grid.sum(),2)
        self.assertEqual(grid[0,0],1)
        self.assertEqual(grid[-1,-1],1)
    def test_validation_uses_screen_pixels(self):
        self.assertAlmostEqual(validation_error([.6,.5],[.5,.5],1920,1080),192)
    def test_model_retains_offscreen_predictions(self):
        x=np.tile(np.linspace(0,1,20)[:,None],(1,8))
        m=GazeModel().fit(x,np.column_stack([x[:,0],x[:,0]]))
        self.assertGreater(m.predict(np.ones(8)*3)[0],1)
    def test_missing_iris_rejected(self):
        self.assertEqual(extract_features([]),(None,'sin_iris'))
    def test_closed_eye_rejected(self):
        points=[SimpleNamespace(x=0.,y=0.) for _ in range(478)]
        points[33].x=.1;points[133].x=.2
        self.assertEqual(extract_features(points)[1],'parpadeo_ojo_cerrado')

class HeadTests(unittest.TestCase):
    def test_pose_recovers_angles_and_translation(self):
        p=head_pose(matrix(yaw=7,x=2,y=-1,dist=55))
        np.testing.assert_allclose(p,[0,7,0,2,-1,55],atol=1e-6)
        self.assertIsNone(head_pose(None))
        self.assertIsNone(head_pose(np.full((4,4),np.nan)))
    def test_deviation_uses_relative_distance(self):
        d=head_deviation(head_pose(matrix(dist=45)),head_pose(matrix(dist=50)))
        self.assertAlmostEqual(d['dist'],-.1)
    def test_monitor_levels_and_hysteresis(self):
        m=HeadMonitor(head_pose(matrix()),window=1)
        self.assertEqual(m.update(head_pose(matrix(yaw=2))).level,0)
        self.assertEqual(m.update(head_pose(matrix(yaw=6))).level,1)
        self.assertEqual(m.update(head_pose(matrix(yaw=6))).key,'yaw')
        self.assertEqual(m.update(head_pose(matrix(yaw=11))).level,2)
        # Histéresis: 9° sigue crítico (≥ 0.8·10), 4.5° sigue en aviso (≥ 0.8·5).
        self.assertEqual(m.update(head_pose(matrix(yaw=9))).level,2)
        self.assertEqual(m.update(head_pose(matrix(yaw=4.5))).level,1)
        self.assertEqual(m.update(head_pose(matrix(yaw=1))).level,0)
    def test_monitor_median_ignores_single_spike(self):
        m=HeadMonitor(head_pose(matrix()),window=5)
        for _ in range(4):m.update(head_pose(matrix()))
        self.assertEqual(m.update(head_pose(matrix(yaw=30))).level,0)
    def test_monitor_distance_and_lost_face(self):
        m=HeadMonitor(head_pose(matrix(dist=50)),window=1,lost_after=3)
        self.assertEqual(m.update(head_pose(matrix(dist=40))).key,'dist')
        self.assertIn('Aléjate',m.message)
        m=HeadMonitor(head_pose(matrix()),window=1,lost_after=3)
        for _ in range(2):self.assertEqual(m.update(None).level,0)
        self.assertEqual(m.update(None).key,'sin_rostro')
        self.assertEqual(m.update(head_pose(matrix())).level,0)
    def test_advice_directions(self):
        self.assertIn('derecha',head_advice('yaw',5))
        self.assertIn('izquierda',head_advice('yaw',5,mirror=True))
        self.assertIn('Levanta',head_advice('pitch',5))
        self.assertIn('Acércate',head_advice('dist',.1))

class RenderTests(unittest.TestCase):
    def make_task(self,folder,w=320,h=200,frames=30):
        folder=Path(folder)
        writer=cv2.VideoWriter(str(folder/'pantalla.avi'),cv2.VideoWriter_fourcc(*'MJPG'),10,(w,h))
        with (folder/'pantalla_tiempos.csv').open('w',newline='') as f:
            out=csv.writer(f);out.writerow(['frame','captura_tarea_s','video_s_nominal'])
            t=0.
            for i in range(frames):
                writer.write(np.full((h,w,3),40+i,np.uint8))
                out.writerow([i,t,i/10]);t+=.1 if i!=10 else .5  # un atraso de captura
        writer.release()
        names=['captura_monotonic_s','tiempo_tarea_s','estado','x_normalizado','y_normalizado','x_pantalla_px',
               'y_pantalla_px','en_pantalla','cabeza_nivel','cabeza_motivo']
        with (folder/'mirada.csv').open('w',newline='') as f:
            out=csv.writer(f);out.writerow(names)
            for i in range(120):
                t=i/30
                if 40<=i<50:out.writerow([t,t,'sin_rostro','','','','','0',2,'sin_rostro']);continue
                level=2 if 60<=i<70 else 0
                out.writerow([t,t,'ok',.5,.5,w/2+20*np.sin(t),h/2,1,level,'yaw' if level else ''])
        (folder/'inicio.json').write_text(json.dumps({'monitor':{'width':w,'height':h}}))
        return folder
    def test_render_creates_video_and_heatmap(self):
        with tempfile.TemporaryDirectory() as d:
            folder=self.make_task(d)
            res=render.render_task(folder)
            out=folder/res['archivo']
            self.assertTrue(out.exists())
            cap=cv2.VideoCapture(str(out));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
            # Tiempo real: ~3.3 s de captura -> ~34 fotogramas a 10 fps aunque solo hubo 30 capturas.
            self.assertGreaterEqual(n,33)
            self.assertTrue((folder/'heatmap_sobre_pantalla.png').exists())
            self.assertEqual(res['muestras_excluidas_por_cabeza_critica'],10)
            self.assertTrue(render.render_task(folder).get('omitido'))
    def test_read_gaze_accepts_old_csv_without_head_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'mirada.csv'
            p.write_text('tiempo_tarea_s,estado,x_pantalla_px,y_pantalla_px\n0.1,ok,10,20\n0.2,sin_iris,,\n')
            g=render.read_gaze(p)
            self.assertEqual(g['ok'].tolist(),[True,False]);self.assertEqual(g['nivel'].tolist(),[0,0])

if __name__=='__main__':
    unittest.main()
