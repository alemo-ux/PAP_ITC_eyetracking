"""Interfaz local para pruebas piloto de seguimiento ocular con webcam."""
import os
# Un único espacio de coordenadas físicas para Tk, MSS y Windows.
if os.name == 'nt':
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
import csv
import json
import queue
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import mss
import numpy as np
from PIL import Image, ImageTk
from pynput import keyboard
from core import GazeModel, extract_features, validation_error, heat_grid

BASE=Path(__file__).resolve().parent
MODEL=BASE/'models'/'face_landmarker.task'
TASKS=[
 '1 - Economía y escasez',
 '2 - Video Historia Mundial',
 '3 - Artículo de ahorro',
 '4 - Lección de presupuesto',
 '5 - Video regla 50/30/20'
]
def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def utc():
    return datetime.now(timezone.utc).isoformat()

class Camera(threading.Thread):
    def __init__(self,index):
        super().__init__(daemon=True)
        self.index=index
        self.items=queue.Queue()
        self.stop=threading.Event()
        self.frame=None
        self.error=None
    def run(self):
        cap=None
        try:
            options=mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,min_face_detection_confidence=.6,
                min_face_presence_confidence=.6,min_tracking_confidence=.6)
            cap = cv2.VideoCapture(self.index)
            if not cap.isOpened():
                raise RuntimeError('No se pudo abrir la cámara. Revisa el índice y los permisos.')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            previous=-1
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                while not self.stop.is_set():
                    ok,frame=cap.read()
                    capture=time.perf_counter()
                    if not ok:
                        raise RuntimeError('La cámara dejó de entregar imágenes.')
                    stamp=max(previous+1,int(capture*1000))
                    previous=stamp
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    result=detector.detect_for_video(image,stamp)
                    features,status=(None,'sin_rostro')
                    if result.face_landmarks:
                        features,status=extract_features(result.face_landmarks[0])
                        for i in [468,473]:
                            if len(result.face_landmarks[0])>i:
                                v=result.face_landmarks[0][i]
                                cv2.circle(frame,(int(v.x*frame.shape[1]),int(v.y*frame.shape[0])),3,(0,255,0),-1)
                    self.frame=frame
                    self.items.put((capture,features,status))
        except Exception as exc:
            self.error=str(exc)
            traceback.print_exc()
        finally:
            if cap is not None: cap.release()

class ScreenRecorder(threading.Thread):
    """Vídeo de fotogramas capturados y CSV de tiempos reales, sin cámara ni audio."""
    def __init__(self,folder,monitor,start):
        super().__init__(daemon=True)
        self.folder,self.monitor,self.start_time=folder,dict(monitor),start
        self.stop=threading.Event()
        self.ready=threading.Event()
        self.error=None
        self.frames=0
    def run(self):
        writer=None
        try:
            width=self.monitor['width']; height=self.monitor['height']
            size=(width-width%2,height-height%2)
            writer=cv2.VideoWriter(str(self.folder/'pantalla.avi'),
                                   cv2.VideoWriter_fourcc(*'MJPG'),10,size)
            if not writer.isOpened():
                raise RuntimeError('No se pudo crear pantalla.avi')
            with mss.mss() as grabber, (self.folder/'pantalla_tiempos.csv').open('w',newline='',encoding='utf-8') as f:
                out=csv.writer(f)
                out.writerow(['frame','captura_tarea_s','video_s_nominal'])
                self.ready.set()
                next_frame=time.perf_counter()
                while not self.stop.is_set():
                    now=time.perf_counter()
                    if now<next_frame:
                        self.stop.wait(next_frame-now)
                        continue
                    frame=np.asarray(grabber.grab(self.monitor))[:,:,:3]
                    captured=time.perf_counter()
                    writer.write(np.ascontiguousarray(frame[:size[1],:size[0]]))
                    out.writerow([self.frames,captured-self.start_time,self.frames/10])
                    f.flush()
                    self.frames+=1
                    # Si la máquina se atrasa no inventar fotogramas.
                    next_frame=max(next_frame+.1,time.perf_counter())
        except Exception as exc:
            self.error=str(exc)
        finally:
            self.ready.set()
            if writer: writer.release()

class App:
    def __init__(self,root):
        self.root=root
        root.title('Eye Tracking | Pruebas piloto')
        root.geometry('760x710')
        self.camera=None; self.model=None; self.validation=None; self.session=None
        self.active=None; self.cal=None; self.latest=None; self.recorder=None
        self.points=[]; self.keys=queue.Queue(); self.pending=False
        with mss.mss() as s:
            # Pantalla situada en el origen de Windows, que usa Tk para fullscreen.
            candidates=[m for m in s.monitors[1:] if m['left']==0 and m['top']==0]
            self.monitor=dict(candidates[0] if candidates else s.monitors[1])
        self.w,self.h=self.monitor['width'],self.monitor['height']
        self.person=tk.StringVar(value='P001')
        self.cam_index=tk.IntVar(value=0)
        self.task=tk.StringVar(value=TASKS[0])
        self.record_screen=tk.BooleanVar(value=False)
        self.consent=tk.BooleanVar(value=False)
        self.state=tk.StringVar(value='Conecta la cámara para comenzar.')
        self.notes=tk.StringVar()
        self.build()
        self.listener=keyboard.Listener(on_release=self.key_release)
        self.listener.start()
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.after(40,self.tick)

    def build(self):
        f=ttk.Frame(self.root,padding=20); f.pack(fill='both',expand=True)
        ttk.Label(f,text='Pruebas de seguimiento ocular',font=('Segoe UI',19,'bold')).pack(anchor='w')
        ttk.Label(f,text='Estimación experimental con webcam • Solo pantalla principal').pack(anchor='w',pady=(0,12))
        row=ttk.Frame(f); row.pack(fill='x')
        ttk.Label(row,text='Participante:').pack(side='left')
        ttk.Entry(row,textvariable=self.person,width=14).pack(side='left',padx=8)
        ttk.Label(row,text='Cámara:').pack(side='left')
        ttk.Spinbox(row,from_=0,to=8,textvariable=self.cam_index,width=4).pack(side='left',padx=8)
        ttk.Button(row,text='Conectar',command=self.connect).pack(side='left')
        self.preview=ttk.Label(f); self.preview.pack(pady=10)
        self.preview.configure(text='La vista de cámara aparece aquí (no se guarda).')
        row=ttk.Frame(f);row.pack(fill='x',pady=5)
        ttk.Button(row,text='1  Calibrar',command=lambda:self.calibrate(False)).pack(side='left',padx=3)
        ttk.Button(row,text='2  Validar precisión',command=lambda:self.calibrate(True)).pack(side='left',padx=3)
        ttk.Label(f,textvariable=self.state,wraplength=700).pack(anchor='w',pady=8)
        ttk.Checkbutton(f,text='El participante otorgó consentimiento para los registros seleccionados',
                        variable=self.consent).pack(anchor='w')
        ttk.Checkbutton(f,text='Grabar pantalla (sin audio ni imagen de webcam)',
                        variable=self.record_screen).pack(anchor='w')
        ttk.Label(f,text='Tarea (prepara la página antes de iniciar):').pack(anchor='w',pady=(12,3))
        ttk.Combobox(f,textvariable=self.task,values=TASKS,state='readonly',width=58).pack(anchor='w')
        ttk.Label(f,text='Notas de la tarea:').pack(anchor='w',pady=(8,2))
        ttk.Entry(f,textvariable=self.notes,width=85).pack(fill='x')
        row=ttk.Frame(f);row.pack(fill='x',pady=12)
        ttk.Button(row,text='3  Iniciar tarea  F8',command=self.start_task).pack(side='left',padx=3)
        ttk.Button(row,text='Detener tarea  F9',command=self.stop_task).pack(side='left',padx=3)
        ttk.Label(f,text='F8 inicia tras 5 s. F9 detiene y muestra el panel.\nF10 cancela la cuenta regresiva. No grabes contraseñas; inicia después del acceso.',
                  wraplength=690).pack(anchor='w')

    def key_release(self,key):
        if key in (keyboard.Key.f8,keyboard.Key.f9,keyboard.Key.f10):
            self.keys.put(key)
    def connect(self):
        if self.active or self.cal or self.pending: return
        if self.camera and self.camera.is_alive():
            messagebox.showinfo('Cámara','La cámara ya está conectada.')
            return
        if not MODEL.exists():
            messagebox.showerror('Modelo','Ejecuta descargar_modelo.py antes de iniciar.')
            return
        self.model=None;self.validation=None;self.session=None;self.latest=None
        self.camera=Camera(self.cam_index.get())
        self.camera.start()
        self.state.set('Abriendo la cámara...')
    def ensure_session(self):
        code=self.person.get().strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',code):
            raise ValueError('Usa un código como P001 (sin nombres ni espacios).')
        if self.session is None:
            self.session=BASE/'resultados'/f'{code}_{datetime.now():%Y%m%d_%H%M%S_%f}'
            self.session.mkdir(parents=True)
            dump(self.session/'sesion.json',{'participante':code,'creada_utc':utc(),
                 'monitor':self.monitor,'python':os.sys.version,
                 'mediapipe':mp.__version__,'opencv':cv2.__version__,
                 'metodo':'ridge lineal experimental; sin clasificación de fijaciones',
                 'camera_index':self.cam_index.get()})
            self.session_code=code
        elif code!=self.session_code:
            raise ValueError('Cierra y vuelve a abrir para cambiar de participante. Recalibra con cada persona.')

    def calibrate(self,validate):
        if self.active or self.cal or self.pending:return
        if not self.consent.get():
            messagebox.showwarning('Consentimiento','Confirma el consentimiento antes de capturar datos.')
            return
        if not self.latest or time.perf_counter()-self.latest[0]>.7 or self.latest[1] is None:
            messagebox.showwarning('Cámara','Espera a detectar los ojos abiertos.')
            return
        if validate and self.model is None:
            messagebox.showwarning('Calibración','Primero realiza la calibración.')
            return
        try:self.ensure_session()
        except ValueError as exc:
            messagebox.showerror('Participante',str(exc));return
        if not validate:
            self.model=None; self.validation=None
        pts=[(.1,.1),(.5,.1),(.9,.1),(.1,.5),(.5,.5),(.9,.5),(.1,.9),(.5,.9),(.9,.9)]
        if validate:pts=[(.25,.25),(.75,.25),(.5,.5),(.25,.75),(.75,.75)]
        win=tk.Toplevel(self.root); win.attributes('-fullscreen',True);win.attributes('-topmost',True)
        canvas=tk.Canvas(win,bg='#142337',highlightthickness=0);canvas.pack(fill='both',expand=True)
        self.cal={'validate':validate,'targets':pts,'i':0,'data':[],'samples':[],
                  'since':time.perf_counter(),'win':win,'canvas':canvas}
        win.bind('<Escape>',lambda e:self.cancel_cal())
        win.protocol('WM_DELETE_WINDOW',self.cancel_cal)
        self.paint_target()
    def paint_target(self):
        c=self.cal;cv=c['canvas'];cv.delete('all')
        x,y=c['targets'][c['i']];x*=self.w;y*=self.h
        cv.create_oval(x-14,y-14,x+14,y+14,fill='#69e3bb',outline='white',width=2)
        cv.create_text(self.w/2,self.h*.03,text=f"Mira el punto sin mover la cabeza • {c['i']+1}/{len(c['targets'])} • Esc para cancelar",fill='white',font=('Segoe UI',16))
    def cancel_cal(self):
        if self.cal:self.cal['win'].destroy();self.cal=None
        self.state.set('Calibración o validación cancelada.')
    def advance_cal(self,now):
        c=self.cal
        if now-c['since']<2.6:return
        if len(c['samples'])<12:
            c['canvas'].create_text(self.w/2,self.h*.96,text='No se capturaron suficientes muestras. Abre los ojos y mira el punto.',fill='white',tags='warning')
            c['since']=now;c['samples']=[]
            return
        c['data'].append({'target':c['targets'][c['i']], 'features':np.median(c['samples'],axis=0).tolist(),
                          'sample_count':len(c['samples'])})
        c['i']+=1;c['samples']=[];c['since']=now
        if c['i']<len(c['targets']):
            self.paint_target();return
        c['win'].destroy();self.cal=None
        if not c['validate']:
            self.model=GazeModel().fit([v['features'] for v in c['data']],[v['target'] for v in c['data']])
            dump(self.session/'calibracion.json',{'datos':c['data'],'modelo':self.model.export()})
            self.state.set('Calibración guardada. Ahora valida la precisión con otros puntos.')
        else:
            errors=[]
            for v in c['data']:
                pred=self.model.predict(v['features'])
                v['predicted']=pred.tolist()
                v['error_px']=validation_error(pred,v['target'],self.w,self.h);errors.append(v['error_px'])
            self.validation={'puntos':c['data'],'error_medio_px':float(np.mean(errors)),
                             'error_maximo_px':float(np.max(errors)), 'fecha_utc':utc()}
            dump(self.session/'validacion.json',self.validation)
            self.state.set(f"Validación: error medio {np.mean(errors):.0f} px; máximo {np.max(errors):.0f} px. Decide si permite distinguir tus áreas de interés.")
            messagebox.showinfo('Validación',self.state.get()+'\nNo es una certificación de precisión. Recalibra si el error es grande.')

    def start_task(self):
        if self.active or self.cal or self.pending:return
        try:
            if not self.consent.get():raise ValueError('Confirma el consentimiento.')
            if self.model is None or self.validation is None:raise ValueError('Primero calibra y valida.')
            self.ensure_session()
            if self.camera.error:raise ValueError(self.camera.error)
            if not self.latest or time.perf_counter()-self.latest[0]>.7 or self.latest[1] is None:
                raise ValueError('No se detectan los ojos abiertos.')
        except ValueError as exc:
            messagebox.showwarning('No se puede iniciar',str(exc));return
        self.pending=True
        self.pending_task=self.task.get();self.pending_notes=self.notes.get()
        self.pending_record=self.record_screen.get()
        self.count=5
        self.countdown()
    def countdown(self):
        if not self.pending:return
        if self.count==0:
            self.root.withdraw()
            # Permite que desaparezca el panel antes de capturar la pantalla.
            self.root.after(350,self.begin_record)
            return
        self.state.set(f'Inicia en {self.count} s. Prepara el navegador. F10 cancela.')
        self.count-=1
        self.root.after(1000,self.countdown)
    def begin_record(self):
        if not self.pending:return
        self.pending=False
        folder=self.session/f'tarea_{self.pending_task[0]}_{datetime.now():%H%M%S_%f}'
        folder.mkdir()
        start=time.perf_counter()
        self.active={'folder':folder,'start':start,'utc':utc(),'task':self.pending_task,
                     'notes':self.pending_notes,'count':0,'valid':0}
        self.points=[]
        self.datafile=(folder/'mirada.csv').open('w',newline='',encoding='utf-8')
        self.csv=csv.writer(self.datafile)
        self.csv.writerow(['captura_monotonic_s','tiempo_tarea_s','estado','x_normalizado','y_normalizado',
                           'x_pantalla_px','y_pantalla_px','en_pantalla']+list(__import__('core').FEATURE_NAMES))
        self.recorder=None
        if self.pending_record:
            self.recorder=ScreenRecorder(folder,self.monitor,start);self.recorder.start()
        dump(folder/'inicio.json',{'inicio_utc':self.active['utc'],'inicio_monotonic':start,
             'tarea':self.pending_task,'notas':self.pending_notes,'monitor':self.monitor,
             'grabacion_solicitada':self.pending_record,
             'calibracion_modelo':self.model.export(),'validacion':self.validation})
        self.state.set('Tarea en curso. F9 para detener.')
    def stop_task(self):
        if self.pending:
            self.pending=False;self.root.deiconify();self.state.set('Inicio cancelado.');return
        if not self.active:return
        end=time.perf_counter()
        a=self.active;self.active=None
        self.datafile.flush();self.datafile.close()
        if self.recorder:
            self.recorder.stop.set();self.recorder.join(timeout=5)
        self.root.deiconify();self.root.lift()
        outcome=tk.StringVar(value='no_indicado')
        dialog=tk.Toplevel(self.root);dialog.title('Resultado de la tarea');dialog.transient(self.root)
        ttk.Label(dialog,text='Selecciona el resultado observado:').pack(padx=20,pady=10)
        combo=ttk.Combobox(dialog,textvariable=outcome,state='readonly',
                          values=['completada','completada_con_ayuda','no_completada','interrumpida','no_indicado'])
        combo.pack(padx=20,pady=5)
        ttk.Button(dialog,text='Guardar',command=dialog.destroy).pack(pady=12)
        dialog.protocol('WM_DELETE_WINDOW',dialog.destroy)
        dialog.grab_set();self.root.wait_window(dialog)
        r=self.recorder
        dump(a['folder']/'resumen.json',{'tarea':a['task'],'inicio_utc':a['utc'],
             'duracion_s':end-a['start'],'resultado':outcome.get(),'notas':a['notes'],
             'muestras':a['count'],'muestras_en_pantalla':a['valid'],
             'validacion':self.validation,'error_grabacion':r.error if r else None,
             'fotogramas':r.frames if r else 0,
             'grabacion_finalizada':not r.is_alive() if r else True,
             'heatmap':'Densidad de muestras, no fijaciones. Coordenadas de pantalla, no de página.'})
        try:
            grid=heat_grid(self.points,self.w,self.h)
            if grid.max()>0:
                grid=cv2.GaussianBlur(grid.astype(np.float32),(0,0),2)
                normalized=np.uint8(grid/grid.max()*255)
                heat=cv2.applyColorMap(normalized,cv2.COLORMAP_TURBO)
                heat[grid<grid.max()*.015]=0
                heat=cv2.resize(heat,(self.w,self.h),interpolation=cv2.INTER_LINEAR)
                cv2.putText(heat,'Densidad de muestras en pantalla - no fijaciones',(20,35),
                            cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                cv2.imwrite(str(a['folder']/'heatmap_pantalla.png'),heat)
            self.state.set(f"Tarea guardada en {a['folder'].name}. Muestras en pantalla: {a['valid']}/{a['count']}.")
            if r and r.error:messagebox.showwarning('Grabación',r.error+'\nEl CSV de mirada se guardó.')
        except Exception as exc:
            messagebox.showerror('Mapa de calor',str(exc)+'\nEl CSV de mirada se guardó.')

    def tick(self):
        now=time.perf_counter()
        while not self.keys.empty():
            key=self.keys.get()
            if key==keyboard.Key.f8:self.start_task()
            elif key==keyboard.Key.f9:self.stop_task()
            elif key==keyboard.Key.f10 and self.pending:
                self.pending=False;self.root.deiconify();self.state.set('Inicio cancelado.')
        if self.camera:
            while not self.camera.items.empty():
                stamp,features,status=self.camera.items.get()
                self.latest=(stamp,features,status)
                if self.cal and features is not None:
                    # Ignorar el desplazamiento inicial de los ojos hacia el nuevo objetivo.
                    if stamp-self.cal['since']>.8:self.cal['samples'].append(features)
                if self.active and stamp>=self.active['start']:
                    pred=self.model.predict(features) if features is not None else None
                    inside=pred is not None and bool(np.isfinite(pred).all()) and 0<=pred[0]<1 and 0<=pred[1]<1
                    self.active['count']+=1
                    if inside:
                        self.points.append((pred[0]*self.w,pred[1]*self.h));self.active['valid']+=1
                    coords=list(pred)+[pred[0]*self.w,pred[1]*self.h] if pred is not None else ['']*4
                    self.csv.writerow([stamp,stamp-self.active['start'],status]+coords+[int(inside)]+
                                      (features.tolist() if features is not None else ['']*8))
            if self.active:self.datafile.flush()
            if self.camera.error:
                if self.active:self.stop_task()
                self.state.set(self.camera.error)
            elif self.camera.frame is not None and not self.active and not self.cal:
                img=cv2.cvtColor(self.camera.frame,cv2.COLOR_BGR2RGB)
                img=Image.fromarray(img).resize((320,240))
                self.photo=ImageTk.PhotoImage(img)
                self.preview.configure(image=self.photo,text='')
                if self.model is None and not self.pending:
                    self.state.set('Cámara conectada. '+(self.latest[2] if self.latest else 'Esperando rostro'))
        if self.cal:self.advance_cal(now)
        self.root.after(40,self.tick)
    def close(self):
        self.pending=False
        if self.cal:self.cancel_cal()
        if self.active:self.stop_task()
        if self.camera:self.camera.stop.set()
        self.listener.stop()
        self.root.destroy()

if __name__=='__main__':
    root=tk.Tk()
    try:
        App(root)
        root.mainloop()
    except Exception:
        error=traceback.format_exc()
        print(error)
        try:messagebox.showerror('Error al iniciar',error)
        except Exception:pass
        raise
