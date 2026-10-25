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
from collections import deque
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
from core import (GazeModel, extract_features, validation_error, heat_grid,
                  FEATURE_NAMES, HEAD_NAMES, HEAD_LIMITS, HeadMonitor, head_pose)
import render

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
                min_face_presence_confidence=.6,min_tracking_confidence=.6,
                output_facial_transformation_matrixes=True)
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
                    head=None
                    if result.facial_transformation_matrixes:
                        # Pose de la cabeza (giros en grados, traslación en cm). Solo números.
                        head=head_pose(result.facial_transformation_matrixes[0])
                    if result.face_landmarks:
                        features,status=extract_features(result.face_landmarks[0])
                        for i in [468,473]:
                            if len(result.face_landmarks[0])>i:
                                v=result.face_landmarks[0][i]
                                cv2.circle(frame,(int(v.x*frame.shape[1]),int(v.y*frame.shape[0])),3,(0,255,0),-1)
                    self.frame=frame
                    self.items.put((capture,features,status,head))
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

class HeadAlertOverlay:
    """Aviso flotante durante la tarea cuando la cabeza sale de la posición de calibración.

    En Windows: siempre encima, no roba el foco, deja pasar los clics al navegador y
    (Windows 10 2004 o posterior) se excluye de la captura, así no aparece en pantalla.avi.
    """
    COLORS={1:'#b45309',2:'#b91c1c'}
    def __init__(self,root,monitor):
        self.monitor=monitor;self.current=(0,'');self.excluded_from_capture=False
        self.win=tk.Toplevel(root);self.win.overrideredirect(True)
        self.win.attributes('-topmost',True)
        self.label=tk.Label(self.win,font=('Segoe UI',16,'bold'),fg='white',bg=self.COLORS[1],padx=22,pady=10)
        self.label.pack()
        if os.name=='nt':
            self.win.attributes('-alpha',0.0)
            self.win.geometry(f"+{monitor['left']+monitor['width']//3}+{monitor['top']+20}")
            self.win.update_idletasks()
            self.apply_styles()
        else:
            self.win.withdraw()
    def hwnd(self):
        return ctypes.windll.user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
    def apply_styles(self):
        try:
            user32=ctypes.windll.user32;hwnd=self.hwnd()
            # LAYERED | TRANSPARENT (clics pasan) | TOOLWINDOW (sin barra de tareas) | NOACTIVATE
            style=user32.GetWindowLongW(hwnd,-20)|0x80000|0x20|0x80|0x08000000
            user32.SetWindowLongW(hwnd,-20,style)
            # WDA_EXCLUDEFROMCAPTURE: no aparece en la grabación de pantalla.
            self.excluded_from_capture=bool(user32.SetWindowDisplayAffinity(hwnd,0x11))
        except Exception:
            traceback.print_exc()
    def show(self,level,message):
        if (level,message)==self.current:return
        self.current=(level,message)
        if level==0:
            if os.name=='nt':self.win.attributes('-alpha',0.0)
            else:self.win.withdraw()
            return
        color=self.COLORS.get(level,self.COLORS[2])
        prefix='⚠  Regresa a tu posición' if level==1 else '⛔  Te moviste demasiado'
        self.label.configure(text=f'{prefix}: {message}',bg=color);self.win.configure(bg=color)
        self.win.update_idletasks()
        x=self.monitor['left']+(self.monitor['width']-self.label.winfo_reqwidth())//2
        y=self.monitor['top']+int(self.monitor['height']*.03)
        self.win.geometry(f'+{x}+{y}')
        if os.name=='nt':
            self.win.attributes('-alpha',.93);self.apply_styles()
            try:  # reafirmar 'encima de todo' sin activar la ventana
                ctypes.windll.user32.SetWindowPos(self.hwnd(),-1,0,0,0,0,0x1|0x2|0x10)
            except Exception:pass
        else:
            self.win.deiconify()
    def hide(self):
        self.show(0,'')

class App:
    def __init__(self,root):
        self.root=root
        root.title('Eye Tracking | Pruebas piloto')
        root.geometry('780x820')
        self.camera=None; self.model=None; self.validation=None; self.session=None
        self.active=None; self.cal=None; self.latest=None; self.recorder=None
        self.points=[]; self.keys=queue.Queue(); self.pending=False
        self.recent_heads=deque(maxlen=60); self.head_ref=None; self.head_monitor=None
        self.render_thread=None; self.render_progress=0.; self.render_result=None; self.last_beep=0.
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
        self.make_video=tk.BooleanVar(value=True)
        self.beep=tk.BooleanVar(value=False)
        self.head_state=tk.StringVar(value='')
        self.render_state=tk.StringVar(value='')
        self.build()
        self.overlay=HeadAlertOverlay(root,self.monitor)
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
        self.preview=ttk.Label(f); self.preview.pack(pady=(10,2))
        self.preview.configure(text='La vista de cámara aparece aquí (no se guarda).')
        self.head_label=ttk.Label(f,textvariable=self.head_state,font=('Segoe UI',10,'bold'),wraplength=700)
        self.head_label.pack(anchor='w')
        row=ttk.Frame(f);row.pack(fill='x',pady=5)
        ttk.Button(row,text='1  Calibrar',command=lambda:self.calibrate(False)).pack(side='left',padx=3)
        ttk.Button(row,text='2  Validar precisión',command=lambda:self.calibrate(True)).pack(side='left',padx=3)
        ttk.Label(f,textvariable=self.state,wraplength=700).pack(anchor='w',pady=8)
        ttk.Checkbutton(f,text='El participante otorgó consentimiento para los registros seleccionados',
                        variable=self.consent).pack(anchor='w')
        ttk.Checkbutton(f,text='Grabar pantalla (sin audio ni imagen de webcam)',
                        variable=self.record_screen).pack(anchor='w')
        ttk.Checkbutton(f,text='Al terminar, generar video de pantalla con mirada y mapa de calor (requiere grabar pantalla)',
                        variable=self.make_video).pack(anchor='w')
        ttk.Checkbutton(f,text='Sonido de alerta cuando la cabeza se mueve demasiado',
                        variable=self.beep).pack(anchor='w')
        ttk.Label(f,text='Tarea (prepara la página antes de iniciar):').pack(anchor='w',pady=(12,3))
        ttk.Combobox(f,textvariable=self.task,values=TASKS,state='readonly',width=58).pack(anchor='w')
        ttk.Label(f,text='Notas de la tarea:').pack(anchor='w',pady=(8,2))
        ttk.Entry(f,textvariable=self.notes,width=85).pack(fill='x')
        row=ttk.Frame(f);row.pack(fill='x',pady=12)
        ttk.Button(row,text='3  Iniciar tarea  F8',command=self.start_task).pack(side='left',padx=3)
        ttk.Button(row,text='Detener tarea  F9',command=self.stop_task).pack(side='left',padx=3)
        ttk.Label(f,text='F8 inicia tras 5 s. F9 detiene y muestra el panel.\nF10 cancela la cuenta regresiva. No grabes contraseñas; inicia después del acceso.',
                  wraplength=690).pack(anchor='w')
        ttk.Label(f,textvariable=self.render_state,wraplength=700).pack(anchor='w',pady=(6,0))

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
        self.head_ref=None;self.head_monitor=None;self.recent_heads.clear()
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
        if validate and not self.head_monitor.ok:
            messagebox.showwarning('Posición de la cabeza','Regresa a la posición que tenías al calibrar:\n'+self.head_monitor.message)
            return
        if not validate:
            now=time.perf_counter()
            poses=np.array([h for s,h in self.recent_heads if now-s<1.])
            if len(poses)<8:
                messagebox.showwarning('Cabeza','No se detecta la posición de la cabeza. Colócate frente a la cámara.')
                return
            spread=np.ptp(poses,axis=0)
            if spread[:3].max()>6 or spread[3:5].max()>3:
                messagebox.showwarning('Cabeza','Siéntate cómodo y quédate quieto un segundo antes de calibrar.\n'
                                       'Esa posición será la referencia para toda la sesión.')
                return
        try:self.ensure_session()
        except ValueError as exc:
            messagebox.showerror('Participante',str(exc));return
        if not validate:
            self.model=None; self.validation=None
            # La posición actual se vuelve la referencia de toda la sesión.
            self.head_ref=np.median(poses,axis=0)
            self.head_monitor=HeadMonitor(self.head_ref)
        pts=[(.1,.1),(.5,.1),(.9,.1),(.1,.5),(.5,.5),(.9,.5),(.1,.9),(.5,.9),(.9,.9)]
        if validate:pts=[(.25,.25),(.75,.25),(.5,.5),(.25,.75),(.75,.75)]
        win=tk.Toplevel(self.root); win.attributes('-fullscreen',True);win.attributes('-topmost',True)
        canvas=tk.Canvas(win,bg='#142337',highlightthickness=0);canvas.pack(fill='both',expand=True)
        self.cal={'validate':validate,'targets':pts,'i':0,'data':[],'samples':[],
                  'since':time.perf_counter(),'win':win,'canvas':canvas,'rejected':0,'head_msg':None}
        win.bind('<Escape>',lambda e:self.cancel_cal())
        win.protocol('WM_DELETE_WINDOW',self.cancel_cal)
        self.paint_target()
    def paint_target(self):
        c=self.cal;cv=c['canvas'];cv.delete('all')
        x,y=c['targets'][c['i']];x*=self.w;y*=self.h
        cv.create_oval(x-14,y-14,x+14,y+14,fill='#69e3bb',outline='white',width=2)
        c['head_msg']=None
        cv.create_text(self.w/2,self.h*.03,text=f"Mira el punto sin mover la cabeza • {c['i']+1}/{len(c['targets'])} • Esc para cancelar",fill='white',font=('Segoe UI',16))
    def paint_cal_head(self):
        """Aviso en la pantalla de calibración; esas muestras no se usan."""
        c=self.cal;m=self.head_monitor
        msg=m.message if m and not m.ok else ''
        if msg==c['head_msg']:return
        c['head_msg']=msg;c['canvas'].delete('head')
        if msg:
            c['canvas'].create_text(self.w/2,self.h*.5+70,text='⚠ Muestras en pausa: '+msg,
                                    fill='#ffb020' if m.level==1 else '#ff5c5c',font=('Segoe UI',20,'bold'),tags='head')
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
                          'sample_count':len(c['samples']),'rechazadas_por_cabeza':c['rejected']})
        c['i']+=1;c['samples']=[];c['since']=now;c['rejected']=0
        if c['i']<len(c['targets']):
            self.paint_target();return
        c['win'].destroy();self.cal=None
        if not c['validate']:
            self.model=GazeModel().fit([v['features'] for v in c['data']],[v['target'] for v in c['data']])
            dump(self.session/'calibracion.json',{'datos':c['data'],'modelo':self.model.export(),
                 'cabeza_referencia':dict(zip(HEAD_NAMES,self.head_ref.tolist())),
                 'limites_cabeza':HEAD_LIMITS})
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
            if not self.head_monitor.ok:
                raise ValueError('Regresa a la posición que tenías al calibrar:\n'+self.head_monitor.message)
            if self.render_thread and self.render_thread.is_alive():
                raise ValueError('Espera a que termine de generarse el video de la tarea anterior.')
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
                     'notes':self.pending_notes,'count':0,'valid':0,
                     'head_levels':[0,0,0],'episodes':0,'alert_s':0.,'last_stamp':None,'last_level':0,
                     'max_dev':{},'excluded':0}
        self.points=[]
        self.datafile=(folder/'mirada.csv').open('w',newline='',encoding='utf-8')
        self.csv=csv.writer(self.datafile)
        self.csv.writerow(['captura_monotonic_s','tiempo_tarea_s','estado','x_normalizado','y_normalizado',
                           'x_pantalla_px','y_pantalla_px','en_pantalla']+list(FEATURE_NAMES)+
                          list(HEAD_NAMES)+['cabeza_nivel','cabeza_motivo'])
        self.recorder=None
        if self.pending_record:
            self.recorder=ScreenRecorder(folder,self.monitor,start);self.recorder.start()
        dump(folder/'inicio.json',{'inicio_utc':self.active['utc'],'inicio_monotonic':start,
             'tarea':self.pending_task,'notas':self.pending_notes,'monitor':self.monitor,
             'grabacion_solicitada':self.pending_record,
             'calibracion_modelo':self.model.export(),'validacion':self.validation,
             'cabeza_referencia':dict(zip(HEAD_NAMES,self.head_ref.tolist())),'limites_cabeza':HEAD_LIMITS,
             'aviso_excluido_de_grabacion':self.overlay.excluded_from_capture})
        self.state.set('Tarea en curso. F9 para detener.')
    def stop_task(self):
        if self.pending:
            self.pending=False;self.root.deiconify();self.state.set('Inicio cancelado.');return
        if not self.active:return
        end=time.perf_counter()
        a=self.active;self.active=None
        self.overlay.hide()
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
             'heatmap':'Densidad de muestras, no fijaciones. Coordenadas de pantalla, no de página.',
             'cabeza':self.head_summary(a)})
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
            out=self.head_summary(a)['porcentaje_fuera_de_posicion']
            self.state.set(f"Tarea guardada en {a['folder'].name}. Muestras en pantalla: {a['valid']}/{a['count']}. "
                           f"Cabeza fuera de posición: {out:.0f}% de las muestras ({a['episodes']} alertas).")
            if r and r.error:messagebox.showwarning('Grabación',r.error+'\nEl CSV de mirada se guardó.')
        except Exception as exc:
            messagebox.showerror('Mapa de calor',str(exc)+'\nEl CSV de mirada se guardó.')
        if r and self.make_video.get():
            if r.error or r.is_alive() or r.frames==0:
                self.render_state.set('No se generó el video con mirada (grabación incompleta). '
                                      'Puedes intentarlo después con generar_videos.bat.')
            else:
                self.start_render(a['folder'])

    def head_summary(self,a):
        n=max(1,a['count']);lv=a['head_levels']
        return {'referencia':dict(zip(HEAD_NAMES,self.head_ref.tolist())),'limites':HEAD_LIMITS,
                'muestras_por_nivel':{'0_bien':lv[0],'1_aviso':lv[1],'2_critico':lv[2]},
                'porcentaje_fuera_de_posicion':100*(lv[1]+lv[2])/n,
                'episodios_de_alerta':a['episodes'],'tiempo_en_alerta_s':a['alert_s'],
                'desviacion_maxima':a['max_dev'],
                'muestras_excluidas_del_heatmap':a['excluded'],
                'excluir_nivel_critico_del_heatmap':render.EXCLUIR_NIVEL_CRITICO,
                'aviso_excluido_de_grabacion':self.overlay.excluded_from_capture}

    def track_head(self,stamp,monitor):
        """Estadísticas de movimiento y sonido opcional al pasar a nivel crítico."""
        a=self.active;level=monitor.level
        a['head_levels'][level]+=1
        if a['last_stamp'] is not None and a['last_level']>=1:a['alert_s']+=stamp-a['last_stamp']
        if level>=1 and a['last_level']==0:a['episodes']+=1
        if level==2 and a['last_level']<2 and self.beep.get() and os.name=='nt' and stamp-self.last_beep>4:
            try:
                import winsound;winsound.MessageBeep(winsound.MB_ICONEXCLAMATION);self.last_beep=stamp
            except Exception:pass
        for k,v in (monitor.deviation or {}).items():
            a['max_dev'][k]=max(a['max_dev'].get(k,0.),abs(v))
        a['last_stamp']=stamp;a['last_level']=level

    def start_render(self,folder):
        self.render_progress=0.;self.render_result=None
        def work():
            try:
                self.render_result=render.render_task(folder,progress=lambda p:setattr(self,'render_progress',p))
            except Exception as exc:
                traceback.print_exc();self.render_result={'error':str(exc)}
        self.render_thread=threading.Thread(target=work,daemon=True);self.render_thread.start()

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
                stamp,features,status,head=self.camera.items.get()
                self.latest=(stamp,features,status)
                if head is not None:self.recent_heads.append((stamp,head))
                monitor=self.head_monitor.update(head) if self.head_monitor else None
                head_ok=monitor is None or monitor.ok
                if self.cal and features is not None:
                    # Ignorar el desplazamiento inicial de los ojos hacia el nuevo objetivo.
                    if stamp-self.cal['since']>.8:
                        # Con la cabeza fuera de la referencia la muestra no sirve para el modelo.
                        if head_ok:self.cal['samples'].append(features)
                        else:self.cal['rejected']+=1
                if self.active and stamp>=self.active['start']:
                    pred=self.model.predict(features) if features is not None else None
                    inside=pred is not None and bool(np.isfinite(pred).all()) and 0<=pred[0]<1 and 0<=pred[1]<1
                    self.active['count']+=1
                    level=monitor.level if monitor else 0
                    if inside:
                        self.active['valid']+=1
                        if level<2 or not render.EXCLUIR_NIVEL_CRITICO:
                            self.points.append((pred[0]*self.w,pred[1]*self.h))
                        else:self.active['excluded']+=1
                    if monitor:self.track_head(stamp,monitor)
                    coords=list(pred)+[pred[0]*self.w,pred[1]*self.h] if pred is not None else ['']*4
                    self.csv.writerow([stamp,stamp-self.active['start'],status]+coords+[int(inside)]+
                                      (features.tolist() if features is not None else ['']*8)+
                                      (head.tolist() if head is not None else ['']*6)+
                                      [level,monitor.key if monitor else ''])
            if self.active and self.head_monitor:
                self.overlay.show(self.head_monitor.level,self.head_monitor.message)
            if self.cal and self.head_monitor:self.paint_cal_head()
            if self.active:self.datafile.flush()
            if self.camera.error:
                if self.active:self.stop_task()
                self.state.set(self.camera.error)
            elif self.camera.frame is not None and not self.active and not self.cal:
                # Solo la vista previa se muestra en espejo (como selfie); los datos no cambian.
                img=cv2.cvtColor(cv2.flip(self.camera.frame,1),cv2.COLOR_BGR2RGB)
                img=Image.fromarray(img).resize((320,240))
                self.photo=ImageTk.PhotoImage(img)
                self.preview.configure(image=self.photo,text='')
                self.update_head_label()
                if self.model is None and not self.pending:
                    self.state.set('Cámara conectada. '+(self.latest[2] if self.latest else 'Esperando rostro'))
        if self.cal:self.advance_cal(now)
        if self.render_thread:
            if self.render_thread.is_alive():
                self.render_state.set(f'Generando video con mirada y mapa de calor: {self.render_progress:.0%} '
                                      '(puedes calibrar, pero no iniciar otra tarea).')
            else:
                res=self.render_result or {}
                self.render_state.set('Error al generar el video: '+res['error'] if 'error' in res
                                      else f"Video con mirada listo: {res.get('archivo','')} y heatmap_sobre_pantalla.png")
                self.render_thread=None
        self.root.after(40,self.tick)
    def update_head_label(self):
        m=self.head_monitor
        if m is None:
            recent=self.recent_heads and time.perf_counter()-self.recent_heads[-1][0]<.7
            self.head_state.set('Cabeza detectada. Siéntate cómodo y estable: al calibrar esta posición será la referencia.'
                                if recent else 'Cabeza no detectada.')
            self.head_label.configure(foreground='')
        elif m.ok:
            self.head_state.set('✓ Cabeza en la posición de calibración.')
            self.head_label.configure(foreground='#15803d')
        else:
            self.head_state.set(('⚠ ' if m.level==1 else '⛔ ')+'Fuera de la posición de calibración: '+m.message)
            self.head_label.configure(foreground='#b45309' if m.level==1 else '#b91c1c')
    def close(self):
        if self.render_thread and self.render_thread.is_alive():
            if not messagebox.askyesno('Generando video','Todavía se está generando el video con mirada. '
                                       'Si cierras, quedará incompleto (puedes regenerarlo con generar_videos.bat).\n¿Cerrar de todos modos?'):
                return
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
