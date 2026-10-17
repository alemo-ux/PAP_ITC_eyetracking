from pathlib import Path
import urllib.request
import zipfile

URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
target = Path(__file__).parent/'models'/'face_landmarker.task'

def download():
    target.parent.mkdir(exist_ok=True)
    if target.exists() and zipfile.is_zipfile(target):
        print('Modelo disponible:', target)
        return
    temporary=target.with_suffix('.download')
    try:
        print('Descargando el modelo oficial de Google...')
        with urllib.request.urlopen(URL,timeout=90) as response, temporary.open('wb') as out:
            while chunk := response.read(1024*1024):
                out.write(chunk)
        if not zipfile.is_zipfile(temporary):
            raise ValueError('El archivo descargado no es un paquete de modelo válido.')
        temporary.replace(target)
        print('Modelo listo:',target)
    finally:
        temporary.unlink(missing_ok=True)

if __name__=='__main__':
    try:
        download()
    except Exception as exc:
        print('No se pudo descargar:',exc)
        print('Revisa tu conexión y ejecuta de nuevo python descargar_modelo.py')
        raise SystemExit(1)
