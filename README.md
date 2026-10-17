# Pruebas de seguimiento ocular con MediaPipe y OpenCV

Primera versión para pruebas piloto en Windows con Python 3.11 de 64 bits.
Ejecuta todo en tu computadora; no envía los registros a un servidor.
MediaPipe localiza puntos faciales y de iris. El programa añade una regresión
calibrada para estimar mirada en pantalla. No es equivalente a Tobii ni un
instrumento validado. No calcula fijaciones, sacadas, atención o comprensión.

## Instalación

1. Instala Python 3.11 de 64 bits si todavía no lo tienes. En PowerShell,
   comprueba: `py -3.11 --version`.
2. Extrae el ZIP completo en una carpeta, por ejemplo
   `C:\Users\vaneb\Documents\EyeTracking`.
3. Abre `instalar.bat`. Instala las dependencias en un entorno aislado y
   descarga el modelo oficial de Google. La primera instalación necesita Internet.
4. Cuando termine sin errores, abre `iniciar.bat`.

Alternativa desde PowerShell abierto en esa carpeta:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe descargar_modelo.py
.\.venv\Scripts\python.exe app.py
```

No hace falta activar el entorno ni cambiar la política de PowerShell.
El modelo se descarga de:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

Documentación oficial del detector:
https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python

## Antes de una sesión

- Usa la pantalla principal situada en el origen del escritorio de Windows.
  Para la primera prueba, desconecta pantallas adicionales.
- Mantén la resolución y escala de Windows durante toda la sesión.
- Usa luz uniforme, webcam fija y rostro frontal. Evita reflejos.
  Mantén una distancia cómoda y estable; mover la cabeza puede invalidar la estimación.
- Prepara una cuenta de prueba de Khan Academy. No grabes contraseñas.
- Usa códigos como P001; no escribas nombres del participante.
- Explica los registros y obtén el consentimiento. La casilla del programa
  documenta que el experimentador lo confirmó; no sustituye la carta firmada.
- Para cambiar de participante, cierra y abre la aplicación. Calibra cada persona.

## Uso paso a paso

1. Introduce el código, selecciona cámara 0 y pulsa **Conectar**.
   Si no abre, cierra y prueba con índice 1.
2. Confirma el consentimiento y pulsa **Calibrar**.
3. Mira cada punto verde sin seguirlo con el ratón ni mover la cabeza.
   La captura dura unos 2.6 segundos por punto y omite el primer tramo.
   Si faltan muestras de ojos abiertos se repite el punto. Esc cancela.
4. Pulsa **Validar precisión**. Mira los cinco puntos.
   El resultado presenta el error medio y máximo en píxeles, usando la
   predicción de las características medianas de cada objetivo.
   No mide la dispersión de todas las muestras ni certifica precisión.
5. Compara el error con la separación y tamaño de los elementos que deseas
   distinguir. Por ejemplo, un error de 180 px no permite afirmar que se
   observó un botón de 40 px. Recalibra y vuelve a validar si no es adecuado.
   El programa permite continuar después de validar, sin aprobar automáticamente
   la calidad. El equipo debe decidir si el registro sirve.
6. Selecciona una de las cinco tareas y escribe notas si hace falta.
7. Activa **Grabar pantalla** únicamente cuando esté contemplado en el consentimiento.
8. Prepara el navegador y pulsa **F8** (o el botón de inicio).
   Hay cinco segundos para volver a la página. El panel desaparece antes de capturar.
   F10 cancela esa cuenta regresiva.
9. Realiza la tarea. Pulsa **F9** para terminar desde el navegador.
   Selecciona el resultado observado y pulsa Guardar.
10. Repite por tarea. Al volver al panel puedes cambiar las notas y la tarea.

En algunas laptops las teclas requieren Fn + F8 / Fn + F9.
Los atajos globales funcionan también con el navegador activo.

Para la primera tarea, realiza el acceso ANTES de iniciar una grabación de pantalla.
Registra en notas que el tramo de inicio de sesión no quedó capturado.
Si necesitan registrar ese tramo, deben cambiar el procedimiento de grabación
para excluir las credenciales; esta versión no permite pausar y reanudar una tarea.

## Archivos generados

Dentro de `resultados/P001_fecha_hora/`:

- `sesion.json`: código, resolución, versiones y método.
- `calibracion.json`: características medianas por objetivo y modelo ajustado.
- `validacion.json`: objetivos, predicciones y errores.
- Una carpeta independiente por ejecución de tarea. Repetir una tarea no sobrescribe la anterior.

Cada tarea contiene:

- `mirada.csv`: tiempo monotónico de captura de webcam, tiempo relativo a la tarea,
  estado de detección, coordenadas normalizadas, píxeles y características usadas.
- `inicio.json`: referencia temporal, tarea y configuración.
- `resumen.json`: duración, resultado, notas, validación y estado de grabación.
- `heatmap_pantalla.png`: densidad suavizada de muestras dentro de pantalla,
  cuando hay datos válidos.
- `pantalla.avi` y `pantalla_tiempos.csv`: si se activó la grabación.

Estados: `ok`, `sin_rostro`, `sin_iris`, `parpadeo_ojo_cerrado`,
`ojo_no_valido`, `rostro_no_valido` o `no_finito`.
El filtro de apertura es una heurística, no una detección de parpadeos validada.
Las coordenadas fuera de pantalla se conservan en el CSV y no se suman al mapa.
El porcentaje de muestras en pantalla no es una medida de precisión.

La webcam se procesa en memoria y no se graba. La captura de pantalla no incluye audio.
La grabación sí incluye cualquier contenido visible en la pantalla principal.

## Interpretación del mapa y sincronización

El mapa es de coordenadas de PANTALLA, no de posiciones de una página web.
No se superpone automáticamente a una captura: navegar o hacer scroll cambia
el contenido bajo una misma coordenada. Tampoco identifica áreas de interés.
Para analizar elementos concretos, divide la tarea por pantallas/estados o
consulta el video y los tiempos. Una mayor densidad no demuestra mayor atención.

El video se escribe a 10 fps nominales con los fotogramas disponibles.
En una máquina lenta su duración de reproducción puede diferir de la sesión.
Para sincronizar, usa `pantalla_tiempos.csv`: cada fila indica el tiempo real
del fotograma desde el inicio de la tarea y su posición nominal en el video.
Relaciona ese tiempo real con `tiempo_tarea_s` del CSV de mirada.
Las marcas de cámara y pantalla usan el mismo reloj monotónico del equipo;
no son una sincronización de hardware. No se rellenan huecos con miradas inventadas.

## Si algo falla

- Falta `mediapipe` o `cv2`: vuelve a ejecutar instalar.bat y conserva el error.
- No abre la cámara: permite acceso a cámara para aplicaciones de escritorio en
  Windows y cierra Teams/Zoom u otros programas que la estén usando.
- No reconoce `py -3.11`: instala Python 3.11 de 64 bits.
- No descarga el modelo: revisa Internet y vuelve a ejecutar descargar_modelo.py.
- No sigue bien la mirada: recalibra, ajusta luz, evita reflejos y valida.
- El video ocupa mucho: AVI MJPG puede ser grande. Revisa espacio antes de cada sesión.
- No cierres a la fuerza ni suspendas la laptop durante una tarea; termina con F9.
  Ante cierre inesperado, el CSV se vacía a disco periódicamente, pero el resumen
  y la finalización del video pueden faltar.

## Verificación del código

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_core.py
```

Incluye pruebas de regresión, error en píxeles, exclusión de coordenadas fuera de
pantalla y rechazo de entradas sin iris/ojos cerrados.
La cámara, los atajos, la calibración real y la captura del escritorio requieren
una prueba en la computadora Windows donde se aplicará el experimento.
