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
  El programa vigila la posición de la cabeza (ver «Control de la posición de la cabeza»).
- Prepara una cuenta de prueba de Khan Academy. No grabes contraseñas.
- Usa códigos como P001; no escribas nombres del participante.
- Explica los registros y obtén el consentimiento. La casilla del programa
  documenta que el experimentador lo confirmó; no sustituye la carta firmada.
- Para cambiar de participante, cierra y abre la aplicación. Calibra cada persona.

## Uso paso a paso

1. Introduce el código, selecciona cámara 0 y pulsa **Conectar**.
   Si no abre, cierra y prueba con índice 1.
2. Confirma el consentimiento. Siéntate cómodo, quédate quieto un segundo y pulsa
   **Calibrar**. Esa posición de la cabeza queda como referencia de toda la sesión.
3. Mira cada punto verde sin seguirlo con el ratón ni mover la cabeza.
   La captura dura unos 2.6 segundos por punto y omite el primer tramo.
   Si faltan muestras de ojos abiertos se repite el punto. Esc cancela.
   Si la cabeza sale de la referencia aparece un aviso y esas muestras no se usan.
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
8. Comprueba que debajo de la cámara diga «✓ Cabeza en la posición de calibración»;
   si no, no se puede iniciar. Prepara el navegador y pulsa **F8** (o el botón de inicio).
   Hay cinco segundos para volver a la página. El panel desaparece antes de capturar.
   F10 cancela esa cuenta regresiva.
9. Realiza la tarea. Si la cabeza se mueve aparece un aviso flotante arriba de la
   pantalla con la corrección (por ejemplo «Acércate un poco a la pantalla»).
   Pulsa **F9** para terminar desde el navegador. Selecciona el resultado y pulsa Guardar.
   Si se grabó la pantalla, se genera el video con mirada en segundo plano; espera a
   que termine antes de iniciar otra tarea.
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
- `pantalla_mirada.mp4` (o `.avi`): la grabación de pantalla a velocidad real con el
  punto de mirada, su estela del último segundo, un mapa de calor reciente que se
  desvanece (vida media de 3 s) y una franja naranja/roja cuando la cabeza estaba
  fuera de posición. Se genera a partir de `pantalla.avi` y `mirada.csv`.
- `heatmap_sobre_pantalla.png`: densidad acumulada de toda la tarea sobre un
  fotograma de la mitad de la tarea. Si hubo scroll, el fondo no corresponde a todo el calor.
- `video_mirada.json`: parámetros y conteos del video generado.

Para generar o regenerar los videos después (por ejemplo, si cerraste la aplicación
antes de que terminara), abre `generar_videos.bat`. Procesa todas las tareas con
`pantalla.avi` que aún no tengan video. `generar_videos.bat --forzar` las rehace todas.
También funciona con sesiones anteriores; en ellas no habrá franja de cabeza porque
su CSV no tiene esas columnas.

Columnas añadidas al final de `mirada.csv` (las anteriores no cambian):
`cabeza_pitch_deg`, `cabeza_yaw_deg`, `cabeza_roll_deg` (giros), `cabeza_x_cm`,
`cabeza_y_cm`, `cabeza_dist_cm` (posición aproximada respecto a la cámara),
`cabeza_nivel` (0 bien, 1 aviso, 2 crítico) y `cabeza_motivo`
(`yaw`, `pitch`, `roll`, `x`, `y`, `dist` o `sin_rostro`).

Estados: `ok`, `sin_rostro`, `sin_iris`, `parpadeo_ojo_cerrado`,
`ojo_no_valido`, `rostro_no_valido` o `no_finito`.
El filtro de apertura es una heurística, no una detección de parpadeos validada.
Las coordenadas fuera de pantalla se conservan en el CSV y no se suman al mapa.
El porcentaje de muestras en pantalla no es una medida de precisión.

La webcam se procesa en memoria y no se graba. La captura de pantalla no incluye audio.
La grabación sí incluye cualquier contenido visible en la pantalla principal.

## Control de la posición de la cabeza

El modelo de mirada se ajusta con la cabeza en una sola posición. Si después la
cabeza se gira, se desplaza o se acerca, las variables de rostro quedan fuera del
rango de la calibración y el error crece (en las sesiones piloto, la distancia a la
pantalla y la inclinación cambiaron mucho a lo largo de la tarea).

MediaPipe entrega la orientación y posición de la cabeza; el programa las compara con
la referencia tomada al calibrar, usando la mediana de los últimos 5 cuadros:

| Medida | Aviso | Crítico |
|---|---|---|
| Giro izquierda/derecha, arriba/abajo | 5° | 10° |
| Inclinación lateral | 6° | 12° |
| Desplazamiento lateral o vertical | 2.5 cm | 5 cm |
| Distancia a la pantalla | 7 % | 15 % |

Si no se detecta el rostro durante ~0.5 s también es crítico. Los límites están en
`HEAD_LIMITS` dentro de `core.py`; son un punto de partida para ajustar con sus
validaciones, no valores validados. Si la webcam entrega la imagen en espejo y las
indicaciones de izquierda/derecha salen al revés, cambien `CAMARA_ESPEJO = True`.

- Calibración y validación: las muestras con la cabeza fuera de posición se descartan.
- Antes de iniciar una tarea: no se permite iniciar fuera de posición.
- Durante la tarea: aviso flotante arriba de la pantalla. En Windows no roba el foco,
  deja pasar los clics y, en Windows 10 2004 o posterior, no aparece en `pantalla.avi`
  (`aviso_excluido_de_grabacion` en `inicio.json` indica si se logró).
  Opcionalmente, un sonido al pasar a nivel crítico.
- `resumen.json` incluye la sección `cabeza`: muestras por nivel, porcentaje fuera de
  posición, número de alertas, tiempo en alerta y desviación máxima.
- Las muestras en nivel crítico se conservan en el CSV pero no se suman a los mapas
  de calor (`EXCLUIR_NIVEL_CRITICO` en `render.py`).

La pose se calcula en memoria a partir de la imagen de la webcam; solo se guardan los
números. No se guarda ninguna imagen de la cámara.

## Interpretación del mapa y sincronización

El mapa es de coordenadas de PANTALLA, no de posiciones de una página web.
`heatmap_pantalla.png` no tiene fondo. `heatmap_sobre_pantalla.png` lo pone sobre un
solo fotograma, pero navegar o hacer scroll cambia el contenido bajo una misma
coordenada; por eso el video `pantalla_mirada.mp4` usa un calor reciente que se
desvanece en lugar de acumular toda la tarea. Tampoco identifica áreas de interés.
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
pantalla, rechazo de entradas sin iris/ojos cerrados, cálculo de la pose y niveles de
alerta de la cabeza, y generación del video con mirada a partir de datos sintéticos.
La cámara, los atajos, la calibración real y la captura del escritorio requieren
una prueba en la computadora Windows donde se aplicará el experimento.
