# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("best.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(1)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS)) #esto es para imprimir los fps de la camara
    print(fps)
    #este codigo es para cambiar la resolucion de la ventana la achique porque mi computadora no tiene placa de video
    window_width = 512
    window_height = 384
    cv2.namedWindow("DETECCION Y SEGMENTACION", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DETECCION Y SEGMENTACION", window_width, window_height)

    #Leemos resultados
    resultados = model.predict(frame, imgsz=640)


    # Mostramos resultados
    anotaciones = resultados[0].plot()
    coordenadas = resultados[0].boxes.xyxy.tolist()

    #este codigo es para mostrar las coorenadas del centro del objeto detectado
    for x in coordenadas:
        cordenadasObjeto = x[:4] #supuestamente toma los 4 ultimos valores de la lista
        x1 , y1 , x2 , y2 = cordenadasObjeto
        xCentro = (x1 + x2) / 2
        yCentro = (y1 + y2) / 2
        texto = "(" + str(round(xCentro)) + "," + str(round(yCentro)) + ")"
        cv2.putText(anotaciones, texto, (int(xCentro) -20 ,int(yCentro)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)

    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION",anotaciones)
    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()