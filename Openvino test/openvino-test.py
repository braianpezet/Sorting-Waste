# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("best.pt")
##vinomodel = model.export(format = "openvino")
vinomodel = YOLO('best_openvino_model')

# Realizar VideoCaptura
cap = cv2.VideoCapture(0) ### 0,1,2 significa el dispositivo de video provar con distintos nuemeros si no funciona

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS)) #esto es para imprimir los fps de la camara
    print(fps)
    #este codigo es para cambiar la resolucion de la ventana
    window_width = 512
    window_height = 384
    cv2.namedWindow("DETECCION Y SEGMENTACION", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DETECCION Y SEGMENTACION", window_width, window_height)

    #Leemos resultados
    resultados = model.predict(frame, imgsz=640) ##intercambiar model por vinomodel y viceversa


    # Mostramos resultados
    anotaciones = resultados[0].plot()
    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION",anotaciones)
    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()