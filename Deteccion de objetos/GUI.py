import cv2
from tkinter import *
from PIL import Image, ImageTk
from ultralytics import YOLO

# Cargar el modelo una sola vez fuera de la función
model = YOLO("best_2.pt")

##Funcion toma una lista y muestra la clase detectada y la clase
def mostrarClsDetectadas(resultados):
    clases_id = resultados[0].boxes.cls.tolist() #devuelve las id de las clases detectadas
    nombresDeClases = resultados[0].names  ##names devulve un diccionario con los nombres de las clases
    conteo = {}  ##diccionario para contar objetos detectados
    for id in clases_id:
        nombre_de_cls = nombresDeClases[id]
        if nombre_de_cls in conteo:  #Si el nombre de la clase esta en el diccionario
            conteo[nombre_de_cls] += 1 #Si ya esta le agrego 1
        else:
            conteo[nombre_de_cls] = 1 #agrego el nombre de cls con el valor 1 ejemplo plastico : 1
    # Crear el texto para mostrar el resultado
    resultado = ""
    for nombre, cantidad in conteo.items():
        resultado += f"{nombre} : {cantidad}\n"

    # Actualizar el widget de texto con el resultado
    informacion_text.delete(1.0, END)
    informacion_text.insert(END, resultado)

def centro_objeto(resultados,anotaciones):
    coordenadas = resultados[0].boxes.xyxy.tolist()
    # este codigo es para mostrar las coorenadas del centro del objeto detectado
    for x in coordenadas:
        cordenadasObjeto = x[:4]  # supuestamente toma los 4 ultimos valores de la lista
        x1, y1, x2, y2 = cordenadasObjeto
        xCentro = (x1 + x2) / 2
        yCentro = (y1 + y2) / 2
        texto = "(" + str(round(xCentro)) + "," + str(round(yCentro)) + ")"
        cv2.putText(anotaciones, texto, (int(xCentro) - 20, int(yCentro)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    3)

def visualizar():
    if cap.isOpened():
        # Leer fotograma
        ret, frame = cap.read()
        if ret:
            # Redimensionar el fotograma directamente con OpenCV
            frame = cv2.resize(frame, (720, 405))
            # Realizar predicción
            resultados = model.predict(frame, imgsz=640, conf=0.55)
            ##Muestro cls detectadas en el GUI
            mostrarClsDetectadas(resultados)
            # Obtener las anotaciones
            anotaciones = resultados[0].plot()
            #centro del objeto
            centro_objeto(resultados, anotaciones)
            # Convertir anotaciones a formato RGB
            frame_show = cv2.cvtColor(anotaciones, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame_show)
            img = ImageTk.PhotoImage(image=im)

            # Mostrar en el GUI
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(33, visualizar)  # Actualizar cada 10 ms
        else:
            lblVideo.image = ""
            cap.release()


## GUI

# Parámetros iniciales
pantalla = Tk()
pantalla.title("CLASIFICADOR AUTOMATICO DE RESIDUOS")
pantalla.geometry("1280x720")

# Fondo
background = PhotoImage(file="./Tkinter/SortingWaste.png")
background_label = Label(image=background, text="fondo")
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Label de video
lblVideo = Label(pantalla)
lblVideo.place(x=350, y=150)

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

# Etiqueta para el título del texto
titulo_label = Label(pantalla, text="Número de reciclables detectados", font=('Arial', 12, 'bold'), bd=0 , bg="#077341", fg="white")
titulo_label.place(x=20, y=170)

# Texto para mostrar informacion:
informacion_text = Text(pantalla, height=5, width=20, bg="#077341", font=('Arial', 12, 'bold') ,bd=0, fg="white")
informacion_text.pack(pady=10)
informacion_text.place(x=20, y=200)


visualizar()
pantalla.mainloop()
