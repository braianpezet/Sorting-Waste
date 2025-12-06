import cv2
import os
import threading
import time
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import Counter


class WasteSortingApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1280x720")

        # --- Configuración ---
        self.camera_index = 0  # Ajusta esto si es necesario (0, 1, 2)
        self.model_path = "best.pt"
        self.bg_image_path = "./Tkinter/SortingWaste.png"

        # Variables compartidas entre hilos
        self.current_frame = None  # Aquí guardaremos la imagen lista para mostrar
        self.current_counts = ""  # Aquí guardaremos el texto del conteo
        self.is_running = True  # Bandera para controlar el hilo
        self.lock = threading.Lock()  # Para evitar conflictos de lectura/escritura

        # --- Cargar Modelo ---
        print("Cargando modelo YOLO...")
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            messagebox.showerror("Error", f"No se encontró el modelo: {e}")
            self.window.destroy()
            return

        # --- Interfaz Gráfica (GUI) ---
        self.setup_gui()

        # --- Inicializar Cámara ---
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        if not self.cap.isOpened():
            messagebox.showerror("Error", f"No se pudo abrir la cámara {self.camera_index}")
            self.window.destroy()
            return

        # --- INICIAR HILO DE PROCESAMIENTO ---
        # Este hilo hará el trabajo pesado (Cámara + YOLO)
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        # --- INICIAR BUCLE DE ACTUALIZACIÓN DE GUI ---
        # Este método solo revisa si hay nuevas imágenes para mostrar
        self.update_gui_loop()

    def setup_gui(self):
        # 1. Fondo
        if os.path.exists(self.bg_image_path):
            self.background = PhotoImage(file=self.bg_image_path)
            self.background_label = Label(self.window, image=self.background)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        else:
            self.window.configure(bg="gray")

        # 2. Label de Video
        self.lblVideo = Label(self.window, bg="black")
        self.lblVideo.place(x=350, y=150)

        # 3. Título
        self.titulo_label = Label(self.window, text="Número de reciclables detectados",
                                  font=('Arial', 12, 'bold'), bd=0, bg="#077341", fg="white")
        self.titulo_label.place(x=20, y=170)

        # 4. Texto de Información
        self.informacion_text = Text(self.window, height=5, width=20,
                                     bg="#077341", font=('Arial', 12, 'bold'), bd=0, fg="white")
        self.informacion_text.place(x=20, y=200)

    def procesar_conteo(self, resultados):
        """Genera el string del conteo (se ejecuta en el hilo secundario)"""
        try:
            clases_id = resultados[0].boxes.cls.tolist()
            nombres_dict = resultados[0].names
            nombres_detectados = [nombres_dict[i] for i in clases_id]
            conteo = Counter(nombres_detectados)

            texto_resultado = ""
            for nombre, cantidad in conteo.items():
                texto_resultado += f"{nombre.capitalize()} : {cantidad}\n"

            return texto_resultado
        except:
            return "Error en conteo"

    def dibujar_centroide(self, resultados, imagen_cv):
        """Dibuja centroides en la imagen (se ejecuta en el hilo secundario)"""
        boxes = resultados[0].boxes.xyxy.tolist()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            texto = f"({cx},{cy})"
            cv2.circle(imagen_cv, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(imagen_cv, texto, (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def video_loop(self):
        """
        Lógica del Hilo Secundario:
        Captura -> Predice -> Dibuja -> Guarda en variable compartida.
        """
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Redimensionar para el GUI
                frame = cv2.resize(frame, (720, 405))

                # Inferencia YOLO
                resultados = self.model.predict(frame, imgsz=640, conf=0.55, verbose=False)

                # Procesar datos
                texto_conteo = self.procesar_conteo(resultados)

                # Dibujar
                anotaciones = resultados[0].plot()
                self.dibujar_centroide(resultados, anotaciones)

                # Convertir a formato compatible con Tkinter (PIL)
                frame_rgb = cv2.cvtColor(anotaciones, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(frame_rgb)

                # SECCIÓN CRÍTICA: Actualizar variables compartidas
                # Usamos lock para que el hilo principal no lea mientras escribimos
                with self.lock:
                    self.current_frame = im_pil
                    self.current_counts = texto_conteo
            else:
                time.sleep(0.1)  # Si la cámara falla, esperar un poco antes de reintentar

    def update_gui_loop(self):
        """
        Lógica del Hilo Principal (Tkinter):
        Lee las variables compartidas y actualiza la pantalla.
        """
        # 1. Obtener datos del hilo secundario de forma segura
        img_para_mostrar = None
        texto_para_mostrar = ""

        with self.lock:
            if self.current_frame is not None:
                img_para_mostrar = self.current_frame
                # Importante: Una vez tomada, podemos ponerla en None para no repintar lo mismo
                # (opcional, depende de la fluidez deseada)
            texto_para_mostrar = self.current_counts

        # 2. Actualizar Widgets (Solo si hay imagen nueva)
        if img_para_mostrar:
            img_tk = ImageTk.PhotoImage(image=img_para_mostrar)
            self.lblVideo.configure(image=img_tk)
            self.lblVideo.image = img_tk  # Referencia para evitar Garbage Collection

        # Actualizar texto (siempre o verificar si cambió)
        current_text_widget = self.informacion_text.get(1.0, END).strip()
        if texto_para_mostrar.strip() != current_text_widget:
            self.informacion_text.delete(1.0, END)
            self.informacion_text.insert(END, texto_para_mostrar)

        # 3. Programar la próxima actualización (30ms = ~33 FPS para la GUI)
        self.window.after(30, self.update_gui_loop)

    def on_closing(self):
        """Limpieza al cerrar"""
        print("Deteniendo hilos y liberando recursos...")
        self.is_running = False  # Esto detendrá el bucle while del hilo secundario

        # Esperar un momento a que el hilo secundario termine
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.cap.isOpened():
            self.cap.release()

        self.window.destroy()


# --- Ejecución ---
if __name__ == "__main__":
    root = Tk()
    app = WasteSortingApp(root, "CLASIFICADOR AUTOMÁTICO (Multihilo)")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()