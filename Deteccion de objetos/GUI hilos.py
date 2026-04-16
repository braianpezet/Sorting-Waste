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
        self.prev_time = 0
        self.fps = 0
        self.last_log_time = time.time()

        # --- Configuración ---
        self.camera_index = 0  # Ajusta esto si es necesario (0, 1, 2)
        self.model_path = "modelo-nano_openvino_model/"  # Ruta al modelo YOLO entrenado
        self.bg_image_path = "./Tkinter/SortingWaste.png"

        # Variables compartidas entre hilos
        self.current_frame = None  # Aquí guardaremos la imagen lista para mostrar
        self.current_counts = Counter()  # Aquí guardaremos el conteo como Counter
        self.current_fps = 0  # Aquí guardaremos el FPS
        self.last_conteo = Counter()  # Para evitar parpadeo innecesario
        self.is_paused = False  # Estado de pausa del video
        self.is_running = True  # Bandera para controlar el hilo
        self.lock = threading.Lock()  # Para evitar conflictos de lectura/escritura
        self.conf_threshold = 0.55  # Umbral de confianza inicial
        self.brightness = 0  # Ajuste de brillo (-100 a +100)
        self.contrast = 1.0  # Ajuste de contraste (0.5 a 1.5)

        # --- Cargar Modelo ---
        print("Cargando modelo YOLO...")
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            messagebox.showerror("Error", f"No se encontró el modelo: {e}")
            self.window.destroy()
            return

        # --- Nombres de clases y selección ---
        self.class_names = self.load_class_names()
        self.class_vars = []
        self.selected_class_ids = list(range(len(self.class_names)))

        # --- Cargar imágenes ---
        self.icon_images = self.load_icon_images()
        self.button_images = self.load_button_images()
        self.recycling_image = self.load_recycling_image()

        # --- Interfaz Gráfica (GUI) ---
        self.setup_gui()

        # --- Inicializar Cámara ---
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.cap.set(3,640)
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

        # Texto a la derecha del video
        self.title_right = Label(self.window, text="Desarrollado por\nBraian Pezet",
                                 font=('Arial', 14, 'bold'), bg="#077341", fg="white", justify='center')
        self.title_right.place(x=1090, y=250)

        if self.recycling_image:
            self.recycling_label = Label(self.window, image=self.recycling_image, bg="#077341")
            self.recycling_label.place(x=1128, y=300)

        # 3. Título
        self.titulo_label = Label(self.window, text="Número de reciclables detectados",
                                  font=('Arial', 12, 'bold'), bd=0, bg="#077341", fg="white")
        self.titulo_label.place(x=20, y=170)

        # 4. Frame de Información con iconos
        self.info_frame = Frame(self.window, bg="#077341")
        self.info_frame.place(x=20, y=200)
        # 5. Slider de Confianza
        self.slider_label = Label(self.window, text="Umbral de Confianza", 
                                  font=('Arial', 10), bg="#077341", fg="white")
        self.slider_label.place(x=20, y=450)
        
        self.conf_slider = Scale(self.window, from_=0, to=100, orient=HORIZONTAL, 
                                 bg="#077341", fg="white", highlightthickness=0)
        self.conf_slider.set(55) # Valor inicial
        self.conf_slider.place(x=20, y=470, width=150)
        self.conf_slider.config(command=self.update_conf)

        # 6. Slider de brillo
        self.brightness_label = Label(self.window, text="Ajuste de brillo", 
                                      font=('Arial', 10), bg="#077341", fg="white")
        self.brightness_label.place(x=20, y=520)

        self.brightness_slider = Scale(self.window, from_=-100, to=100, orient=HORIZONTAL, 
                                       bg="#077341", fg="white", highlightthickness=0)
        self.brightness_slider.set(0)
        self.brightness_slider.place(x=20, y=540, width=150)
        self.brightness_slider.config(command=self.update_brightness)

        # 7. Slider de contraste
        self.contrast_label = Label(self.window, text="Ajuste de contraste", 
                                    font=('Arial', 10), bg="#72be5c", fg="white")
        self.contrast_label.place(x=20, y=615)

        self.contrast_slider = Scale(self.window, from_=50, to=150, orient=HORIZONTAL, 
                                     bg="#077341", fg="white", highlightthickness=0)
        self.contrast_slider.set(100)
        self.contrast_slider.place(x=20, y=640, width=150)
        self.contrast_slider.config(command=self.update_contrast)

        # 8. Selector de clases
        self.class_selector_label = Label(self.window, text="Clases a detectar", 
                                          font=('Arial', 10, 'bold'), bg="#077341", fg="white")
        self.class_selector_label.place(x=350, y=580)

        self.class_frame = Frame(self.window, bg="#077341")
        self.class_frame.place(x=350, y=605)

        for i, clase in enumerate(self.class_names):
            var = BooleanVar(value=True)
            cb = Checkbutton(self.class_frame, text=clase.capitalize(), variable=var,
                             command=self.update_selected_classes, bg="#077341", fg="white",
                             selectcolor="#077341", activebackground="#077341", activeforeground="white")
            cb.grid(row=0, column=i, padx=6, pady=2, sticky="w")
            self.class_vars.append(var)

        self.class_buttons_frame = Frame(self.window, bg="#077341")
        self.class_buttons_frame.place(x=350, y=645)

        self.select_all_button = Button(self.class_buttons_frame, text="Seleccionar todo",
                                        command=self.select_all_classes, bg="#00a651", fg="white",
                                        font=('Arial', 9, 'bold'))
        self.select_all_button.grid(row=0, column=0, padx=2, pady=2)

        self.deselect_all_button = Button(self.class_buttons_frame, text="Deseleccionar todo",
                                          command=self.deselect_all_classes, bg="#d32f2f", fg="white",
                                          font=('Arial', 9, 'bold'))
        self.deselect_all_button.grid(row=0, column=1, padx=2, pady=2)

        # 10. Botones de control
        self.control_frame = Frame(self.window, bg="#077341")
        self.control_frame.place(x=850, y=110)

        self.play_button = Button(self.control_frame, image=self.button_images.get("play"), command=self.play_video, bg="#077341", bd=0, relief='flat')
        self.play_button.grid(row=0, column=0, padx=5)

        self.stop_button = Button(self.control_frame, image=self.button_images.get("stop"), command=self.stop_video, bg="#077341", bd=0, relief='flat')
        self.stop_button.grid(row=0, column=1, padx=5)

        self.camera_button = Button(self.control_frame, image=self.button_images.get("camera"), command=self.take_screenshot, bg="#077341", bd=0, relief='flat')
        self.camera_button.grid(row=0, column=2, padx=5)

        # 11. Label de FPS
        self.lblFPS = Label(self.window, text="FPS: 0", font=('Arial', 10, 'bold'), 
                            bg="black", fg="lime")
        self.lblFPS.place(x=350, y=120)

    def update_conf(self, value):
        self.conf_threshold = float(value) / 100.0

    def update_brightness(self, value):
        self.brightness = int(value)

    def update_contrast(self, value):
        self.contrast = float(value) / 100.0

    def load_icon_images(self):
        icon_dir = "./Tkinter"
        icons = {}
        for file in os.listdir(icon_dir):
            if file.startswith("icon-") and file.endswith(".png"):
                class_name = file[5:-4]
                try:
                    img = Image.open(os.path.join(icon_dir, file))
                    img = img.resize((32, 32), Image.Resampling.LANCZOS)
                    icons[class_name] = ImageTk.PhotoImage(img)
                except Exception as e:
                    print(f"Error loading icon {file}: {e}")
        return icons

    def load_button_images(self):
        button_dir = "./Tkinter"
        buttons = {}
        for name in ["play", "stop", "camera"]:
            file = f"{name}.png"
            try:
                img = Image.open(os.path.join(button_dir, file))
                img = img.resize((32, 32), Image.Resampling.LANCZOS)
                buttons[name] = ImageTk.PhotoImage(img)
            except Exception as e:
                print(f"Error loading button {file}: {e}")
        return buttons

    def load_recycling_image(self):
        path = "./Tkinter/icons8-recycling-100.png"
        try:
            img = Image.open(path)
            img = img.resize((100, 100), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error loading recycling image: {e}")
            return None

    def update_selected_classes(self):
        self.selected_class_ids = [idx for idx, var in enumerate(self.class_vars) if var.get()]

    def select_all_classes(self):
        for var in self.class_vars:
            var.set(True)
        self.update_selected_classes()

    def deselect_all_classes(self):
        for var in self.class_vars:
            var.set(False)
        self.update_selected_classes()

    def play_video(self):
        if not self.thread.is_alive():
            self.is_running = True
            self.thread = threading.Thread(target=self.video_loop, daemon=True)
            self.thread.start()

    def stop_video(self):
        self.is_running = False
        self.current_frame = None

    def take_screenshot(self):
        with self.lock:
            frame_to_save = self.current_frame.copy() if self.current_frame else None

        if frame_to_save:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            frame_to_save.save(filename)
            messagebox.showinfo("Captura", f"Imagen guardada como {filename}")
        else:
            messagebox.showwarning("Captura", "No hay imagen disponible para capturar")

    def load_class_names(self):
        if hasattr(self.model, "names") and self.model.names:
            return [self.model.names[key] for key in sorted(self.model.names)]
        return []

    def update_info_frame(self, conteo):
        for widget in self.info_frame.winfo_children():
            widget.destroy()

        if not conteo:
            label = Label(self.info_frame, text="No se detectaron objetos", bg="#077341", fg="white", font=('Arial', 12, 'bold'))
            label.pack(anchor="w")
            return

        for nombre, cantidad in conteo.items():
            item_frame = Frame(self.info_frame, bg="#077341")
            item_frame.pack(anchor="w", pady=2)

            icon_name = nombre.lower()
            if icon_name in self.icon_images:
                icon_label = Label(item_frame, image=self.icon_images[icon_name], bg="#077341")
                icon_label.pack(side=LEFT)
            else:
                icon_label = Label(item_frame, text="[Icon]", bg="#077341", fg="white", font=('Arial', 10))
                icon_label.pack(side=LEFT)

            text_label = Label(item_frame, text=f"{nombre.capitalize()} : {cantidad}", bg="#077341", fg="white", font=('Arial', 12, 'bold'))
            text_label.pack(side=LEFT, padx=(5, 0))

    def ajusta_brillo_y_contraste(self, imagen):
        if self.brightness == 0 and self.contrast == 1.0:
            return imagen

        imagen_ajustada = cv2.convertScaleAbs(imagen, alpha=self.contrast, beta=self.brightness)
        return imagen_ajustada

    def procesar_conteo(self, resultados):
        """Calcula el conteo del frame actual como Counter."""
        if resultados is None or len(resultados) == 0:
            return Counter()

        try:
            clases_id = resultados[0].boxes.cls.tolist()
            nombres_dict = resultados[0].names
            nombres_detectados = [nombres_dict[i] for i in clases_id]
            return Counter(nombres_detectados)
        except:
            return Counter()

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
            current_time = time.time()
            if self.prev_time > 0:
                self.fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time

            ret, frame = self.cap.read()
            if ret:
                # Redimensionar para el GUI
                frame = cv2.resize(frame, (720, 405))

                # Ajustar brillo/contraste para condiciones de poca luz
                frame = self.ajusta_brillo_y_contraste(frame)

                # Inferencia YOLO
                if self.selected_class_ids:
                    resultados = self.model.predict(frame, imgsz=640, conf=self.conf_threshold, classes=self.selected_class_ids, verbose=False)
                else:
                    resultados = self.model.predict(frame, imgsz=640, conf=self.conf_threshold, verbose=False)

                # Procesar datos
                conteo = self.procesar_conteo(resultados)

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
                    self.current_counts = conteo
                    self.current_fps = self.fps
            else:
                time.sleep(0.1)  # Si la cámara falla, esperar un poco antes de reintentar

    def update_gui_loop(self):
        """
        Lógica del Hilo Principal (Tkinter):
        Lee las variables compartidas y actualiza la pantalla.
        """
        # 1. Obtener datos del hilo secundario de forma segura
        img_para_mostrar = None
        conteo_para_mostrar = Counter()
        fps_to_show = 0

        with self.lock:
            if self.current_frame is not None:
                img_para_mostrar = self.current_frame
                # Importante: Una vez tomada, podemos ponerla en None para no repintar lo mismo
                # (opcional, depende de la fluidez deseada)
            conteo_para_mostrar = self.current_counts.copy()
            fps_to_show = self.current_fps

        # 2. Actualizar Widgets (Solo si hay imagen nueva)
        if img_para_mostrar:
            img_tk = ImageTk.PhotoImage(image=img_para_mostrar)
            self.lblVideo.configure(image=img_tk)
            self.lblVideo.image = img_tk  # Referencia para evitar Garbage Collection
        else:
            self.lblVideo.configure(image='', bg="black")

        # Actualizar información de detecciones
        if conteo_para_mostrar != getattr(self, 'last_conteo', Counter()):
            self.update_info_frame(conteo_para_mostrar)
            self.last_conteo = conteo_para_mostrar.copy()

        # Actualizar FPS
        self.lblFPS.config(text=f"FPS: {fps_to_show:.2f}")

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