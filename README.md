
# Sorting Waste 

Proyecto de clasificación de reciclables a traves IA y machine learning

## Clasificación de residuos
La parte de clasificacion de residuos consta principalmente de dos archivos **entrenador.ipynb** y **loadModel.ipynb**.

**entrenador.ipynb**

Usa transfer learning y mobilnetV2 para entrenar un modelo de clasificacion por imagenes.

Recomiendo guardar los modelos entrenados, cambiando estas rutas en el codigo para no sobreescribir los modelos anteriores

```bash
  modelo.save('saved_model/my_model2')
```

**loadModel.ipynb**

Basicamente carga un modelo guardado en la carpeta saved_model y ejecuta un algoritmo de opencv que va sacando capturas de pantalla y va mostrando la prediccion en pantalla.


## Detección de residuos

La otra parte del proyecto tiene que ver con deteccion de objetos. Todo esto se encuentra en la carpeta **POCYOLO**

Actualmente para esto recomiendo el uso del ide pycharm en ves de usar anaconda y jupyter notebook.

**dataset.yaml**

Este es uno de los archivos más importantes, en el se configuran las rutas de las imagenes y la cantidad de clases

```bash
  train:  C:\Users\braianpezet\Nueva carpeta\Sorting-Waste\pocyolo\Data\train\
  val:  C:\Users\braianpezet\Nueva carpeta\Sorting-Waste\pocyolo\Data\val\
  test:
  nc: 4
  names: ["glass", "metal", "papel", "plastic"]
```

deben poner las rutas correspondientes a las carpetas de su computadora para el entrenamiento.

**Estructura de los directorios**

Basicamente hay dos carpetas principales **train** y **val**, en el hay que poner las fotos todas juntas, junto con las anotaciones. En mi caso use un 10% para validacion.

**Prepar conjunto de datos**

Para preparar un conjunto de datos, puede que este este en el formato xml y no en el formato YOLO para ello hay que poner todos los datos como pide el script **xml2yolo.py**, este se encargara de realizar la conversion de manera automatica

**Hacer un entrenamiento**

En mi caso use la consola de pychar, en la cual estos son unos ejemplos de los comandos que se pueden usar para el entrenamiento

```bash
yolo task=detect mode=train epochs=30 data=dataset.yaml model=yolov8m.pt imgsz=640 batch=1
```

A su ves si tienen algun problema como un corte de luz es posible reanudar el entrenamiento con estos comandos

```bash
Yolo task=detect mode=train resume model=/runs/detect/train/weights/last.pt data=dataset.yaml epochs=10 imgsz=640 batch=2
```

Para más informacion consultar https://github.com/ultralytics/ultralytics
hay distintos modelos que se pueden utilizar algunos más pesados que otros.

El entrenamiento va a generar la carpeta **runs** donde los archivos más importantes son **best** y **last**, last se puede usar para reanudar un entrenamiento y best para usar el codigo de la camara que es **segment.py**

el archivo **best** actual esta entrenado con el modelo "s" y con 50 epocas
Se recomienda hacer una copia de los modelos entrenados para luego poder hacer comparaciones

**Archivo segment.py**

Este codigo carga un modelo y muestra los resultados a traves de la camara, en la ultima actualizacion se imprimen en pantalla las coordenadas. Pueden jugar con el tamaño de la ventana, etc.

Algunos de los requerimientos para ejecutar el programa son:

```bash
pip install ultralytics
pip install opencv-python
```
Para usar OpenCv tambien es necesario tener instalado

https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

Recomiendo el tutorial de Aprende e ingenia. Muchas gracias!

https://www.youtube.com/watch?v=rk7zOBRJWCc






