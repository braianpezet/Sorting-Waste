import os
import glob
import matplotlib.pyplot as plt

def contar_y_graficar_etiquetas(carpeta_anotaciones):
    archivo_clases = os.path.join(carpeta_anotaciones, "classes.txt")
    
    # 1. Leer las clases
    if not os.path.exists(archivo_clases):
        print(f"❌ Error: No se encontró el archivo 'classes.txt' en {carpeta_anotaciones}")
        return

    with open(archivo_clases, 'r') as f:
        # Guardamos las clases tal cual están en el archivo (el índice es la posición en la lista)
        clases = [linea.strip() for linea in f.readlines() if linea.strip()]
    
    if not clases:
        print("❌ Error: El archivo 'classes.txt' está vacío.")
        return

    # Inicializar un diccionario para contar las apariciones de cada clase (empiezan en 0)
    conteo_clases = {clase: 0 for clase in clases}

    # 2. Leer todos los archivos .txt (ignorando classes.txt)
    archivos_txt = glob.glob(os.path.join(carpeta_anotaciones, "*.txt"))
    
    for ruta_txt in archivos_txt:
        if os.path.basename(ruta_txt) == "classes.txt":
            continue
            
        with open(ruta_txt, 'r') as f:
            for linea in f:
                elementos = linea.strip().split()
                if not elementos:
                    continue
                
                try:
                    # El primer valor es el índice de la clase en YOLO
                    indice_clase = int(elementos[0])
                    
                    if 0 <= indice_clase < len(clases):
                        nombre_clase = clases[indice_clase]
                        conteo_clases[nombre_clase] += 1
                except ValueError:
                    continue # Ignorar si la línea no empieza con un número

    # 3. Preparar los datos para el gráfico (ordenados de mayor a menor)
    # Ordenamos el diccionario por la cantidad de etiquetas (el valor)
    conteo_ordenado = dict(sorted(conteo_clases.items(), key=lambda item: item[1], reverse=True))
    
    nombres = list(conteo_ordenado.keys())
    cantidades = list(conteo_ordenado.values())

    # Mostrar por consola el resumen
    print("\n📊 Resumen de etiquetas:")
    for nombre, cantidad in conteo_ordenado.items():
        print(f" - {nombre}: {cantidad}")

    # 4. Generar el gráfico de barras
    plt.figure(figsize=(10, 6)) # Tamaño de la ventana
    barras = plt.bar(nombres, cantidades, color='skyblue', edgecolor='black')
    
    # Personalizar el gráfico
    plt.title('Cantidad de Etiquetas por Clase', fontsize=14, fontweight='bold')
    plt.xlabel('Clases', fontsize=12)
    plt.ylabel('Cantidad de Apariciones', fontsize=12)
    
    # Rotar las etiquetas del eje X por si los nombres son muy largos y evitar que se superpongan
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Agregar el número exacto encima de cada barra
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., altura,
                 f'{int(altura)}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout() # Ajusta los márgenes automáticamente
    
    # Guardar el gráfico como imagen y luego mostrarlo
    plt.savefig(os.path.join(carpeta_anotaciones, 'grafico_etiquetas.png'))
    print(f"\n✅ Gráfico guardado exitosamente como 'grafico_etiquetas.png' en tu carpeta.")
    
    plt.show()

# ==========================================
# CONFIGURACIÓN 
# ==========================================
# Cambia esta ruta por la de tu carpeta de imágenes y anotaciones
ruta_carpeta = r"F:\Proyecto Reciclables backup 1\datasets\Dataset\Dataset Yolo\Dataset-crudo\labels"

contar_y_graficar_etiquetas(ruta_carpeta)