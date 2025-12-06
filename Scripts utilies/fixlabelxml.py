import os
import glob
import xml.etree.ElementTree as ET

# --- CONFIGURA ESTO ---
input_dir = "annotations/"
etiqueta_mala = "plastico"   # <--- Pon aquí el nombre en español que usaste por error
etiqueta_buena = "plastic"   # <--- Pon aquí el nombre correcto que quieres usar
# ----------------------

files = glob.glob(os.path.join(input_dir, '*.xml'))
archivos_corregidos = 0

print(f"Buscando la etiqueta '{etiqueta_mala}' para cambiarla a '{etiqueta_buena}'...")

for fil in files:
    tree = ET.parse(fil)
    root = tree.getroot()
    cambio_realizado = False

    # Revisamos cada objeto dentro del XML
    for obj in root.findall('object'):
        name_node = obj.find("name")
        
        if name_node.text == etiqueta_mala:
            name_node.text = etiqueta_buena
            cambio_realizado = True

    # Si encontramos el error y lo corregimos, guardamos el archivo
    if cambio_realizado:
        tree.write(fil)
        archivos_corregidos += 1
        print(f"Corregido: {os.path.basename(fil)}")

print(f"\nListo. Se corrigieron {archivos_corregidos} archivos.")