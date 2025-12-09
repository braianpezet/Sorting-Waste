import os

# 1. Forzamos una ruta nueva y corta en el disco C:
# Esto evita problemas de permisos o rutas largas en "Users/tu_usuario/..."
os.environ['FIFTYONE_DATABASE_DIR'] = r'C:\fo_db_temp'

# 2. Ahora sí importamos fiftyone
import fiftyone as fo
import fiftyone.zoo as foz

# ... El resto de tu código sigue igual ...
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Wine glass"], 
    max_samples=300,
    seed=51,
    shuffle=True,
    only_matching=True
)

output_dir = "dataset_xml_temporal"

# (Asegúrate de crear el directorio de salida si no existe, por seguridad)
os.makedirs(output_dir, exist_ok=True)

dataset.export(
    export_dir=output_dir,
    dataset_type=fo.types.VOCDetectionDataset,
    label_field="ground_truth",
)

print("¡Listo! Revisa C:\\fo_db_temp si quieres ver que se crearon archivos, luego puedes borrar esa carpeta.")