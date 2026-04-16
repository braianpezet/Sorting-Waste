from ultralytics import YOLO

# Cargamos tus pesos de la tesis
model = YOLO('best.pt')

# Exportamos a OpenVINO para CPU Intel
# imgsz=640 asegura que mantenga la resolución que te dio buen mAP
model.export(format='openvino', imgsz=640)
print("✅ Exportación completada. Busca la carpeta 'best_openvino_model'")