
import fiftyone as fo
import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    label_types=["detections"],
    classes=["Light bulb"], #aca escribir la clase a buscar
    seed=51,
    shuffle=True,
)

session = fo.launch_app(dataset.view())
session.wait()

#mas informacion https://docs.voxel51.com/tutorials/open_images.html
## https://storage.googleapis.com/openimages/web/index.html