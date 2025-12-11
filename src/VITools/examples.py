from .phantom import Phantom
from .hooks import hookimpl
import numpy as np
# Below is an example of creating a custom phantom and registering it to the phantoms list

class WaterPhantom(Phantom):
    'basic example of a phantom'
    patient_name = 'Water Phantom'

    def __init__(self, matrix_size=100, radius=50, HU=100):
        image_shape = 3*[matrix_size]
        img = np.full(image_shape, -1000, dtype=np.int16)  # Air
        center = tuple(s // 2 for s in image_shape)
        z, x, y = np.ogrid[-center[0]:image_shape[0]-center[0], -center[1]:image_shape[1]-center[1], -center[2]:image_shape[2]-center[2]]
        mask = x*x + y*y + z*z <= radius*radius
        img[mask] = HU  # Set sphere to a value like soft tissue

        # Define voxel spacings in mm (z, x, y)
        spacings = (1, 2, 2)
        super().__init__(img, spacings, patient_name=WaterPhantom.patient_name)

@hookimpl
def register_phantom_types():
    return {WaterPhantom.patient_name: WaterPhantom}