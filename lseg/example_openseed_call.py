import os, sys
# Add the parent directory (MSc-Thesis) to the module search path:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.call_openseed import openseed_pixels
import numpy as np
pix = openseed_pixels("/home/jovyan/teaching_material/msc/OpenSeeD/images/animals.png")    # ndarray (256, H, W)
print(pix.shape)
