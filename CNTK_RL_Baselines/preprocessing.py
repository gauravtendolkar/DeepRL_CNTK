from PIL import Image
import numpy as np

def downscale(observation0):
    im = Image.fromarray(observation0, mode='RGB')
    im = im.split()[0]
    im = im.resize((84, 84), Image.BILINEAR)
    return np.array(im, dtype=np.float32).reshape(1,84,84)
