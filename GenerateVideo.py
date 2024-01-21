import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from subprocess import call
from os import remove, chdir

def generate_video(img, name):
    for i in range(len(img)):
        plt.imshow(img[i], cmap="hot", vmin=np.min(img), vmax=np.max(img))
        plt.savefig("file%02d.png" % i, dpi=250)
        
    call([
        'ffmpeg', '-loglevel', 'panic', '-framerate', '32', '-i', 'file%02d.png', '-r', '32', '-pix_fmt', 'yuv420p',
         name, '-y'
    ])
    for file_name in glob("*.png"):
        remove(file_name)