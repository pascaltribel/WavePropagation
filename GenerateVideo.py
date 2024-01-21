import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from subprocess import call
from os import remove, chdir
from tqdm.notebook import tqdm
from ColorMap import get_seismic_cmap

def generate_video(img, name, dt, verbose=False):
    if verbose:
        print("Generating", len(img), "images.")
    for i in tqdm(range(len(img))):
        plt.imshow(img[i], cmap=get_seismic_cmap(), vmin=-np.max(np.abs(img[i:])), vmax=np.max(np.abs(img[i:])))
        plt.title("t = " + str(dt*i)[:4] + "s")
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(name + "%02d.png" % i, dpi=250)
        plt.close()
        
    call([
        'ffmpeg', '-loglevel', 'panic', '-framerate', '32', '-i', name + '%02d.png', '-r', '32', '-pix_fmt', 'yuv420p',
         name + ".mp4", '-y'
    ])
    for file_name in glob("*.png"):
        remove(file_name)