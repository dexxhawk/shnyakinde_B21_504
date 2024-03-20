import os
from glob import iglob
import numpy as np
from PIL import Image as pim

def to_halftone(inp_path, out_path):
    inp_img = pim.open(inp_path)
    if inp_img.mode != 'RGB':
        inp_img = inp_img.convert('RGB')

    inp_arr = np.array(inp_img)
    h, w = inp_arr.shape[:2]
    out_arr = np.zeros((h, w, inp_arr.shape[2]), dtype=inp_arr.dtype)

    for y in range(h):
        for x in range(w):
            out_arr[y, x] = np.mean(inp_arr[y, x])
    
    out_img = pim.fromarray(out_arr)
    out_img.save(out_path)

def main():
    print("Please wait...")
    for img in iglob('../input/*.png'):
        out_path = './output/' + os.path.basename(img)
        to_halftone(img, out_path)
    print("Done!")


if __name__ == '__main__':
    main()