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


def FengTan(inp_path, out_path, a1, k1, k2, gamma=2, R=128, win_size=15):
    to_halftone(inp_path, out_path)
    inp_img = pim.open(out_path)
    inp_arr = np.array(inp_img)
    h, w = inp_arr.shape[:2]
    out_arr = np.zeros((h, w, inp_arr.shape[2]), dtype=inp_arr.dtype)

    M = inp_arr.min()
    padded_img = np.pad(inp_arr, win_size, mode='reflect')

    for y in range(h):
        for x in range(w):
            window = padded_img[y : y + win_size, x : x + win_size]
            s = np.std(window)
            m = np.mean(window)

            a2 = k1 * (s / R) ** gamma
            a3 = k2 * (s / R) ** gamma
            T = (1 - a1) * m + a2 * s * (m - M) / R + a3 * M
            if inp_arr[y, x][0] > T:
                out_arr[y, x] = 255
            else:
                out_arr[y, x] = 0
    
    out_img = pim.fromarray(out_arr)
    out_img.save(out_path)

def main():
    print("Please wait...")
    for img in iglob('../input/*.png'):
        out_path = './output/' + os.path.basename(img)
        FengTan(img, out_path, 0.15, 0.2, 0.03)
    print("Done!")


if __name__ == '__main__':
    main()