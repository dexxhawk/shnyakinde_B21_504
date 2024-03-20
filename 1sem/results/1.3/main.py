import os
from glob import iglob
import numpy as np
from PIL import Image as pim

def interpolate(inp_path, out_path, m):
    inp_img = pim.open(inp_path)
    if inp_img.mode != 'RGB':
        inp_img = inp_img.convert('RGB')

    inp_arr = np.array(inp_img)

    h, w = inp_arr.shape[:2]

    new_h = h * m
    new_w = w * m

    out_arr = np.zeros((new_h, new_w, inp_arr.shape[2]), dtype=inp_arr.dtype)

    for y in range(new_h):
        for x in range(new_w):
            orig_y = y // m
            orig_x = x // m
            out_arr[y, x] = inp_arr[orig_y, orig_x]
    
    out_img = pim.fromarray(out_arr)
    out_img.save(out_path)

def decimate(inp_path, out_path, n):
    inp_img = pim.open(inp_path)
    if inp_img.mode != 'RGB':
        inp_img = inp_img.convert('RGB')

    inp_arr = np.array(inp_img)

    h, w = inp_arr.shape[:2]

    new_h = h // n
    new_w = w // n

    out_arr = np.zeros((new_h, new_w, inp_arr.shape[2]), dtype=inp_arr.dtype)

    for y in range(new_h):
        for x in range(new_w):
            orig_y = n * y
            orig_x = n * x
            out_arr[y, x] = inp_arr[orig_y, orig_x]
    
    out_img = pim.fromarray(out_arr)
    out_img.save(out_path)

def main():
    m = int(input("Enter m: "))
    n = int(input("Enter n: "))
    print("Please wait...")
    for img in iglob('../input/*.png'):
        out_path = './output/' + os.path.basename(img)
        interpolate(img, out_path, m)
        decimate(out_path, out_path, n)
    print("Done!")


if __name__ == '__main__':
    main()