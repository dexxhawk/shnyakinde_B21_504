import os
from glob import iglob
import numpy as np
from PIL import Image as pim

def resample(inp_path, out_path, m, n):
    inp_img = pim.open(inp_path)
    if inp_img.mode != 'RGB':
        inp_img = inp_img.convert('RGB')

    inp_arr = np.array(inp_img)

    h, w = inp_arr.shape[:2]

    new_h = h * m // n
    new_w = w * m // n

    out_arr = np.zeros((new_h, new_w, inp_arr.shape[2]), dtype=inp_arr.dtype)

    for y in range(new_h):
        for x in range(new_w):
            orig_y = n * y // m
            orig_x = n * x // m
            out_arr[y, x] = inp_arr[orig_y, orig_x]
    
    out_img = pim.fromarray(out_arr)
    out_img.save(out_path)

def main():
    m = int(input("Enter m: "))
    n = int(input("Enter n: "))
    print("Please wait...")
    for img in iglob('../input/*.png'):
        out_path = './output/' + os.path.basename(img)
        resample(img, out_path, m, n)
    print("Done!")


if __name__ == '__main__':
    main()