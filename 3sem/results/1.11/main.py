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
    out_arr = np.zeros((h, w), dtype=np.uint8)

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
    
    out_img = pim.fromarray(out_arr, mode='L')
    out_img.save(out_path)


def check_window(img_arr, x, y, win_size):
    y_upper_border = y - win_size if y - win_size >= 0 else 0
    y_lower_border = y + win_size if y + win_size < img_arr.shape[0] else img_arr.shape[0] - 1
    x_left_border = x - win_size if x - win_size >= 0 else 0
    x_right_border = x + win_size if x + win_size < img_arr.shape[1] else img_arr.shape[1] - 1

    for i in range(y_upper_border, y_lower_border + 1):
        for j in range(x_left_border,x_right_border + 1):
            if img_arr[i, j] == 0:
                return False
    return True


def Opening(inp_path, out_path, bin_out_path, win_size=1):
    FengTan(inp_path, bin_out_path, 0.15, 0.2, 0.03)
    inp_img = pim.open(bin_out_path)
    inp_arr = np.array(inp_img)
    h, w = inp_arr.shape[:2]
    out_arr = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            flag = check_window(inp_arr, x, y, win_size)
            out_arr[y, x] = 255 if flag else 0

    out_img = pim.fromarray(out_arr, mode='L')
    out_img.save(out_path)

def diff_img(bin_img_path, opened_img_path, out_path):
    bin_img = pim.open(bin_img_path)
    bin_arr = np.array(bin_img)
    opened_img = pim.open(opened_img_path)
    opened_arr = np.array(opened_img)

    h, w = opened_arr.shape[:2]
    out_arr = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            out_arr[y, x] = bin_arr[y, x] ^ opened_arr[y, x]

    out_img = pim.fromarray(out_arr, mode='L')
    out_img.save(out_path)



def main():
    print("Please wait...")
    for img_path in iglob('../input/*.png'):
        bin_out_path = './output/' + 'bin_' + os.path.basename(img_path)
        opened_out_path = './output/' + 'opened_' + os.path.basename(img_path)
        Opening(img_path, opened_out_path, bin_out_path)
        diff_out_path = './output/' + 'diff_' + os.path.basename(img_path)
        diff_img(bin_out_path, opened_out_path, diff_out_path)
        print(img_path)
    print("Done!")

if __name__ == '__main__':
    main()