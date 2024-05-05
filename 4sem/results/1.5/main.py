import os
import numpy as np
from glob import iglob
from PIL import Image as pim
from scipy.signal import convolve2d


def binarize(inp_path, out_path, threshold):
    print(inp_path)
    inp_img = pim.open(inp_path)    
    inp_arr = np.array(inp_img)
    pim.fromarray(np.where(inp_arr < threshold, 0, 255).astype(np.uint8), mode='L').save(out_path)


def to_semitone(inp_path, out_path):
    inp_img = pim.open(inp_path)
    if inp_img.mode != 'RGB':
        inp_img = inp_img.convert('RGB')

    inp_arr = np.array(inp_img)
    h, w = inp_arr.shape[:2]
    out_arr = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            out_arr[y, x] = np.mean(inp_arr[y, x])
    
    out_img = pim.fromarray(out_arr, mode='L')
    out_img.save(out_path)


def sharr(semitone_img_path, out_path):
    inp_img = pim.open(semitone_img_path)
    inp_arr = np.array(inp_img)

    kernel_x = np.array([[3, 0, -3], 
                [10, 0, -10],
                [3, 0, -3]])
    kernel_y = np.array([[3, 10, 3], 
                [0, 0, 0],
                [-3, -10, -3]])
    
    Gx = convolve2d(inp_arr, kernel_x, mode='same')
    Gy = convolve2d(inp_arr, kernel_y, mode='same')
    G = np.sqrt(Gx**2 + Gy**2)

    G_max = np.max((np.max(Gx), np.max(Gy), np.max(G)))
    Gx = Gx * 255 / G_max
    Gy = Gy * 255 / G_max
    G = G * 255 / G_max

    pim.fromarray(Gx.astype(np.uint8), mode='L').save(out_path + 'Gx_' + os.path.basename(semitone_img_path))
    pim.fromarray(Gy.astype(np.uint8), mode='L').save(out_path + 'Gy_' + os.path.basename(semitone_img_path))
    G_path = out_path + 'G_' + os.path.basename(semitone_img_path)
    pim.fromarray(G.astype(np.uint8), mode='L').save(G_path)
    return G_path

def main():
    print("Please wait...")
    for img_path in iglob('../input/*.png'):
        out_folder_name = './output/' + os.path.basename(img_path) + '/'
        if not os.path.exists(out_folder_name):
            os.makedirs(out_folder_name)
        to_semitone(img_path, out_folder_name + os.path.basename(img_path))
        G_path = sharr(out_folder_name + os.path.basename(img_path), out_folder_name)
        binarize(G_path, out_folder_name + "bin_" + os.path.basename(G_path), 27)
    print("Done!")


if __name__ == '__main__':
    main()
