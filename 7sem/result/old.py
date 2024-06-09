import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import iglob
from PIL import Image, ImageFont, ImageDraw
from scipy.signal import convolve2d


def calculate_stats(img):
    bin_img = np.zeros(shape=img.shape, dtype=int)
    bin_img[img != 255] = 1

    w, h = bin_img.shape[:2]
    # Calculate com - center of mass
    weight = np.sum(bin_img)
    y_idx, x_idx = np.indices(bin_img.shape)
    y_com = np.sum(y_idx * bin_img) / weight
    x_com = np.sum(x_idx * bin_img) / weight
    com = (x_com, y_com)

    norm_com = ((x_com - 1) / (w - 1), (y_com - 1) / (h - 1))

    # Calculate inertia
    i_x = np.sum((y_idx - y_com) ** 2 * bin_img) / weight
    i_y = np.sum((x_idx - x_com) ** 2 * bin_img) / weight
    i = (i_x, i_y)
    norm_i_x = i_x / (w**2 * h**2)
    norm_i_y = i_y / (w**2 * h**2)
    norm_i = (norm_i_x, norm_i_y)

    return {
        "weight": weight,
        "x_center_of_mass": x_com,
        "y_center_of_mass": y_com,
        "inertia_x": i_x,
        "inertia_y": i_y,
    }


def generate_symbol_imgs(font_path, out_folder, font_size=52, inverted=False):
    os.makedirs(out_folder, exist_ok=True)
    font = ImageFont.truetype(font_path, font_size)
    symbols = "αβγδεζηθικλμνξοπρστυφχψω"

    for symbol in symbols:
        w, h = font.getsize(symbol)
        image_size = (w, h)
        image = Image.new("RGB", image_size, color="white")
        draw = ImageDraw.Draw(image)
        sym_pos = ((image_size[0] - w) // 2, (image_size[1] - h) // 2)
        draw.text(sym_pos, symbol, fill=(0, 0, 0), font=font)
        img_arr = np.asarray(image.convert("L"), dtype=np.uint8)
        if inverted:
            img_arr = 255 - img_arr
        image = Image.fromarray(img_arr.astype(np.uint8), mode="L")
        image.save(out_folder + f"{symbol}.png")


def write_stats(out_path):
    with open(f"{out_path}data.csv", "w", newline="") as csvfile:
        fieldnames = ["weight", "center_of_mass", "inertia"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for img_path in iglob("./output/orig_symb/*.png"):
            img = np.array(Image.open(img_path).convert("L"))
            stats = calculate_stats(img)
            writer.writerow(stats)


def calculate_profiles():
    os.makedirs("./output/profiles/x", exist_ok=True)
    os.makedirs("./output/profiles/y", exist_ok=True)
    for img_path in iglob("./output/orig_symb/*.png"):
        img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        bin_img = np.zeros(shape=img.shape, dtype=np.uint8)
        bin_img[img != 255] = 1

        x_values = np.arange(start=1, stop=bin_img.shape[0] + 1).astype(int)
        height_values = np.sum(bin_img, axis=1)
        plt.bar(x=x_values, height=height_values)
        plt.ylim(0, 60)
        plt.xlim(0, 60)
        plt.savefig(f"./output/profiles/y/{os.path.basename(img_path)}")
        plt.clf()

        x_values = x_values = np.arange(start=1, stop=bin_img.shape[1] + 1).astype(int)
        height_values = np.sum(bin_img, axis=0)
        plt.bar(x=x_values, height=height_values)
        plt.ylim(0, 60)
        plt.xlim(0, 60)
        plt.savefig(f"./output/profiles/x/{os.path.basename(img_path)}")
        plt.clf()


def main():
    print("Please wait...")
    generate_symbol_imgs("./input/TimesNewRomanRegular.ttf", "./output/orig_symb/")
    generate_symbol_imgs(
        "./input/TimesNewRomanRegular.ttf", "./output/inverted_symb/", inverted=True
    )
    out_path = "./output/"
    write_stats(out_path)
    calculate_profiles()


if __name__ == "__main__":
    main()
