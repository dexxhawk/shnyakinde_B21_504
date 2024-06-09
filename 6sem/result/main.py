from glob import iglob
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageOps


TEXT = 'Καλή ώρα της ημέρας'
FONT = ImageFont.truetype("input/TimesNewRomanRegular.ttf", 52)
THRESHOLD = 100

def generate_text_image():
    spacing = 1
    width = sum(FONT.getsize(ch)[0] for ch in TEXT) + spacing * (len(TEXT) - 1)
    height = max(FONT.getsize(ch)[1] for ch in TEXT)
    img = Image.new("L", (width, height), "white")
    drawer = ImageDraw.Draw(img)

    current_x = 0
    for ch in TEXT:
        char_width, char_height = FONT.getsize(ch)
        drawer.text((current_x, height - char_height), ch, "black", font=FONT)
        current_x += char_width + spacing
    img_data = np.array(img)
    binary_img = (img_data > THRESHOLD).astype(np.uint8) * 255
    img = Image.fromarray(binary_img)
    os.makedirs("output", exist_ok=True)
    img.save("output/original_text.bmp")
    ImageOps.invert(img).save("output/inverted_text.bmp")
    create_profiles(binary_img)
    return binary_img


def create_profiles(image_data):
    os.makedirs("output/profiles", exist_ok=True)
    binary_image = (image_data != 255).astype(int)
    plt.bar(
        x=np.arange(1, binary_image.shape[1] + 1),
        height=np.sum(binary_image, axis=0),
        width=1
    )
    plt.savefig('output/profiles/horizontal_profile.png')
    plt.clf()

    plt.barh(
        y=np.arange(1, binary_image.shape[0] + 1),
        width=np.sum(binary_image, axis=1),
        height=1
    )
    plt.savefig('output/profiles/vertical_profile.png')
    plt.clf()

def split_into_characters(image_data):
    projection = np.sum(image_data == 0, axis=0)
    in_character = False
    char_boundaries = []

    for idx in range(len(projection)):
        if projection[idx] > 0:
            if not in_character:
                in_character = True
                start = idx
        else:
            if in_character:
                in_character = False
                end = idx
                char_boundaries.append((start - 1, end))

    if in_character:
        char_boundaries.append((start, len(projection)))

    return char_boundaries

def draw_bboxes(image_data, bounds):
    img = Image.fromarray(image_data)
    draw = ImageDraw.Draw(img)
    for left, right in bounds:
        draw.rectangle([left, 0, right, image_data.shape[0]], outline="red")
    img.save("output/boxed_text.bmp")


def calculate_symb_profiles(char_boundaries, img):
    os.makedirs("./output/profiles/x", exist_ok=True)
    os.makedirs("./output/profiles/y", exist_ok=True)
    i = 0
    for boundaries in char_boundaries:
        bin_img = np.zeros(shape=img.shape, dtype=np.uint8)
        bin_img[img != 255] = 1
        area = bin_img[: , boundaries[0] : boundaries[1]]
        x_values = np.arange(start=1, stop=area.shape[0] + 1).astype(int) 
        height_values = np.sum(area, axis=1)
        plt.bar(x=x_values, height=height_values)
        plt.ylim(0, 60)
        plt.xlim(0, 60)
        plt.savefig(f'./output/profiles/y/{i}')
        plt.clf()
        
        x_values = x_values = np.arange(start=1, stop=area.shape[1] + 1).astype(int)
        height_values = np.sum(area, axis=0)
        plt.bar(x=x_values, height=height_values)
        plt.ylim(0, 60)
        plt.xlim(0, 60)
        plt.savefig(f'./output/profiles/x/{i}')
        plt.clf()
        i += 1

if __name__ == "__main__":
    img_data = generate_text_image()
    char_bounds = split_into_characters(img_data)
    draw_bboxes(img_data, char_bounds)
    calculate_symb_profiles(char_bounds, img_data)
