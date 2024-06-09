import csv
import old
from glob import iglob
import math
import numpy as np
from PIL import Image

SYMBOLS = "αβγδεζηθικλμνξοπρστυφχψω"
TEXT = "Καλήώρατηςημέρας"


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


def get_alphabet_info():
    def parse_tuple(string):
        return tuple(map(float, string.strip("()").split(",")))

    tuples_list = dict()
    with open("output/data.csv", "r") as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            weight = int(row["weight"])
            center_of_mass = parse_tuple(row["center_of_mass"])
            inertia = parse_tuple(row["inertia"])
            tuples_list[SYMBOLS[i]] = weight, *center_of_mass, *inertia
            i += 1
    return tuples_list


def create_hypothesis(alphabet_info, target_features):
    def euclidean_distance(feature1, feature2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(feature1, feature2)))

    distances = dict()
    for letter, features in alphabet_info.items():
        distance = euclidean_distance(target_features, features)
        distances[letter] = distance

    max_distance = max(distances.values())

    similarities = [
        (letter, round(1 - distance / max_distance, 2))
        for letter, distance in distances.items()
    ]

    return sorted(similarities, key=lambda x: x[1])


def get_phrase_from_hypothesis(img: np.array, bounds) -> str:
    alphabet_info = get_alphabet_info()
    res = []
    for start, end in bounds:
        letter_features = calculate_stats(img[:, start:end])
        hypothesis = create_hypothesis(alphabet_info, letter_features)
        best_hypotheses = hypothesis[-1][0]
        res.append(best_hypotheses)
    return "".join(res)


def write_res(recognized_phrase: str):
    max_len = max(len(TEXT), len(recognized_phrase))
    orig = TEXT.ljust(max_len)
    detected = recognized_phrase.ljust(max_len)

    with open("output/result.txt", "w") as f:
        correct_letters = 0
        by_letter = ["has | got"]
        for i in range(max_len):
            is_correct = orig[i] == detected[i]
            by_letter.append(f"{orig[i]}\t{detected[i]}\t{is_correct}")
            correct_letters += int(is_correct)
        f.write(
            "\n".join(
                [
                    f"correct:     {math.ceil(correct_letters / max_len * 100)}%\n\n",
                    f"phrase:      {orig}",
                    f"detected:    {detected}",
                ]
            )
        )
        f.write("\n".join(by_letter))


if __name__ == "__main__":
    img = np.array(Image.open(f"input/original_text.bmp").convert("L"))
    old.generate_symbol_imgs("./input/TimesNewRomanRegular.ttf", "./output/orig_symb/")
    bounds = split_into_characters(img)
    out_path = "./output/"
    old.write_stats(out_path)
    recognized_phrase = get_phrase_from_hypothesis(img, bounds)
    write_res(recognized_phrase)
