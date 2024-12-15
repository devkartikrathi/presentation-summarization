from utils import ImgSum
import os

def summary(res):
    for image_file in os.listdir("figures"):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join("figures", image_file)
            res.append(ImgSum.summarize_image(ImgSum._encode_image(image_path)))
    return res

print(summary([]))