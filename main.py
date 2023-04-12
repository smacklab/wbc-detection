from ultralytics import YOLO
import os
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm


wbc_model = YOLO("wbc-model-Feb24.pt")
model = YOLO("blood_smear_model_Apr10.pt")
if not os.path.exists("output"):
    os.makedirs("output")

def create_output_folders(image_name):
    if not os.path.exists("output/" + image_name):
        os.makedirs("output/" + image_name)

    # create folders for each class Dense, Good, Sparse
    for class_name in model.names.values():
        if not os.path.exists("output/" + image_name + "/" + class_name):
            os.makedirs("output/" + image_name + "/" + class_name)


if __name__ == "__main__":
    for filename in os.listdir("data"):
        f = os.path.join("data", filename)
        print(f)
        if not os.path.isfile(f) or not f.endswith(".ndpi"):
            continue
        with tifffile.TiffFile(f) as tif:
            image = tif.asarray()
            image = Image.fromarray(image)
            w, h = image.size
            print("NDPI Image size: {}x{}".format(w, h))
            image_name = os.path.splitext(filename)[0]
            create_output_folders(image_name)

            # split image into 512x512 tiles, column by column
            for c in tqdm(range(0, w, 512)):
                for r in tqdm(range(0, h, 512), leave=False):
                    cropped_image = image.crop((c, r, c + 512, r + 512))
                    cropped_image_name = image_name + \
                        "_cropped_" + str(c) + "_" + str(r)

                    # run prediction on image
                    results = model(cropped_image)
                    parsed_results = list([float(x) for x in results[0].probs])
                    max_index = np.argmax(parsed_results)
                    class_name = model.names[max_index]
                    if class_name == 'Dense':
                        continue
                    if parsed_results[2] < .99 and max_index == 1:
                        results = wbc_model(cropped_image)
                        for result in results:
                            boxes = result.boxes
                            for i, box in enumerate(boxes):
                                if box.conf < 0.25:
                                    continue
                                xyxy = box.xyxy.numpy()
                                for find in xyxy:
                                    left, top, right, bottom = int(find[0]), int(find[1]), int(find[2]), int(find[3])
                                    width = abs(left - right)
                                    height = abs(top - bottom)
                                    if (width/height < 0.7) or (height/width < 0.7) :
                                        continue
                                    if left < 0:
                                        left = 0
                                    if top < 0:
                                        top = 0
                                    if right > cropped_image.size[0]:
                                        right = cropped_image.size[0]
                                    if bottom > cropped_image.size[1]:
                                        bottom = cropped_image.size[1]
                                    image1 = cropped_image.crop((left, top, right, bottom))
                                    save_path = os.path.join("output", image_name,
                                                            class_name,  cropped_image_name + str(i) + ".jpg")
                                    image1.save(save_path, "JPEG", quality=100)

