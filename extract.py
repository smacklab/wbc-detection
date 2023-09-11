# extract.py - Extracts white blood cells from images using YOLOv8
# Set input directory to the directory containing the images to be processed
# Set output directory to the directory where the extracted images will be saved
# run WhiteBloodCellDetector extract() method to extract the white blood cells
#
# Example:
# detector = WhiteBloodCellDetector("wbc-model-Feb24.pt")
# detector.set_input_directory("data")
# detector.set_output_directory("wbc")
# detector.extract()

from ultralytics import YOLO
from PIL import Image
import os
from tqdm import tqdm


class WhiteBloodCellDetector:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        self.model = YOLO(model_path)
        self.input_directory = ""
        self.output_directory = ""

    def set_input_directory(self, input_directory):
        self.input_directory = input_directory
        print(f"Input directory set to '{input_directory}'")

    def set_output_directory(self, output_directory):
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            print(f"Creating new output directory '{output_directory}'")
            os.makedirs(self.output_directory)
        else:
            print(f"Output directory set to '{output_directory}'")

    def _process_image(self, image_path, filename):
        image = Image.open(image_path)
        results = self.model(image, device=self.device)
        for i, result in enumerate(results):
            for box in result.boxes:
                if box.conf < 0.25:
                    continue
                if not self.isCPU:
                    box = box.cpu()
                xyxy = box.xyxy.numpy()
                for find in xyxy:
                    left, top, right, bottom = map(int, find)
                    width = abs(left - right)
                    height = abs(top - bottom)
                    if (width / height < 0.7) or (height / width < 0.7):
                        continue
                    mid = ((right + left) / 2, (bottom + top) / 2)
                    left = max(mid[0] - 100, 0)
                    top = max(mid[1] - 100, 0)
                    right = min(mid[0] + 100, image.size[0])
                    bottom = min(mid[1] + 100, image.size[1])
                    image1 = image.crop((left, top, right, bottom))
                    # string format box.conf to 2 decimal places
                    confidence = "{:.2f}".format(box.conf.item())
                    image1.save(os.path.join(self.output_directory, f"{confidence}_{filename}_result{i}.jpg"), 'JPEG', quality=100)

    def extract(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        image_files = [f for f in os.listdir(self.input_directory) if f.endswith(".jpg")]
        image_count = len(image_files)

        if image_count == 0:
            print(f"No .jpg images found in the input directory: {self.input_directory}")
            return

        print(f"Input directory set to '{self.input_directory}' with {image_count} images.")

        # loop through all images in the input directory
        for filename in tqdm(image_files, desc="Processing Images"):
            # print("Processing image:", filename)
            image_path = os.path.join(self.input_directory, filename)
            self._process_image(image_path, filename)


if __name__ == "__main__":
    detector = WhiteBloodCellDetector("wbc-model-Feb24.pt", device="cpu")
    detector.set_input_directory("data")
    detector.set_output_directory("wbc")
    detector.extract()
