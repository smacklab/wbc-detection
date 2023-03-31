from ultralytics import YOLO
from PIL import Image
import os


model = YOLO("wbc-model-Feb24.pt")

for filename in os.listdir("data"):
    f = os.path.join("data", filename)
    if not os.path.isfile(f) or not f.endswith(".jpg"):
        continue
    image = Image.open(f)

    if not os.path.exists("wbc"):
        os.makedirs("wbc")
    
    results = model(f, save = True)
    for i, result in enumerate(results):
        # conf = result.confidence.cpu().numpy()[0]
        # print("Confidence: {}".format(conf))
        boxes = result.boxes
        xyxy = boxes.xyxy.numpy()
        for find in xyxy:
            if (xyxy[4] < 0.25):
                continue
            left, top, right, bottom = int(find[0]), int(find[1]), int(find[2]), int(find[3])
            width = abs(left - right)
            height = abs(top - bottom)
            if (width/height < 0.7) or (height/width < 0.7) :
                continue
            left -= 20
            top -= 20
            right += 20
            bottom += 20
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            if right > image.size[0]:
                right = image.size[0]
            if bottom > image.size[1]:
                bottom = image.size[1]
            image1 = image.crop((left, top, right, bottom))
            image1.save("wbc/" + filename + "result" + str(i) + ".jpg", 'JPEG', quality=100)
