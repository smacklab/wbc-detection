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
    
    results = model(image)
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            print(box.conf)
            if box.conf < 0.25:
                continue
            xyxy = box.xyxy.numpy()
            for find in xyxy:
                left, top, right, bottom = int(find[0]), int(find[1]), int(find[2]), int(find[3])
                width = abs(left - right)
                height = abs(top - bottom)
                if (width/height < 0.7) or (height/width < 0.7) :
                    continue
                mid = ((right + left)/2, (bottom + top)/2)
                left = mid[0] - 100
                top = mid[1] - 100
                right = mid[0] + 100
                bottom = mid[1] + 100
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