import cv2

from PIL import Image, ImageFont, ImageDraw 

from utils_yolov5.utils_yolov5 import yolov5_charaters_init


def draw_predict(pathImg, predict):
    img = Image.open(pathImg)
    image_editable = ImageDraw.Draw(img)  
    color = (255, 0, 0)
    fontScale = 15
    myfontFont = ImageFont.truetype('static/font/verdanab.ttf', fontScale)
    for item in predict:
        start_point = (item["box"][0],item["box"][1])
        end_point = (item["box"][2],item["box"][3])
        shape = [start_point,end_point]
        image_editable.rectangle(shape, outline ="green")
        image_editable.text(start_point, item['class'],font=myfontFont,fill=color)
    img.show()
    return img




weight = "model/fruits.pt"
model = yolov5_charaters_init(weight_path=weight, device="cpu")
pathImg = r'static\img_to_test\1.jpg'
img = cv2.imread(pathImg) 
result = model.predictJson(img)
predict = result["predictions"]
img_predict = draw_predict(pathImg, predict)
print(result)