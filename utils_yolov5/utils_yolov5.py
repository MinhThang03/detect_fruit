
import cv2
import torch
import torch.nn as nn


class Yolov5_detection_charaters(nn.Module):
    def __init__(self, weight_path, device="cpu") -> None:
        super().__init__()
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=weight_path, device=device, force_reload=False)
        self.names = self.model.names
        # self.model.agnostic = True
    
    def forward(self, images):
        images = [images[..., ::-1]]
        pred = self.model(images)
        return pred
    
    def predict(self, images):
        images = [images[..., ::-1]]
        pred = self.model(images)
        result = []
        print(type(pred))
        for item in pred.crop(save=False):
            newItem = {}
            newItem['box'] = [int(boxItem) for boxItem in item['box']]
            newItem['conf'] = float(item['conf'])
            newItem['cls'] = int(item['cls'])
            newItem['im'] = item['im']
            result.append(newItem)
        return result
    
    def predictJson(self, images,threshold=0.5):
        # vocab = ["Apple","Avocado","Banana","Eggplant","Grapefruit","Guava","Lemon","Lychee","Mandarine","Mango","Melon","Orange","Papaya","Pear","Pineapple","Pitahaya","Plum","Pomegranate","Potato","Rambutan","Strawberry","Tomato"]
        # vocab = ["Táo", "Bơ", "Chuối", "Cà tím", "Bưởi", "Ổi", "Chanh", "Vải thiều", "Mandarine", "Xoài", "Dưa", "Cam", " Đu đủ "," Lê "," Dứa "," Pitahaya "," Mận "," Lựu "," Khoai tây "," Chôm chôm "," Dâu tây "," Cà chua "]
        vocab = self.names
        images = [images[..., ::-1]]
        pred = self.model(images)
        result = {}
        listItem = []
        for item in pred.crop(save=False):
            newItem = {}
            newItem['box'] = [int(boxItem) for boxItem in item['box']]
            newItem['class'] = vocab[int(item['cls'])]
            newItem['confidence'] = float(item['conf'])
            if newItem['confidence'] > threshold:
                listItem.append(newItem)
        result["predictions"] = listItem
        return result


def yolov5_charaters_init(weight_path, device="cpu"):
    return Yolov5_detection_charaters(weight_path, device=device)


def main():
    weight = "model\Fruits_2.pt"
    model = yolov5_charaters_init(weight_path=weight, device="cpu")
    print(model.names)
    print(len(model.names))
    # img2 = cv2.imread(r'static/img_to_test/0_jpg.rf.d9427c8661a9ee6edf6f14172ad1355e.jpg')  # OpenCV image (BGR to RGB)
    # result = model.predictJson(img2)
    # print(result)


if __name__ == "__main__":
    main()
