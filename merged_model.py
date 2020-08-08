# Note, newest version of opencv may have issues, please use
# pip install opencv-python==4.1.2.30
# to get opencv
import glob
import os
import random
import timeit

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

torch.manual_seed(1)

# Class Wrapper for YOLO bbox generator and localization
class SignDetector(object):
    def __init__(self, cfg_path, weights_path):
        """Constructor reads a YOLO cfg_path and weights_path

        Parameters
        ----------
        cfg_path : string
            YOLO cfg filepath
        weights_path : string
            YOLO weights filepath
        """
        self.cfg_path = os.path.abspath(cfg_path)
        self.weights_path = os.path.abspath(weights_path)

        self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def __call__(self, *input, **kwargs):
        """Enables forward function to be called as Object(input)

        Returns
        -------
        img : cv2 image
            input image to generate bbox for.
        """
        return self.forward(*input, **kwargs)

    def forward(self, img):
        """Calculates the bbox for a cv2 image

        Parameters
        ----------
        img : cv2 image
            input image

        Returns
        -------
        list of tuples
            list of tuples, each tuple represents a bbox as (x_top_left, y_top_left, width, height)
        """
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        height, width, channels = img.shape

        # Create a blob which acts as input to our model
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Run the Darknet algorithm
        start = timeit.default_timer()
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        stop = timeit.default_timer()
        print("Yolo took {} to run".format(stop-start))

        # Process the output from darknet.
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append((x, y, w, h))
                    confidences.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        return [boxes[x[0]] for x in indexes]

class SignClassifier(nn.Module):
    def __init__(self):
        super(SignClassifier, self).__init__()
        self.name = "signclassifierAdvancedWithDropout"
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096,1000)
        self.fc3 = nn.Linear(1000,75)
        self.dropout1 = nn.Dropout(0.5) # 50% dropout rate
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6) #flatten feature data
        x = F.relu(self.fc1(self.dropout1(x)))
        x = F.relu(self.fc2(self.dropout2(x)))
        x = self.fc3(self.dropout3(x))
        return x

class TrafficSignRecognizer(object):
    def __init__(self, cfg_path, weights_path, classifier_state, index_to_classes, stride=256, size=416, use_cuda=False):
        """Constructor for the model which does bbox detection and classification of traffic signs

        Parameters
        ----------
        cfg_path : string
            path to YOLO cfg file
        weights_path : string
            path to YOLO weights file
        classifier_state : string
            path to classifier torch model save file
        index_to_classes : string
            path to file containing mapping of classifier output index to classname
        use_cuda : bool, optional
            Turn on CUDA for model, currently not working, by default False
        """
        self.stride = stride
        self.size = size

        self.detector = SignDetector(cfg_path, weights_path)
        self.classifier = SignClassifier()

        # Load the pretrained state into our classifier and set to eval mode to ignore dropout
        state = torch.load(classifier_state, map_location=torch.device('cpu'))
        self.classifier.load_state_dict(state)
        self.classifier.eval()

        # alexnet pretrained model for feature extraction
        import torchvision.models
        self.alexnet = torchvision.models.alexnet(pretrained=True)

        # map the output of classifier to a classname
        self.index_to_class = list()
        with open(index_to_classes, "r") as fp:
            for line in fp:
                self.index_to_class.append(line.strip())

        # define the transformations for our classifier
        self.transform = transforms.Compose(
                [transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __get_bboxes(self, img):
        # img = Image.open(image_file)
        bboxes = self.detector(img)
        res = []
        for bbox in bboxes:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = x1 + bbox[2]
            y2 = y1 + bbox[3]
            res.append(torch.tensor((x1, y1, x2, y2)))
        return res

    def __break_up_image(self, img):
        # Calc Image size
        imgwidth, imgheight = img.size
        index_to_bbox = []

        # Crop
        index = 0
        for i in range(0, imgheight, self.stride):
            for j in range(0, imgwidth, self.stride):
                # Out of bounds!
                if j+self.size >= imgwidth or i+self.size >= imgheight:
                    continue
                crop_box = (j, i, j+self.size, i+self.size)
                cropped = img.crop(crop_box)
                index_to_bbox.append((torch.tensor(crop_box), cropped))
                index +=1
        return index_to_bbox

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def forward(self, img, iou_threshold=0.25):
        img = img.convert('RGB')
        # Map the crop index to bbox based on original image that specifies that crop
        index_to_bbox = self.__break_up_image(img)

        yolobbox_tensor = list()
        for index in range(len(index_to_bbox)):
            crop_bbox, crop_img = index_to_bbox[index]
            # Get the bbox from yolo
            yolo_bboxes = self.__get_bboxes(crop_img)

            # Change a relative bbox to absolute bbox
            for yolo_bbox in yolo_bboxes:
                abs_bbox = [yolo_bbox[0]+crop_bbox[0],
                            yolo_bbox[1]+crop_bbox[1],
                            yolo_bbox[2]+crop_bbox[0],
                            yolo_bbox[3]+crop_bbox[1]]
                yolobbox_tensor.append(abs_bbox)
        if not yolobbox_tensor:
            print("No bounding boxes")
            return

        # Convert list of bbox to torch tensor and run nms on the bboxes to merge
        yolobbox_tensor = torch.tensor(yolobbox_tensor, dtype=torch.float)
        kept_yolobbox = torchvision.ops.nms(yolobbox_tensor, torch.tensor([1.0] * yolobbox_tensor.shape[0]), iou_threshold)

        # kept is a list of indexes that correspond to a bbox in bbox_tensor
        # We want to generate a return which is: [(bbox, class, probability)]
        res = list()
        for index in kept_yolobbox:
            bbox = [int(x) for x in yolobbox_tensor[index]]
            cropped = img.crop(bbox)

            # Prepare the cropped yolo bbox for input to our classifier
            cropped = self.transform(cropped).float()
            cropped = cropped.unsqueeze(0)
            features = self.alexnet.features(cropped)
            features = torch.from_numpy(features.detach().numpy())

            # Run the classifier on the cropped yolo bbox
            output = F.softmax(self.classifier(features), dim=1)
            prediction = torch.argmax(output)
            probability = torch.max(output)
            res.append((int(prediction), float(probability), bbox))
        return res

    def get_class_name_from_index(self, index):
        return self.index_to_class[index]

    def annotate_image(self, img, annotation_items):
        img = img.convert('RGB')
        drawing = ImageDraw.Draw(img)

        for prediction, probability, bbox in annotation_items:
            label = self.get_class_name_from_index(prediction)
            drawing.rectangle(bbox, outline=(252, 3, 177), width=2)
            try:
                font = ImageFont.truetype("font.ttf")
            except:
                raise Exception("Current directory missing font.ttf for use in drawing labels on final image.")
            drawing.text(bbox[:2], label, font=font)
        return img


########################################
# Example Run + How to Use

# YOLO config and pretrained weights
cfg_filename = "/Users/carsonliu/workdir/APS360-Project/yolo/yolov3_testing.cfg"
weights_filename = "/Users/carsonliu/workdir/APS360-Project/yolo/yolov3_training.weights"
state_filename = "/Users/carsonliu/workdir/APS360-Project/yolo/model_signclassifierAdvancedWithDropout_bs256_lr3e-05_wd0.0001_dropout50_epoch66"
index_filename = "/Users/carsonliu/workdir/APS360-Project/yolo/index_to_class.txt"
input_image = "/Users/carsonliu/workdir/APS360-Project/yolo/test.png"

start = timeit.default_timer()

# Run the merged model
model = TrafficSignRecognizer(cfg_filename, weights_filename, state_filename, index_filename, stride=416)
img = Image.open(input_image)
output = model(img)

# Save the output annotated file
img = model.annotate_image(img, output)
out_path = os.path.join(os.path.split(input_image)[0], "annotated_" + os.path.split(input_image)[1])
img.save(out_path)

# Debug
print(output)

stop = timeit.default_timer()
print("Took {} to run".format(stop-start))
########################################