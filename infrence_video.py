import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import time
from albumentations.pytorch import ToTensorV2
import config
import albumentations as A
from infrence import get_class

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUT_DIR = 'infrence/video_outputs'
VIDEO_PATH = 'Human Action Recognition/videos/Drunk man fighting for nothing.mp4'


os.makedirs(OUT_DIR, exist_ok=True)
# set the computation device


# Validation transforms
transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(
 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
checkpoint = torch.load('my_checkpoint.pth.tar')

model =  EfficientNet.from_pretrained('efficientnet-b0',in_channels=3,num_classes=15)
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, 15) 
model = model.to(config.DEVICE)

model.load_state_dict(checkpoint['state_dict'])

cap = cv2.VideoCapture(VIDEO_PATH)
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = f"{VIDEO_PATH.split('/')[-1].split('.')[0]}"

out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

frame_count = 0

total_fps = 0 

window_size = 10
rolling_predictions = []


while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        prep = transform(image=rgb_frame)

        tensor_frame = prep['image']

        input_batch = tensor_frame.float().unsqueeze(0) 

        input_batch = input_batch.to(DEVICE)

        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_batch)
            end_time = time.time()
        # get the softmax probabilities
        probabilities = F.softmax(outputs, dim=1).cpu()
        # get the top 1 prediction
        # top1_prob, top1_catid = torch.topk(probabilities, k=1)
        output_class = np.argmax(probabilities)

        rolling_predictions.append(output_class)
        if len(rolling_predictions) > window_size:
            rolling_predictions.pop(0)

        # Calculate rolling average of predictions
        rolling_avg_prediction = int(np.mean(rolling_predictions))

        # get the current fps
        fps = 1 / (end_time - start_time)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1
        cv2.putText(frame, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{get_class(rolling_avg_prediction)}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)
        # cv2.imshow('Result', frame)
        out.write(frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        break
    
# # release VideoCapture()
# cap.release()
# # close all frames and video windows
# cv2.destroyAllWindows()
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")