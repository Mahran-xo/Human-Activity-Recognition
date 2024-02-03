import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import glob
from albumentations.pytorch import ToTensorV2
import config
import albumentations as A

CLASS_NAMES = {'texting': 0,
                                'sitting': 1,
                                'sleeping': 2,
                                'fighting': 3,
                                'running': 4,
                                'calling': 5,
                                'dancing': 6,
                                'using_laptop': 7,
                                'laughing': 8,
                                'cycling': 9,
                                'hugging': 10,
                                'listening_to_music': 11,
                                'clapping': 12,
                                'eating': 13,
                                'drinking': 14}

# Constants and other configurations.


def get_class(val):
   
    for key, value in CLASS_NAMES.items():
        if val == value:
            return key


def annotate_image(image, output_class):
    image = image.cpu()
    image = image.squeeze(0).permute((1, 2, 0)).numpy()
    # image = np.ascontiguousarray(image, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    class_name = get_class(output_class)
    cv2.putText(
        image,
        class_name,
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2, 
        lineType=cv2.LINE_AA
    )
    return image
def inference(model, testloader, DEVICE):
    """
    Function to run inference.
    :param model: The trained model.
    :param testloader: The test data loader.
    :param DEVICE: The computation device.
    """
    model.eval()
    counter = 0
    with torch.no_grad():
        counter += 1
        image = testloader
        image = image.to(DEVICE)

        outputs = model(image.float())

    predictions = F.softmax(outputs, dim=1).cpu().numpy()

    output_class = np.argmax(predictions)

    result = annotate_image(image, output_class)
    return result

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMAGE_RESIZE = 200

    transform = A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            ToTensorV2(),

        ])

    os.makedirs('infrence', exist_ok=True)
    checkpoint = torch.load('my_checkpoint.pth.tar')
    
    # Load the model
    model =  EfficientNet.from_pretrained('efficientnet-b0',in_channels=3,num_classes=15)
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 15) 

    model.load_state_dict(checkpoint['state_dict'])
    all_image_paths = glob.glob(os.path.join('Human Action Recognition', 'test', '*'))

    for i, image_path in enumerate(all_image_paths):
        print(f"Inference on image: {i+1}")
        print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prep = transform(image=image)
        image = prep['image']
        image = torch.unsqueeze(image, 0)
        result = inference(
            model, 
            image,
            DEVICE
        )
        # Save the image to disk.
        image_name = image_path.split(os.path.sep)[-1]
        cv2.imwrite(os.path.join('infrence', image_name), result)