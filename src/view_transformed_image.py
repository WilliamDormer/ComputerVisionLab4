import torch
import torchvision.transforms as transforms
from PIL import Image

train_transform = transforms.Compose([
        #transforms.ToPILImage(),
        # transorm.RandomAffine
        # transforms.RandomAffine(
        #     degrees=rotation_range,
        #     translate=translation_range,
        #     scale=scale_range
        # ),
        transforms.Resize((224, 224)), # could also use RandomResizedCrop, not sure that's a good idea though
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # what does CenterCrop do?
        # transforms.ToTensor(),
        # # gaussian blur?
        # # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

#load one image from the kitti8 rois to test with
file_path = '../data/Kitti8_ROIs/test/0_1.png'
image = Image.open(file_path).convert("RGB")
image.show()
image = train_transform(image)
image.show()