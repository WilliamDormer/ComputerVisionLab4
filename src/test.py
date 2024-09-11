'''
This program will:

1. Accept an image path from the user from the Kitti dataset
2. Split it into a set of ROIs. Save the bounding box coordinates for each ROI
3. Build a batch from the ROIs, and pass that through the YODA classifier
4. For each ROI Classified as a 'Car', calculate it's IoU score against the original kitti image.
5. display bounding boxes on the original kitti image
6. Calculate the mean IoU value for all detected 'Car' ROIs, over the complete Kitti test partition.
7. Calculate the confusion matrix for the kitti test partition (needed for report)
'''

import argparse
from KittiDataset import KittiDataset
from KittiAnchors import Anchors
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms
import torch
import torchvision.models as models
from statistics import mean
from PIL import Image

# from KittiToYodaROIs import calc_IoU
def calc_IoU(boxA, boxB):
    # print('break 209: ', boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][1], boxB[0][1])
    yA = max(boxA[0][0], boxB[0][0])
    xB = min(boxA[1][1], boxB[1][1])
    yB = min(boxA[1][0], boxB[1][0])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--image_names', type=str, nargs='+', help="list of image names, separated by spaces")
    parser.add_argument('-ttd', '--test_directory', type=str, default="../data/Kitti8/test/", help="the directory that points to the testing images")
    parser.add_argument('-lf', '--label_file', type=str, default='labels.txt', help="The name of the label file in the image directories (should be the same for train and test)")
    parser.add_argument('-b', '--batch_size', type=int, default=128, help="The size of the batches.")
    parser.add_argument('-md', '--model_directory', type=str, default="../model_parameters/", help="The director to save the model parameters.")
    parser.add_argument('-mn', '--model_name', type=str, required=True, help="The name of the model in the model directory")
    parser.add_argument("-rw", "--resize_width", default=224, type=int)
    parser.add_argument("-rh", "--resize_height", default=224, type=int)
    parser.add_argument('-ft', "--full_test", default="F", type=str, help="Determines whether to use the whole train dataset (Y/N) overwrites other inputs")
    args = parser.parse_args()

    show_images = True
    if args.full_test == "Y":
        show_images = False
        #replace this input with all the test images.
        args.image_names = os.listdir(os.path.join(args.test_directory, "image"))


    print("Starting YODA test script")

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception("CUDA not available! Check that you're using the right version of Pytorch.")

    anchors = Anchors()

    # create dataset from the selected images

    class SmallKittiDataset(Dataset):
        def __init__(self, dir, image_names, transform=None):
            self.dir = dir
            self.image_names = image_names
            self.transform = transform

            self.img_dir = os.path.join(dir, "image")
            self.label_dir = os.path.join(dir, "label")
            self.num=0
            self.img_files = []

            all_image_files = os.listdir(self.img_dir)
            self.img_files = [file for file in all_image_files if file in self.image_names]

            self.max = len(self)
        def __len__(self):
            return len(self.img_files)
        def __getitem__(self, idx):
            filename = os.path.splitext(self.img_files[idx])[0]
            img_path = os.path.join(self.img_dir, self.img_files[idx])
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            label_path = os.path.join(self.label_dir, filename+'.txt')
            labels_string = None

            with open(label_path) as label_file:
                labels_string = label_file.readlines()
            labels = []

            for i in range(len(labels_string)):
                lsplit = labels_string[i].split(' ')
                label = [lsplit[0], int(self.class_label[lsplit[0]]), float(lsplit[4]), float(lsplit[5]),
                         float(lsplit[6]), float(lsplit[7])]
                labels += [label]
            return image, labels

        def __iter__(self):
            self.num = 0
            return self
        def __next__(self):
            if (self.num >= self.max):
                raise StopIteration
            else:
                self.num += 1
                return self.__getitem__(self.num - 1)

        class_label = {'DontCare': 0, 'Misc': 1, 'Car': 2, 'Truck': 3, 'Van': 4, 'Tram': 5, 'Cyclist': 6, 'Pedestrian': 7,
                   'Person_sitting': 8}

        def strip_ROIs(self, class_ID, label_list):
            ROIs = []
            for i in range(len(label_list)):
                ROI = label_list[i]
                if ROI[1] == class_ID:
                    pt1 = (int(ROI[3]), int(ROI[2]))
                    pt2 = (int(ROI[5]), int(ROI[4]))
                    ROIs += [(pt1, pt2)]
            return ROIs

    test_transform = transforms.Compose([
        transforms.Resize((args.resize_width, args.resize_height)), # could also use RandomResizedCrop, not sure that's a good idea though
        # transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # what does CenterCrop do?
        transforms.ToTensor(),
        # gaussian blur?
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class SmallROIDataset(Dataset):
        def __init__(self, ROIs, boxes, transform=None):

            #need to convert the boxes to a sensible tensor type.
            #they come as a python list of tuples looking like ([0,0], [0,0])
            #so make a 3 dimensional tensor, first dimension is the box index, second is the corner index, 3rd is the x/y index.
            self.boxes = torch.zeros((len(boxes), 2, 2))
            for q in range(len(boxes)):
                self.boxes[q][0][0] = boxes[q][0][0]
                self.boxes[q][0][1] = boxes[q][0][1]
                self.boxes[q][1][0] = boxes[q][1][0]
                self.boxes[q][1][1] = boxes[q][1][1]

            self.transform = transform
            # get the ROIs and turn the into images
            rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in ROIs]
            pil_images = [Image.fromarray(img) for img in rgb_images]

            # pil_images[0].show()

            # self.transform = transform
            # rgb_images = self.transform(rgb_images)
            # #desired dimensions for each image CxHxW (where C is R,G,B)
            # #so we keep the first dimension the same (list), then put the Channel first 3, then the height then width
            # self.new_size = (args.resize_width, args.resize_height)
            # resized_images = [cv2.resize(img, self.new_size) for img in rgb_images]
            # torch_images = torch.tensor(resized_images).permute(0,3,1,2)
            # torch_images = torch_images.to(torch.float32)

            self.ROI_images = pil_images

            self.new_size = (args.resize_width, args.resize_height)

            # print("self.boxes len:", len(self.boxes))
            # print("self.roi_images len: ", len(self.ROI_images))

        def __len__(self):
            return len(self.ROI_images)

        def __getitem__(self, item):
            # img_name = os.path.join(self.directory_path, self.image_filenames[item])
            # image = Image.open(img_name).convert("RGB")
            # image = image.resize(self.new_size)

            image = self.ROI_images[item]
            box = self.boxes[item]

            if self.transform:
                # # convert the pillow image to a tensor
                # image = transforms.PILToTensor()(image)
                # # convert the int tensor to float
                # image = image.to(torch.float32)
                # # perform the normalization transform
                image = self.transform(image)

            # label = self.labels[item]

            return image, box #, label

    IoU_threshold = 0.02

    dataset = SmallKittiDataset(args.test_directory, args.image_names)
    print("length of dataset: ", len(dataset))

    #load in model information:
    resnet18 = models.resnet18()

    #TODO add this if it was trained with this modification:

    num_features = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(num_features, 2)

    resnet18.load_state_dict(torch.load(os.path.join(args.model_directory, args.model_name)))
    resnet18.to(device)
    resnet18.eval()

    image_counter = 0
    labels = []
    overall_ious = []

    for item in enumerate(dataset):
        # print("starting item")
        idx = item[0]
        image = item[1][0]
        label = item[1][1]

        # get labels with cars im them:
        label_with_car = [label_example for label_example in label if label_example[1] == 2]
        # print("# car labels for this image: ", len(label_with_car))
        # print(label_with_car)
        # print("label: ", label)

        # get the label number for the cars
        idx = dataset.class_label['Car']
        # get the regions of interest that have cars in them
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)
        # find the anchor centers
        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)

        if show_images:
            image1 = image.copy()
            for j in range(len(anchor_centers)):
                x = anchor_centers[j][1]
                y = anchor_centers[j][0]
                cv2.circle(image1, (x, y), radius=4, color=(255, 0, 255))

        # get the possible regions of interest and boxes for the anchors.
        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)

        # print("length of ROIs: ", len(ROIs))
        # print("length of boxes: ", len(boxes))
        # print(boxes[0])
        # print(type(boxes))
        # print(type(boxes[0]))

        # we're going to find the region of interest IOUs between the anchor and the label
        ROI_IoUs = []
        for idx in range(len(ROIs)):
            ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]


        # for k in range(len(boxes)):
        #     filename = str(i) + '_' + str(k) + '.png'
        #     # if save_ROIs == True:
        #     #     cv2.imwrite(os.path.join(output_dir,filename), ROIs[k])
        #     name_class = 0
        #     name = 'NoCar'
        #     if ROI_IoUs[k] >= IoU_threshold:
        #         name_class = 1
        #         name = 'Car'
        #     labels += [[filename, name_class, name]]

        if show_images: #this one just shows the anchor points with purple dots
            cv2.imshow('image', image1)

        if show_images: # does the calculation to find the boxes that fit
            image2 = image1.copy()

            for k in range(len(boxes)):
                if ROI_IoUs[k] > IoU_threshold:
                    box = boxes[k]
                    pt1 = (box[0][1],box[0][0])
                    pt2 = (box[1][1],box[1][0])
                    cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))

        if show_images: #shows the boxes that for the regions of interest in yellow.
            cv2.imshow('boxes', image2)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break

        # ----------------------------
        # now for the ml model part of it:
        # print("type of ROIs: ", type(ROIs[0]))
        # print("type of boxes: ", type(boxes))
        # I believe boxes contains two points in the form ([x,y],[x,y]) that denotes the corners of the boxes
        # ROIs contains the regions of interest to analyze
        # construct a batch from the ROIs
        roi_dataset = SmallROIDataset(ROIs, boxes, test_transform)
        roi_dataloader = DataLoader(roi_dataset, batch_size=args.batch_size, shuffle=False)

        ious_for_image = []

        predicted_boxes = []
        # print('starting model prediction')

        # print(label)

        with torch.no_grad():
            for i, t_data in enumerate(roi_dataloader):
                t_inputs, boxes = t_data

                # print(boxes)

                # print("len t_inputs: ", len(t_inputs))
                # print("len boxes: ", len(boxes))

                if torch.cuda.is_available():
                    t_inputs = t_inputs.cuda()
                predicted = resnet18(t_inputs)
                _, predicted = torch.max(predicted.data, 1)

                # if predicted is 0, then it predicts no car, do nothing
                # if predicted is 1, then calculate the IOU score against the original Kitti image

                for j in range(len(predicted)):
                    if predicted[j] == 1:
                        predicted_boxes.append(boxes[j])
                        #compute the IoU score against the original Kitti image.

                        # apparently the way to do this is to calculate the IOU against every label in the original image, and then find the one with the highest IOU and take that.
                        max_iou = 0
                        for n in range(len(label_with_car)): #TODO need to fix this to only handle the labels that predict cars obviously lol

                            pt1 = [boxes[j][0][1].item(), boxes[j][0][0].item()]
                            pt2 = [boxes[j][1][1].item(), boxes[j][1][0].item()]
                            boxA = (pt1, pt2)
                            pt3 = [label_with_car[n][2],label_with_car[n][3]]
                            pt4 = [label_with_car[n][4],label_with_car[n][5]]
                            boxB = (pt3, pt4)
                            iou = calc_IoU(boxA, boxB)
                            if iou > max_iou:
                                max_iou = iou

                        ious_for_image.append(max_iou)

        if(len(ious_for_image) > 0):
            print("average ious for image: ", mean(ious_for_image))

        # append ious_for_image to overall_ious
        overall_ious.extend(ious_for_image)

        # print("number of predicted boxes: ", len(predicted_boxes))

        #display the rois that got a car prediction on the image
        if show_images: # does the calculation to find the boxes that fit
            image3 = image2.copy()
            for k in range(len(predicted_boxes)):
                box = predicted_boxes[k]
                box = box.to(torch.int)
                # print(box)
                # print(type(box))
                pt1 = (box[0][1].item(),box[0][0].item())
                pt2 = (box[1][1].item(),box[1][0].item())
                cv2.rectangle(image3, pt1, pt2, color=(255, 0, 0))

        if show_images: #shows the boxes that for the regions of interest in yellow.
            cv2.imshow('predictions', image3)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break

        # ---------------------------

        image_counter = image_counter + 1
        print("image processed: ", image_counter)


        # if max_ROIs > 0 and i >= max_ROIs:
        #     break
    if len(overall_ious) > 0:
        print("mean iou over all images: ", mean(overall_ious))
    print("done")


