# This trains a classification model that accepts regions of interest, and predicts whether they contain a car or not.

import torchvision.models as models
import torchvision.transforms as transforms
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image, ImageFile
from torchvision.models import ResNet18_Weights
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

from sklearn.metrics import confusion_matrix
import seaborn as sns

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10, help="the number of epochs to train for")
    parser.add_argument('-tnd', '--train_directory', type=str, default="../data/Kitti8_ROIs/train/", help="the directory that points to the training images")
    parser.add_argument('-ttd', '--test_directory', type=str, default="../data/Kitti8_ROIs/test/", help="the directory that points to the testing images")
    parser.add_argument('-lf', '--label_file', type=str, default='labels.txt', help="The name of the label file in the image directories (should be the same for train and test)")
    parser.add_argument('-b', '--batch_size', type=int, default=128, help="The size of the batches.")
    parser.add_argument('-md', '--model_directory', type=str, default="../model_parameters/", help="The director to save the model parameters.")
    parser.add_argument('-en', '--experiment_name', type=str, required=True, help="The name to give the experiment for saving.")
    parser.add_argument("-f", "--figure_directory", default="../figures/")
    parser.add_argument("-rw", "--resize_width", default=224, type=int)
    parser.add_argument("-rh", "--resize_height", default=224, type=int)
    args = parser.parse_args()

    print("beginning training for experiment: ", args.experiment_name)

    # https://pytorch.org/vision/0.9/models.html
    # resnet18 = models.resnet18(pretrained=True, progress=True)
    resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT, progress=True)
    #TODO add a final layer to reduce to 2 output classes
    # Modify the final fully connected layer
    num_features = resnet18.fc.in_features
    print("num features before truncation: ", num_features)
    resnet18.fc = torch.nn.Linear(num_features, 2)

    #TODO optionally disable training of the earlier parts of the network
    # Freeze all the layers except the last FC layer
    print('Printing layers')

    # for name, module in resnet18.named_modules():
    #     print(name)

    # for param in resnet18.parameters():
    #     param.requires_grad = False
    #
    # for param in resnet18.fc.parameters():
    #     param.requires_grad = True

    # # Update the last FC layer to be trainable
    # resnet18.fc.requires_grad = True


    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # # Define the range of transformation parameters
    # rotation_range = (-30, 30)
    # translation_range = (0.1, 0.1)  # Fraction of total height and width
    # scale_range = (0.8, 1.2)  # Range for scaling

    train_transform = transforms.Compose([
        #transforms.ToPILImage(),
        # transorm.RandomAffine
        # transforms.RandomAffine(
        #     degrees=rotation_range,
        #     translate=translation_range,
        #     scale=scale_range
        # ),
        transforms.Resize((args.resize_width, args.resize_height)), # could also use RandomResizedCrop, not sure that's a good idea though
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # what does CenterCrop do?
        transforms.ToTensor(),
        # gaussian blur?
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # get the dataset

    # class ROIDataset(Dataset):
    #     def __init__(self, directory_path, label_file_name, transform=None):
    #         # super().__init__() # do i need this line?
    #         self.directory_path = directory_path
    #         self.label_file_name = label_file_name
    #         self.transform = transform
    #
    #         self.new_size = (args.resize_width, args.resize_height)
    #
    #         self.image_filenames = [filename for filename in os.listdir(self.directory_path) if filename.lower().endswith(".png") and filename != self.label_file_name]
    #
    #         # print("size of image_filenames: ", len(self.image_filenames))
    #
    #         labels_file = os.path.join(self.directory_path, self.label_file_name)
    #         with open(labels_file, 'r') as file:
    #             #break it up into the 3 parts: name, label, text_label. and keep only the label part.
    #             self.labels = [int(line.split()[1]) for line in file]
    #         # print("example label: ", self.labels[0])
    #         # print("num labels: ", len(self.labels))
    #
    #     def __len__(self):
    #         return len(self.image_filenames)
    #
    #     def __getitem__(self, item):
    #         #TODO move the processing here to the init function.
    #         img_name = os.path.join(self.directory_path, self.image_filenames[item])
    #         image = Image.open(img_name).convert("RGB") #maybe this line?
    #         image = self.transform(image)
    #         # image = image.resize(self.new_size)
    #         #
    #         # if self.transform:
    #         #     #convert the pillow image to a tensor
    #         #     image = transforms.PILToTensor()(image) #will return a tensor of shape CxHxW (where C is R,G,B)
    #         #     # convert the int tensor to float
    #         #     image = image.to(torch.float32)
    #         #     # perform the normalization transform
    #         #     image = self.transform(image)
    #
    #         label = self.labels[item]
    #
    #         return image, label

    class custom_dataset(Dataset):
        def __init__(self, label_file, root_dir, transform=None):
            super().__init__()
            Image.MAX_IMAGE_PIXELS = None
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            csv_file_location = os.path.join(root_dir, label_file)
            self.labels_frame = pd.read_csv(csv_file_location, header=None, sep=' ')
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.labels_frame)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_name = os.path.join(self.root_dir,
                                    self.labels_frame.iloc[idx, 0])
            image = Image.open(img_name)
            labels = self.labels_frame.iloc[idx, 1]

            if self.transform:
                image = self.transform(image)

            # sample = {'image': image, 'labels': labels}

            return image, labels

    # training dataset
    # test_dataset = ROIDataset(args.test_directory, args.label_file, train_transform)
    # train_dataset = ROIDataset(args.train_directory, args.label_file, train_transform)

    test_dataset = custom_dataset(root_dir=args.test_directory, label_file=args.label_file, transform=train_transform)
    train_dataset = custom_dataset(root_dir=args.train_directory, label_file=args.label_file, transform=train_transform)

    # split the test dataset into test and validation partitions.
    testval_generator = torch.Generator().manual_seed(42)
    testval_datasets = torch.utils.data.random_split(test_dataset, [0.5,0.5], generator=testval_generator)

    test_dataset = testval_datasets[0]
    val_dataset = testval_datasets[1]

    print("size of train dataset: ", len(train_dataset))

    # #some code to reduce the size of the training set for testing purposes
    # percentage = 0.1
    # num_samples = int(len(train_dataset) * percentage)
    # print("num samples: ", num_samples)
    # print("other: ", len(train_dataset) - num_samples)
    # train_dataset, _ = random_split(train_dataset, [num_samples, len(train_dataset) - num_samples])
    # print("new size of train dataset: ", len(train_dataset))
    # # ---------------

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("size of train loader (#batches per epoch): ", len(train_loader))

    # optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(resnet18.parameters(), lr=1e-3, weight_decay=0.00001)
    optimizer = optim.Adam(resnet18.parameters(), lr=1e-4, weight_decay=0.00001)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss()

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception("CUDA not available! Check that you're using the right version of Pytorch.")

    resnet18.to(device)

    # set it into train mode:
    resnet18.train()

    def train_epoch():
        resnet18.train()
        running_loss = 0.

        for i, data in enumerate(train_loader):
            inputs, labels = data

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # Zero gradients
            optimizer.zero_grad()

            # compute the output of the network
            output = resnet18(inputs)

            #calculate the loss
            loss = loss_fn(output, labels)

            # back propagate
            loss.backward()

            # adjust learning weights
            optimizer.step()

            # Gather data and report
            avg_batch_loss = loss / len(inputs)
            if i % 100 == 0:
                print(f"batch {i} loss {avg_batch_loss}")

            running_loss += avg_batch_loss

        # calculate the average loss for the epoch:
        avg_loss = running_loss / len(train_loader)

        return avg_loss

    loss_record = torch.empty((args.epochs, 1), requires_grad=False)
    v_loss_record = torch.empty((args.epochs,1), requires_grad=False)

    best_vloss = 1000000.
    best_model_path = None #TODO do I still need this?

    for epoch in tqdm(range(args.epochs)):

        # train one epoch
        avg_loss = train_epoch()

        #run validation
        valid_loss = 0
        resnet18.eval()

        correct=0
        total=0
        with torch.no_grad():
            for i, v_data in enumerate(val_loader):
                v_inputs, v_labels = v_data
                if torch.cuda.is_available():
                    v_inputs, v_labels = v_inputs.cuda(), v_labels.cuda()

                outputs = resnet18(v_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += v_labels.size(0)
                correct += (predicted == v_labels).sum().item()
                v_loss = loss_fn(outputs, v_labels)
                avg_batch_v_loss = v_loss.item() / len(v_inputs)
                valid_loss += avg_batch_v_loss

            # divide the sum of the average valid loss over all batches, and divide by the number of batches.
            avg_vloss = valid_loss / len(val_loader)

        accuracy = correct/total

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print("validation accuracy: ", accuracy)

        # Log the running loss averaged per batch
        # for both training and validation
        loss_record[epoch] = avg_loss
        v_loss_record[epoch] = avg_vloss

        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = args.model_directory + args.experiment_name + "_" + str(epoch + 1) + ".pth"
            best_model_path = model_path
            torch.save(resnet18.state_dict(), model_path)
        elif epoch == args.epochs - 1:
            # save it anyways
            model_path = args.model_directory + args.experiment_name + "_final_" + str(epoch + 1) + ".pth"
            torch.save(resnet18.state_dict(), model_path)

        # update scheduler
        scheduler.step()

    # run the test dataset to measure accuracy.
    # load the best model version:
    resnet18.load_state_dict(torch.load(best_model_path))
    resnet18.eval()
    total_acc = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i, t_data in enumerate(test_loader):
            t_inputs, t_labels = t_data
            if torch.cuda.is_available():
                t_inputs, t_labels = t_inputs.cuda(), t_labels.cuda()
            predicted = resnet18(t_inputs)
            # print(predicted.size())
            _, predicted = torch.max(predicted.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(t_labels.cpu().numpy())

            # print(predicted.size())
            # print(t_labels.size())
            acc = (predicted == t_labels).sum().item()
            total_acc += acc / len(t_inputs)
    total_acc = total_acc / len(test_loader)
    print("total accuracy on test set: ", total_acc)

    conf_matrix = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Car", "Car"], yticklabels=["No Car", "Car"])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(args.figure_directory + args.experiment_name + "_confusion_matrix" + ".jpg")

    loss_record = loss_record.cpu().detach().numpy()
    v_loss_record = v_loss_record.cpu().detach().numpy()

    # Generate and save the loss plots
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(loss_record) + 1))
    plt.plot(epochs, loss_record, label="Training Loss", marker='o')
    plt.plot(epochs, v_loss_record, label="Validation Loss", marker='o')

    plt.title('Loss Over Epochs')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.savefig(args.figure_directory + args.experiment_name + ".jpg")








