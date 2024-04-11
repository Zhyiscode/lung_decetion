import os
import numpy as np
import random
import shutil
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
from torchvision import transforms, datasets
data_dir = "D:\\pycharmyunxing\\venv\\pytorch练习\\肺部的病理检测\\data\\COVID-19_Radiography_Dataset\\COVID-19_Radiography_Dataset"
class_names = os.listdir(data_dir)
#print(class_names)

image_files = [[os.path.join(data_dir, class_name, x)
               for x in os.listdir(os.path.join(data_dir, class_name))]
               for class_name in class_names]
len(image_files)
print(len(image_files))

image_file_list = []
image_label_list = []

for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files[i])
    image_label_list.extend([i] * len(image_files[i]))
num_total = len(image_label_list)
print(num_total)

image_label_names = []
for i in range(len(image_label_list)):
    if image_label_list[i] == 0:
        image_label_names.append(class_names[0])
    elif image_label_list[i] == 1:
        image_label_names.append(class_names[1])
    elif image_label_list[i] == 2:
        image_label_names.append(class_names[2])
    elif image_label_list[i] == 3:
        image_label_names.append(class_names[3])
print(image_label_names)

images = []
for i in range(len(image_file_list)):
    im = Image.open(image_file_list[i])
#     im = np.array(im)
    images.append(im)
print(images)

sx = sns.countplot(image_label_names)
sx.set_xticklabels(labels=sx.get_xticklabels(), rotation=90)
plt.show()

# for i in range(4):
#     print(class_names[i], len(image_files[i]))

plt.figure(figsize=(16, 12))
for i in range(len(image_files)):
    for j in range(4):
        image = plt.imread(image_files[i][random.randint(0, len(image_files[i]))])
        plt.subplot(4, 4, i*4+j+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title(class_names[i], fontdict={'fontsize':8}, pad=1.5)


training_images, training_labels = [], []
validation_images, validation_labels = [], []
test_images, test_labels = [], []

for i in range(len(image_file_list)):
    rand = np.random.random()
    if rand < 0.1:
        validation_images.append(images[i])
        validation_labels.append(image_label_list[i])
    elif rand < 0.2:
        test_images.append(images[i])
        test_labels.append(image_label_list[i])
    else:
        training_images.append(images[i])
        training_labels.append(image_label_list[i])

print(len(training_images), len(validation_images), len(test_images))

sx = sns.countplot(training_labels)
for container in sx.containers:
    sx.bar_label(container)

sx = sns.countplot(validation_labels)
for container in sx.containers:
    sx.bar_label(container)

sx = sns.countplot(test_labels)
for container in sx.containers:
    sx.bar_label(container)

training_images = np.array(training_images)
training_labels = np.array(training_labels)

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

imgs = datasets.ImageFolder(data_dir, transform=transform)

next(iter(imgs))

training_images, training_labels = [], []
validation_images, validation_labels = [], []
test_images, test_labels = [], []

for i in range(len(image_file_list)):
    rand = np.random.random()
    if rand < 0.1:
        validation_images.append(next(iter(imgs)))
        validation_labels.append(image_label_list[i])
    elif rand < 0.2:
        test_images.append(next(iter(imgs)))
        test_labels.append(image_label_list[i])
    else:
        training_images.append(next(iter(imgs)))
        training_labels.append(image_label_list[i])

print(len(training_images), len(validation_images), len(test_images))

# training_images = torch.Tensor(training_images)
training_images[-100]

os.mkdir('dataset')
os.mkdir(os.path.join('dataset', 'train'))
os.mkdir(os.path.join('dataset', 'val'))
os.mkdir(os.path.join('dataset', 'test'))
for i in range(len(class_names)):
    os.mkdir(os.path.join('dataset', 'train', class_names[i]))
    os.mkdir(os.path.join('dataset', 'val', class_names[i]))
    os.mkdir(os.path.join('dataset', 'test', class_names[i]))
    for j in range(len(image_files[i])):
        rand = np.random.random()
        if rand < 0.1:
            shutil.copyfile(os.path.join(image_files[i][j]), os.path.join('dataset/val', class_names[i], f'{j}.png'))
        elif rand < 0.2:
            shutil.copyfile(os.path.join(image_files[i][j]), os.path.join('dataset/test', class_names[i], f'{j}.png'))
        else:
            shutil.copyfile(os.path.join(image_files[i][j]), os.path.join('dataset/train', class_names[i], f'{j}.png'))

train = datasets.ImageFolder(os.path.join('dataset', 'train'), transform=transform)
val = datasets.ImageFolder(os.path.join('dataset', 'val'), transform=transform)
test = datasets.ImageFolder(os.path.join('dataset', 'test'), transform=transform)

len(train), len(val), len(test)

train_loader = torch.utils.data.DataLoader(train, batch_size=32)
val_loader = torch.utils.data.DataLoader(val, batch_size=32)
test_loader = torch.utils.data.DataLoader(test, batch_size=32)

len(train_loader), len(val_loader), len(test_loader)

resnet = torchvision.models.resnet18(pretrained=True)
print(resnet)

resnet.fc = torch.nn.Linear(in_features = 512, out_features = 4)
loss_fn     = torch.nn.CrossEntropyLoss()
optimizer   = torch.optim.Adam(resnet.parameters(), lr = 3e-5)


def training(epochs=1):
    for i in range( epochs ):

        train_loss = 0
        val_loss = 0

        resnet.train()
        for train_step, (images, labels) in enumerate( train_loader ):
            optimizer.zero_grad()
            outputs = resnet( images )
            loss = loss_fn( outputs, labels )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if train_step % 20 == 0:
                print( "Evaluation at step", train_step )

                accuracy = 0
                resnet.eval()

                for val_step, (images, labels) in enumerate( test_loader ):
                    outputs = resnet( images )
                    loss = loss_fn( outputs, labels )
                    val_loss += loss.item()

                    _, preds = torch.max( outputs, 1 )
                    accuracy += sum( (preds == labels).numpy() )

                val_loss /= (val_step + 1)
                accuracy = accuracy / len( val )
                print( f"Validatoin Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}" )

                resnet.train()

                if accuracy >= 0.95:
                    print( "performance condition satisfied" )
                    return

        train_loss /= (train_step + 1)
        print( f"Training loss: {train_loss:.4f}" )
    print( "Training complete" )

