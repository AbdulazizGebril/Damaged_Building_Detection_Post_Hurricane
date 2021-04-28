import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import itertools
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from PIL import Image
import seaborn as sns
os.system("sudo wget https://storage.googleapis.com/satellite_images_post_harvey/Satellite_images.zip")
os.system("sudo unzip Satellite_images.zip")

def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

#-----------------------------------------------Images Examples--------------------------------------------------------
x = Image.open( 'train_another/'+ 'damage/-93.6141_30.754263.jpeg')
np.array(x).shape
imshow(x)

transform= transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_transform = transforms.Compose([transforms.Resize((128, 128)),
                                   transforms.RandomHorizontalFlip(),
                                   #transforms.RandomRotation(90),
                                   transforms.ColorJitter(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])


# Load train data
training_data = datasets.ImageFolder(root='train_another/', transform=train_transform)
train_loader = DataLoader(training_data, batch_size=200, shuffle=True)
x_train, y_train = iter(train_loader).next()

# Load validation data
validation_data = datasets.ImageFolder(root='validation_another/', transform=transform)
val_loader = DataLoader(validation_data, batch_size=200, shuffle=True)
x_val, y_val = iter(val_loader).next()

# Load test data
unbalanced_test_data = datasets.ImageFolder(root='test_another/', transform=transform)
unbalanced_test_loader = DataLoader(unbalanced_test_data, batch_size = 50, shuffle=False)

balanced_test_data = datasets.ImageFolder(root='test/', transform=transform)
balanced_test_loader = DataLoader(balanced_test_data, batch_size = 50, shuffle=False)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

x_train=x_train.to("cuda")
y_train=y_train.to("cuda")

x_val=x_val.to("cuda")
y_val=y_val.to("cuda")


LR = 0.01
N_EPOCHS = 100
BATCH_SIZE = 10
DROPOUT = 0.5


# %% ---------------------------------------------- Helper Functions ---------------------------------------------------
def acc(x, y, return_labels = False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


# %% -------------------------------------------------- CNN Class-------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 1, padding = 1)
        self.convnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1)
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.convnorm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d((2, 2))

        self.dropout = nn.Dropout(DROPOUT)
        self.linear1 = nn.Linear(64 * 16 * 16, 32)
        self.linear1_bn = nn.BatchNorm1d(32)
        self.linear2 = nn.Linear(32, 2)
        self.linear2_bn = nn.BatchNorm1d(2)
        self.sigmoid = torch.sigmoid
        self.relu = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.relu(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.relu(self.conv2(x))))
        x = self.pool3(self.convnorm3(self.relu(self.conv3(x))))
        # print(x.shape)
        x = self.dropout(self.linear1_bn(self.relu(self.linear1(x.view(-1, 64 * 16 * 16)))))
        x = self.dropout(self.linear2_bn(self.relu(self.linear2(x))))
        x = self.sigmoid(x)
        return x


# %% ------------------------------------------------- Training Prep ---------------------------------------------------
model = CNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
criterion = nn.CrossEntropyLoss()

def acc(x, y, return_labels = False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis = 1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


# %% -------------------------------------------------- Training Loop --------------------------------------------------
print("Training........")
history_li = []
for epoch in range(N_EPOCHS):

    # keep track of training and validation loss each epoch
    train_loss = 0.0
    val_loss = 0.0

    train_acc = 0
    val_acc = 0

    # Set to training
    model.train()


    loss_train = 0
    model.train()

    for batch in range(len(x_train)//BATCH_SIZE):

        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

        # Track train loss
        train_loss += loss.item()
        train_acc = acc(x_train, y_train)

    model.eval()

    with torch.no_grad():
        y_val_pred = model(x_val)
        loss = criterion(y_val_pred, y_val)
        val_loss = loss.item()
        val_acc = acc(x_val, y_val)
        loss_test = loss.item()

        history_li.append([train_loss/BATCH_SIZE, val_loss, train_acc, val_acc])
        torch.save(model.state_dict(), 'model_gebril_3.pt')
        torch.cuda.empty_cache()
    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Validation Loss {:.5f}, Validation Acc {:.2f}".format(
        epoch, loss_train/BATCH_SIZE, acc(x_train, y_train), val_loss, acc(x_val, y_val)))

    history = pd.DataFrame(history_li, columns=['train_loss', 'val_loss', 'train_acc', 'val_acc'])


# ---------------------------------------------------Visualization-----------------------------
history.to_csv("gebril_result.csv")


plt.figure()
train_acc = history['train_acc']
val_acc = history['val_acc']
epoch = range(0, len(train_acc), 1)
plot1, = plt.plot(epoch, train_acc, linestyle = "solid", color = "skyblue")
plot2, = plt.plot(epoch, val_acc, linestyle = "dashed", color = "orange")
plt.legend([plot1, plot2], ['Training', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy per Batch')
plt.title(' Gebril (Custom) Training and Validation Accuracy', pad = 20)
plt.savefig('Gebril_Accuracy.png')
plt.show()


plt.figure()
train_loss = history['train_loss']
val_loss = history['val_loss']
epoch = range(0, len(train_loss), 1)
plot1, = plt.plot(epoch, train_loss, linestyle = "solid", color = "skyblue")
plot2, = plt.plot(epoch, val_loss, linestyle = "dashed", color = "orange")
plt.legend([plot1, plot2], ['Training', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss per Batch')
plt.title('Gebril (Custom) Training and Validation Loss', pad = 20)
plt.savefig('Gebril_loss.png')
plt.show()

#----------------------------------------------------Prediction----------------------------------


def predict(mymodel, model_name_pt, loader):

    model = mymodel
    model.load_state_dict(torch.load(model_name_pt))
    model.to(device)
    model.eval()
    y_actual_np = []
    y_pred_np = []
    for idx, data in enumerate(loader):
        test_x, test_label = data[0], data[1]
        test_x = test_x.to(device)
        y_actual_np.extend(test_label.cpu().numpy().tolist())

        with torch.no_grad():
            y_pred_logits = model(test_x)
            pred_labels = np.argmax(y_pred_logits.cpu().numpy(), axis=1)
            y_pred_np.extend(pred_labels.tolist())

    return y_actual_np, y_pred_np

#----------------------------------------Unbalanced Dataset---------------------------------------------------------

y_actual, y_predict = predict(model, "model_gebril_3.pt", unbalanced_test_loader)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#----------------------------------------------Unbalanced Dataset--------------------------------------------

acc_rate = 100*accuracy_score(y_actual, y_predict)
print("The Accuracy rate for the model is: ", acc_rate)
confusion_mtx= confusion_matrix(y_actual, y_predict)
print("F1-score:", f1_score(y_actual, y_predict, average='macro') )
plot_confusion_matrix(confusion_mtx,
            classes = ["Damaged", "No Damage"])

#------------------------------Balanced Dataset---------------------------------------------------
y_actual, y_predict = predict(model, "model_gebril_3.pt", balanced_test_loader)

acc_rate = 100*accuracy_score(y_actual, y_predict)
print("The Accuracy rate for the model is: ", acc_rate)
confusion_mtx= confusion_matrix(y_actual, y_predict)
print("F1-score:", f1_score(y_actual, y_predict, average='macro') )
plot_confusion_matrix(confusion_mtx,
            classes = ["Damaged", "No Damage"])