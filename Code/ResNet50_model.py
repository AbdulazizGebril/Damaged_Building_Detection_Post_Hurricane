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

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.RandomHorizontalFlip(),
                                   #transforms.RandomRotation(20),
                                   transforms.ColorJitter(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])



# Load train data
training_data = datasets.ImageFolder(root='train_another/', transform=train_transform)
train_loader = DataLoader(training_data, batch_size=300, shuffle=True)
x_train, y_train = iter(train_loader).next()

# Load validation datan
validation_data = datasets.ImageFolder(root='validation_another/', transform=transform)
val_loader = DataLoader(validation_data, batch_size=300, shuffle=True)
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

#----------------------------------------VGG model--------------------------------------
#model = models.vgg16(pretrained=True)
#print("VGG16 model structure:")
#print(model)

# Freeze early layers
#for param in model.parameters():
    #param.requires_grad = False

#n_inputs = model.classifier[6].in_features
#n_classes= 2
# Add on classifier
#model.classifier[6] = nn.Sequential(
    #nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
    #nn.Linear(256, n_classes))

#print(model.classifier)

#total_params = sum(p.numel() for p in model.parameters())
#print(f'{total_params:,} total parameters.')
#total_trainable_params = sum(
    #p.numel() for p in model.parameters() if p.requires_grad)
#print(f'{total_trainable_params:,} training parameters.')

#model = model.to("cuda")

def ResNet50_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features
        n_classes = 2

        # Add on classifier
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, n_classes))


    # Move to GPU
    MODEL = model.to("cuda")

    return MODEL

model_resnet50 = ResNet50_model('resnet50')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_resnet50.parameters(), lr=0.00002, momentum=0.9)

def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model_resnet50(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


def train(model, xtrain, ytrain, xval, yval, save_file_name, n_epochs, BATCH_SIZE):

    history1 = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Training.........\n')

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        val_loss = 0.0

        train_acc = 0
        val_acc = 0

        # Set to training
        model.train()

        #Training loop
        for batch in range(len(xtrain)//BATCH_SIZE):
            idx = slice(batch * BATCH_SIZE, (batch+1)*BATCH_SIZE)

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs
            output = model(xtrain[idx])
            # Loss and BP of gradients
            loss = criterion(output, ytrain[idx])
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Track train loss
            train_loss += loss.item()
            train_acc = acc(xtrain, ytrain)

        # After training loops ends, start validation
        # set to evaluation mode
        model.eval()
        # Don't need to keep track of gradients
        with torch.no_grad():
            # Evaluation loop
            # F.P.
            y_val_pred = model(xval)
            # Validation loss
            loss = criterion(y_val_pred, yval)
            val_loss = loss.item()
            val_acc = acc(xval, yval)

            history1.append([train_loss / BATCH_SIZE, val_loss, train_acc, val_acc])
            torch.save(model.state_dict(), save_file_name)
            torch.cuda.empty_cache()

            # Print training and validation results
        print("Epoch {} | Train Loss: {:.5f} | Train Acc: {:.2f} | Valid Loss: {:.5f} | Valid Acc: {:.2f} |".format(
            epoch, train_loss / BATCH_SIZE, acc(xtrain, ytrain), val_loss, acc(xval, yval)))
        # Format history
        history = pd.DataFrame(history1, columns=['train_loss', 'val_loss', 'train_acc', 'val_acc'])
    return model, history

N_EPOCHS = 50

model, history = train(model_resnet50,
                       x_train,
                       y_train,
                       x_val,
                       y_val,
                       save_file_name = 'model_resnet50_2.pt',
                       n_epochs = N_EPOCHS,
                       BATCH_SIZE = 10)

#-----------------------------------Visualisation---------------------------
plt.figure()
train_acc = history['train_acc']
val_acc = history['val_acc']
epoch = range(0, len(train_acc), 1)
plot1, = plt.plot(epoch, train_acc, linestyle = "solid", color = "skyblue")
plot2, = plt.plot(epoch, val_acc, linestyle = "dashed", color = "orange")
plt.legend([plot1, plot2], ['Training', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy per Batch')
plt.title('ResNet50 Training and Validation Accuracy', pad = 20)
plt.savefig('ResNet50_Accuracy.png')
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
plt.title('ResNet50 Training and Validation Loss', pad = 20)
plt.savefig('ResNet50_loss.png')
plt.show()

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
            #print("Predicting ---->", pred_labels)
            y_pred_np.extend(pred_labels.tolist())

    return y_actual_np, y_pred_np

y_actual, y_predict = predict(model, "model_resnet50_2.pt", unbalanced_test_loader)


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
#-----------------------------------------Unbalanced dataset------------------------------------------------------
acc_rate = 100*accuracy_score(y_actual, y_predict)
print("The Accuracy rate for the model is: ", acc_rate)
confusion_mtx= confusion_matrix(y_actual, y_predict)
print("F1-score:", f1_score(y_actual, y_predict, average='macro') )
plot_confusion_matrix(confusion_mtx,
            classes = ["Damaged", "No Damage"])

#------------------------------Balanced Dataset---------------------------------------------------
y_actual, y_predict = predict(model, "model_resnet50_2.pt", balanced_test_loader)

acc_rate = 100*accuracy_score(y_actual, y_predict)
print("The Accuracy rate for the model is: ", acc_rate)
confusion_mtx= confusion_matrix(y_actual, y_predict)
print("F1-score:", f1_score(y_actual, y_predict, average='macro') )
plot_confusion_matrix(confusion_mtx,
            classes = ["Damaged", "No Damage"])