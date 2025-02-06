from torch import nn,optim
from torchvision import models,datasets, transforms

import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,classification_report

# Define the number of emotion classes (e.g., 7 for emotions like Happy, Sad, Angry, etc.)
num_classes = 7

# Load a pre-trained ResNet model
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Replace the final fully connected layer
# ResNet's original fully connected layer has 512 input features; we need to adapt it for our classes
model.fc = nn.Linear(model.fc.in_features, num_classes)


#transforming image data as for model
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

train_dataset= datasets.ImageFolder(root=r'D:\GUVI\Main\final_project\Emotion detection\data\FER2013\train',transform=transform)
#test_dataset =datasets.ImageFolder(root=r'F:\GUVI\Main course\final_project\Emotion detection\data\FER2013\test',transform=transform)
train_dataloader= DataLoader(train_dataset,shuffle=True,batch_size=64)


# Define device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
epoch= 5
for i in range(epoch):
    model.train()
    for input, label in train_dataloader:
        optimizer.zero_grad()
        output = model(input)
        loss= criterion(output,label)
        loss.backward()
        optimizer.step()
    print(f'{i}/{epoch} loss {loss.item()}')



# finding metrics of train dataset
correct_pred = 0
total =0
total_2=0

all_labels = []
all_preds = []


with torch.no_grad():
    for input,label in train_dataloader:
        output =model(input)
        _,prediction = torch.max(output,1)  
        correct_pred = correct_pred+(prediction==label).sum().item()
        total = total+output.size(0)
        total_2 = total_2+label.size(0)
        #sklearn metrics
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(prediction.cpu().numpy())

print('------train data set metrics------')
print(f"train accuracy: {correct_pred/total} ")# out of total values max probability values ration found to be as accuracy
#print(f'predicted class :{classes[prediction.item()]}')
print(f"train accuracy new: {correct_pred/total_2} ")

#sklearn metrics
accuracy = accuracy_score(all_labels,all_preds)
precision = precision_score(all_labels,all_preds,average='weighted')
recall = recall_score(all_labels,all_preds,average='weighted')
f1score = f1_score(all_labels,all_preds,average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1score:.2f}")
        

