import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch import nn,optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,classification_report

#train_dataset = datasets.FER2013(root=r'F:\GUVI\Main course\final_project\Project2\data',split='train',transform=transforms.ToTensor())
#test_dataset = datsets.FER2013(root=r'F:\GUVI\Main course\final_project\Project2\data',train=False,download=True,transform=transforms.ToTensor())
classes= ['angry','disgust','fear','happy','neutral','sad','surprise']

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

train_dataset= datasets.ImageFolder(root=r'D:\GUVI\Main\final_project\Emotion detection\data\FER2013\train',transform=transform)
test_dataset =datasets.ImageFolder(root=r'D:\GUVI\Main\final_project\Emotion detection\data\FER2013\test',transform=transform)
train_dataloader= DataLoader(train_dataset,shuffle=True,batch_size=64)
test_dataloader= DataLoader(test_dataset,shuffle=True,batch_size=64)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1 =nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=0,stride=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.conv2 =nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=0,stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=3,padding=0,stride=1)
        self.conv3 =nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=3,padding=1,stride=2)
        self.conv4 =nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=0,stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=3,padding=1,stride=2)
        self.fc1= nn.Linear(256*17*17,512)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(512,1024)
        self.fc3 = nn.Linear(1024,7)
        self.dropout=nn.Dropout(0.5)
    def forward(self,x):
        x= self.bn1(self.conv1(x))
        x= self.relu(x)
        x= self.pool1(x)
        x= self.bn2(self.conv2(x))
        x= self.relu(x)
        x= self.pool2(x)
        x= self.pool3(self.relu(self.bn3(self.conv3(x))))
        x= self.pool4(self.relu(self.bn4(self.conv4(x))))
        x=x.view(-1,256*17*17)
        x= self.fc1(x)
        x= self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        # x=F.log_softmax(x,dim=1) # Log probabilities
        return x

# training 
model= SimpleCNN()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(),lr=0.001)
optimizer =optim.SGD(model.parameters(),lr=0.001,weight_decay=1e-4)
epoch= 5
for i in range(epoch):
    model.train()
    for input, label in train_dataloader:
        optimizer.zero_grad()
        output = model(input)
        loss= criterion(output,label)
        loss.backward()

        # Apply gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    print(f'{i}/{epoch} loss {loss.item()}')


#----------------------------------------------------------------------------------------------------------------------------------
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
        


#----------------------------------------------------------------------------------------------------------------------------------------
# finding Metrics of test data set
correct_pred = 0
total =0
total_2=0

all_labels = []
all_preds = []

model.eval()
with torch.no_grad():
    for input,label in test_dataloader:
        output =model(input)
        _,prediction = torch.max(output,1)  
        correct_pred = correct_pred+(prediction==label).sum().item()
        total = total+output.size(0)# Size(0) used to find no of rows in the tensor
        total_2 = total_2+label.size(0)
        #sklearn metrics
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(prediction.cpu().numpy())

print('------test data set metrics------')
print(f"test accuracy: {correct_pred/total} ")# out of total values max probability values ration found to be as accuracy
#print(f'predicted class :{classes[prediction.item()]}')
print(f"test accuracy new: {correct_pred/total_2} ")

#sklearn metrics
accuracy = accuracy_score(all_labels,all_preds)
precision = precision_score(all_labels,all_preds,average='weighted')
recall = recall_score(all_labels,all_preds,average='weighted')
f1score = f1_score(all_labels,all_preds,average='weighted')
 
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1score:.2f}")


# saving model
torch.save(model.state_dict(),r'D:\GUVI\Main\final_project\Emotion detection\Emotion detection cnn_model.pth')