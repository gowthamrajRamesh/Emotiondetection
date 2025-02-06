#Libraries required 

import torch
from torchvision import models,datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch import nn,optim
from PIL import Image
import torch.nn.functional as F


#
import streamlit as st




#emotion detection using own model 
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

model=SimpleCNN()
model.load_state_dict(torch.load(r'D:\GUVI\Main\final_project\Emotion detection\Emotion detection cnn_model.pth'))
model.eval()

classes= ['angry','disgust','fear','happy','neutral','sad','surprise']

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])


#test_dataset=datasets.ImageFolder(root=r'D:\GUVI\Main\final_project\Emotion detection\data',transform = transform)

#test_dataloader= DataLoader(test_dataset,shuffle=True,batch_size=64)
st.write('# Upload your image')
image = st.file_uploader('''# Upload your image with proper face to find your emotion''',type=['png','jpg'],key='image_input')

if image is not None:
    st.write('uploaded image')
    input_image = Image.open(image)
    st.image(input_image)

#--------------------------------- prediction using own model------------------------------------------
    #image converted into tensors
    img = transform(input_image).unsqueeze(0) 


    correct_pred = 0
    total =0

    with torch.no_grad():
        output = model(img)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _,prediction =torch.max(probabilities,1)
        st.write(':red[_The emotion of the uploaded image is by model create by me_]')
        st.write(classes[prediction])
    # img = img.squeeze(0)

    # img = img/2+0.5
    # img=img.numpy()
    # img = np.transpose(img,(1,2,0)) # matplot linb require data as H,W,C
    # st.write(plt.imshow(img))
    # st.write(plt.title(classes[prediction]))
    # st.write(plt.show())

#----------------------------------emotion prediction using pretrained model------------------------------
    # Define the number of emotion classes (e.g., 7 for emotions like Happy, Sad, Angry, etc.)
    num_classes = 7

    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=True)

    # Replace the final fully connected layer
    # ResNet's original fully connected layer has 512 input features; we need to adapt it for our classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(r'D:\GUVI\Main\final_project\Emotion detection\pretrainedModel_emotion_detection.pth'))
    model.eval()

    classes= ['angry','disgust','fear','happy','neutral','sad','surprise']

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    #image = Image.open(r'F:\GUVI\Main course\final_project\Emotion detection\data\neutral.jpg')
    image = transform(input_image).unsqueeze(0) 

    correct_pred = 0
    total =0

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _,prediction =torch.max(probabilities,1)
        st.write(':red[_The emotion of the uploaded image is by resnet Pretrained model_]')
        st.write(classes[prediction])

