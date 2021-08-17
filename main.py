from model import Model
from load_data import load_data
import torch
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from train import train
from matplotlib import pyplot as plt

# loader dataset
trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        #transforms.RandomResizedCrop(size=(3, 180, 180), scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        transforms.ToTensor(),
    ])
batchsize = 1
train_loader, valid_loader = load_data('./slice_list.csv', 'slice_layout/', batchsize, trans)
print(len(train_loader.dataset), len(valid_loader.dataset))

for data, target in train_loader:
    # move tensors to GPU if CUDA is available
    print(data.shape)
    break

model = Model()

## load model
# model.load_state_dict(torch.load('savemodel/977.pth'))

if torch.cuda.is_available():
    model.cuda()
    
from torchsummary import summary
summary(model.cuda(), (3, 180, 180))

n_epochs = 1
optimizer1 = torch.optim.Adam(model.parameters(), lr = 1e-5)

criterion = CrossEntropyLoss()
train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model = train(model, n_epochs, train_loader, valid_loader, optimizer1, criterion)

#model1 = torch.load('..../Dogcat_resnet18')