from mydataset import MyDataset
import torch
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_path = '', image_path = '', batchize = 8, transform = None):

    train_data = MyDataset(csv_path, image_path, transform)
    print(train_data.__len__())

    train_size = int(0.7 * train_data.__len__())
    valid_size = len(train_data) - train_size
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])
    
    # scaler = MinMaxScaler()
    # train_data = scaler.fit_transform(train_data)
    # valid_data = scaler.transform(valid_data)
    
    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batchize, shuffle=True)

    return train_loader, valid_loader