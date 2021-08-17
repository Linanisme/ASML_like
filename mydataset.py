from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MyDataset(Dataset):
    def __init__(self, csv_path, image_path, transform = None):
        
        df = pd.read_csv(csv_path, dtype=str)
        len, _ = df.shape
        imgs = []
        for i in range(len):
            img_name = image_path + df['File'][i] + '.png'
            
            img_label = int(df['score'][i])
            if img_label > 0:
                img_label = 1
                
            imgs.append((img_name, img_label))
            self.imgs = imgs
            self.transform = transform


    def __getitem__(self, index):

        fn, label = self.imgs[index]
        img = Image.open(fn)
        # img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__': 
    batchsize = 64
    train_data = MyDataset('./slice_list.csv', 'slice_layout/', None)
    t = train_data[0][0]
    array = np.array(t)
