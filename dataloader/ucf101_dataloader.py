import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ucf101_dataset import DataSet


project_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(project_dir,'data')


transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])

train_dataset = DataSet(data_dir,  #r'C:\Users\e0010758\Documents\video classification\frames'
                        data_list_name='data_file_small.csv',
                        train_or_test='train',
                        nb_frames=20,
                        transform=transform)
test_dataset = DataSet(data_dir,  #r'C:\Users\e0010758\Documents\video classification\frames'
                       data_list_name='data_file_small.csv',
                       train_or_test='test',
                       nb_frames=20,
                       transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1000,shuffle=True, num_workers=0)
