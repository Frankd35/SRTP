import torchvision
from torchvision import transforms
import torch.utils.data

# image_shape = (3, 400, 400)
dataset = torchvision.datasets.ImageFolder(
    r'C:\Users\Frankd\PycharmProjects\SRTP\Demo\archive(1)\g-images-dataset',
    transform=transforms.Compose([
        transforms.Resize([200, 200]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))


# batch_size = 32
n_classes = 26

def getDataLoader():
    return torch.utils.data.DataLoader(dataset, n_classes, drop_last=True)

