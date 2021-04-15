import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from gan import Generator

file_path = r"./gan_models/generator.t7"

batch_size = 1
latent_dim = 100
n_classes = 26

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sample noise and labels as generator input
z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))).to(device))
gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)).to(device))

# generator = Generator()
generator = torch.load(file_path)
generator.eval()

img = generator(z, gen_labels)
save_image(img.data, "images/%d.png", nrow=10, normalize=True)
