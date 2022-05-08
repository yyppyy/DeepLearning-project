#import necessary modules
from binhex import REASONABLY_LARGE
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch import embedding, optim as optim
# for visualization
from matplotlib import pyplot as plt
import math
import numpy as np

from processData import get_data_loader

# loading the dataset
training_parameters = {
    "noise_size": 512,
    "img_size": 512,
    "n_epochs": 24,
    "batch_size": 32,
    "learning_rate_generator": 0.0002,
    "learning_rate_discriminator": 0.0002,
}

# tells PyTorch to use an NVIDIA GPU, if one is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_data_loader()

# Create the Generator model class, which will be used to initialize the generator
class Generator(nn.Module):
  def __init__(self, input_dim, output_dim): # to initialize the model-wide parameters. When you run `generator = Generator(params)`, those "params" are passed to __init__.
    super(Generator,self).__init__() # initialize the parent class
    # TODO (5.4) Turn this Generator into a Conditional Generator by
    # 1. Adjusting the input dimension of the first hidden layer.
    # 2. Modifying the input to the first hidden layer in the forward class.
    # self.label_embedding = nn.Embedding(10, 10) # This function will be useful.
    self.hidden_layer1 = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LeakyReLU(0.2)
    )
    self.hidden_layer2 = nn.Sequential(
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2)
    )
    self.hidden_layer3 = nn.Sequential(
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.2)
    )
    self.hidden_layer4 = nn.Sequential(
        nn.Linear(1024, output_dim),
        nn.Tanh()
    )

  def forward(self, x):
      output = self.hidden_layer1(x)
      output = self.hidden_layer2(output)
      output = self.hidden_layer3(output)
      output = self.hidden_layer4(output)
      return output.to(device)

class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Discriminator, self).__init__()
        # TODO (5.4) Modify this discriminator to function as a conditional discriminator.
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x): # labels to be used in 5.4.
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)
        return output.to(device)


discriminator = Discriminator(3 * training_parameters["img_size"] ** 2,1).to(device) # initialize both models, and load them to the GPU or CPU.
generator = Generator(training_parameters["noise_size"], 3 * training_parameters["img_size"] ** 2).to(device)

discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=training_parameters['learning_rate_discriminator'])
generator_optimizer = optim.Adam(generator.parameters(), lr=training_parameters['learning_rate_generator'])

# TODO: Implement the GAN training procedure.
lossFunc = nn.BCELoss()
def train_generator(batch_size):
    """
    Performs a training step on the generator by
        1. Generating fake images from random noise.
        2. Running the discriminator on the fake images.
        3. Computing loss on the result.
    :arg batch_size: the number of training examples in the current batch
    Returns the average generator loss over the batch.
    """
    # TODO: This function should perform a single training step on the generator
    # Start by zeroing the gradients of the optimizer
    generator_optimizer.zero_grad()
    # 1. Create a new batch of fake images (since the discriminator has just been trained on the old ones)
    noise = torch.randn(batch_size,training_parameters["noise_size"]).to(device) # whenever you create new variables for the model to process, send them to the device, like this.
    # ...
    # generator.train()
    # discriminator.eval()
    generated = generator(noise)
    pred = discriminator(generated)
    loss = lossFunc(pred, torch.ones(batch_size, 1))
    loss.backward()
    generator_optimizer.step()
    return loss

def train_discriminator(batch_size, images): # labels to be used in 5.4.
    """
    Performs a training step on the discriminator by
        1. Generating fake images from random noise.
        2. Running the discriminator on the fake images.
        3. Running the discriminator on the real images
        3. Computing loss on the results.
    :arg batch_size: the number of training examples in the current batch
    :arg images: the current batch of images, a tensor of size BATCH x 1 x 64 x 64
    :arg labels: the labels corresponding to images, a tensor of size BATCH
    Returns the average loss over the batch.
    """
    # TODO: And this function should perform a single training step on the discriminator
    # generator.eval()
    # discriminator.train()
    print(images.size())
    discriminator_optimizer.zero_grad()
    noise = torch.randn(batch_size,training_parameters["noise_size"]).to(device)
    fake = generator(noise).detach()
    fake_pred = discriminator(fake)
    loss_fake = 0.5 * lossFunc(fake_pred, torch.zeros(batch_size, 1))
    loss_fake.backward()
    real_pred = discriminator(images.reshape(-1, 3 * training_parameters["img_size"] ** 2))
    loss_real = 0.5 * lossFunc(real_pred, torch.ones(batch_size, 1))
    loss_real.backward()
    discriminator_optimizer.step()

    return loss_fake + loss_real


for epoch in range(training_parameters['n_epochs']):
    G_loss = []  # for plotting the losses over time
    D_loss = []
    for batch, (imgs, labels) in enumerate(train_loader):
        batch_size = labels.shape[0]  # if the batch size doesn't evenly divide the dataset length, this may change on the last epoch.
        # print(batch_size)
        lossG = train_generator(batch_size)
        G_loss.append(lossG)
        lossD = train_discriminator(batch_size, imgs)
        # lossD = train_discriminator(batch_size, imgs)
        D_loss.append(lossD)

        if ((batch + 1) % 500 == 0 and (epoch + 1) % 1 == 0):
            # Display a batch of generated images and print the loss
            print("Training Steps Completed: ", batch)
            with torch.no_grad():  # disables gradient computation to speed things up
                noise = torch.randn(batch_size, training_parameters["noise_size"]).to(device)
                fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
                generated_data = generator(noise).cpu().view(batch_size, 3, training_parameters["img_size"], training_parameters["img_size"])

                # display generated images
                batch_sqrt = int(training_parameters['batch_size'] ** 0.5)
                fig, ax = plt.subplots(batch_sqrt, batch_sqrt, figsize=(15, 15))
                for i, x in enumerate(generated_data):
                    ax[math.floor(i / batch_sqrt)][i % batch_sqrt].imshow(x.detach().numpy(), interpolation='nearest', cmap='gray')
                    ax[math.floor(i / batch_sqrt)][i % batch_sqrt].get_xaxis().set_visible(False)
                    ax[math.floor(i / batch_sqrt)][i % batch_sqrt].get_yaxis().set_visible(False)
                # plt.show()
                fig.savefig(f"./results/CGAN_Generations_Epoch_{epoch}")
                print(
                    f"Epoch {epoch}: loss_d: {torch.mean(torch.FloatTensor(D_loss))}, loss_g: {torch.mean(torch.FloatTensor(G_loss))}")