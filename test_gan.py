import os
import time
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image


def data_load():
    transforms_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    data_dir = '../Face-Mask-Classification-20000-Dataset/'
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    print('Train dataset size: ', len(train_dataset))
    
    class_names = train_dataset.classes
    print('Class names: ', class_names)
    
    return train_dataloader


class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=64, class_num=2):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.label_embed = nn.Embedding(self.class_num, self.class_num)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_embed(labels)), -1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=64, class_num=2):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num
        
        self.label_embed = nn.Embedding(self.class_num, 1 * 64 * 64)

        def make_block(in_channels, out_channels, bn=True):
            block = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)]
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout2d(0.25))
            if bn:
                block.append(nn.BatchNorm2d(out_channels, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *make_block(2, 32, bn=False),
            *make_block(32, 64),
            *make_block(64, 128),
            *make_block(128, 256),
            *make_block(256, 512),
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        embed = self.label_embed(labels).view((img.size(0), 1, 64, 64))
        x = torch.cat((img, embed), 1)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train(opt):
    train_dataloader = data_load()

    # ?????????(generator)??? ?????????(discriminator) ?????????
    infogan_generator = Generator()
    infogan_discriminator = Discriminator()

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        infogan_generator.cuda()
        infogan_discriminator.cuda()

    # ?????????(weights) ?????????
    infogan_generator.apply(weights_init_normal)
    infogan_discriminator.apply(weights_init_normal)

    # ?????? ??????(loss function)
    adversarial_loss = nn.MSELoss()
    adversarial_loss.cuda()

    # ?????????(learning rate) ??????
    lr = 0.0001

    # ???????????? ???????????? ?????? ????????? ??????
    optimizer_G = torch.optim.Adam(infogan_generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(infogan_discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # n_epochs = 200 # ????????? ??????(epoch) ??????
    # latent_dim = 100
    n_classes = 2
    sample_interval = 500 # ??? ?????? ??????(batch)?????? ????????? ????????? ????????? ??????
    start_time = time.time()

    os.makedirs('./results/infogan/', exist_ok=True)

    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(train_dataloader):

            # ??????(real) ???????????? ??????(fake) ???????????? ?????? ?????? ????????? ??????
            real = torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(1.0) # ??????(real): 1
            fake = torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(0.0) # ??????(fake): 0

            real_imgs = imgs.cuda()
            labels = labels.cuda()

            """ ?????????(generator)??? ???????????????. """
            optimizer_G.zero_grad()

            # ?????? ?????????(noise) ??? ?????? ?????????(label) ?????????
            z = torch.normal(mean=0, std=1, size=(imgs.shape[0], opt.latent_dim)).cuda()
            generated_labels = torch.randint(0, n_classes, (imgs.shape[0],)).cuda()

            # ????????? ??????
            generated_imgs = infogan_generator(z, generated_labels)

            # ?????????(generator)??? ??????(loss) ??? ??????
            g_loss = adversarial_loss(infogan_discriminator(generated_imgs, generated_labels), real)

            # ?????????(generator) ????????????
            g_loss.backward()
            optimizer_G.step()

            """ ?????????(discriminator)??? ???????????????. """
            optimizer_D.zero_grad()

            # ?????????(discriminator)??? ??????(loss) ??? ??????
            real_loss = adversarial_loss(infogan_discriminator(real_imgs, labels), real)
            fake_loss = adversarial_loss(infogan_discriminator(generated_imgs.detach(), generated_labels), fake)
            d_loss = (real_loss + fake_loss) / 2

            # ?????????(discriminator) ????????????
            d_loss.backward()
            optimizer_D.step()

            done = epoch * len(train_dataloader) + i
            if done % sample_interval == 0:
                # ???????????? 8?????? ???????????? ???????????? 2 X 8 ?????? ???????????? ??????
                z = torch.normal(mean=0, std=1, size=(n_classes * 8, opt.latent_dim)).cuda()
                labels = torch.LongTensor([i for i in range(n_classes) for _ in range(8)]).cuda()
                generated_imgs = infogan_generator(z, labels)
                save_image(generated_imgs, f'./results/infogan/{done}.png', nrow=8, normalize=True)

        # ????????? epoch??? ?????? ????????? ??????(log) ??????
        print(f'[Epoch {epoch}/{opt.n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]')

    # ?????? ???????????? ??????
    torch.save(infogan_generator.state_dict(), 'infoGAN_Generator_for_Face_Mask.pt')
    torch.save(infogan_discriminator.state_dict(), 'infoGAN_Discriminator_for_Face_Mask.pt')
    print('Model saved!')


def test(latent_dim):
    infogan_generator = Generator()
    infogan_generator.cuda()
    infogan_generator.load_state_dict(torch.load('infoGAN_Generator_for_Face_Mask.pt'))
    infogan_generator.eval()

    for label_idx in range(2):
        # ?????? ?????????(noise) ??? ?????? ?????????(label) ?????????
        z = torch.normal(mean=0, std=1, size=(100, latent_dim)).cuda()
        generated_labels = torch.cuda.IntTensor(100).fill_(label_idx) # fill_(0)

        # ????????? ??????
        generated_imgs = infogan_generator(z, generated_labels)

        # ????????? ????????? ????????? 100?????? ???????????? 10 X 10 ?????? ???????????? ??????
        dir_name = 'with_mask' if label_idx == 0 else 'with_no_mask'
        save_image(generated_imgs.data[:100], f'./results/infogan/{dir_name}.png', nrow=10, normalize=True)

    # fid data
    os.makedirs('./results/infogan/with_mask/', exist_ok=True)
    os.makedirs('./results/infogan/without_mask/', exist_ok=True)

    # ????????? ??????/????????? 10 * 100?????? ?????? ???????????? ??????
    def gen_face_image(label_idx):
        dir_name = 'with_mask' if label_idx == 0 else 'without_mask'

        for i in range(10):
            # ?????? ?????????(noise) ??? ?????? ?????????(label) ?????????
            z = torch.normal(mean=0, std=1, size=(100, latent_dim)).cuda()
            generated_labels = torch.cuda.IntTensor(100).fill_(label_idx)
            # ????????? ??????
            generated_imgs = infogan_generator(z, generated_labels)

            for j in range(100):
                save_image(generated_imgs.data[j], f'./results/infogan/{dir_name}/{i * 100 + j}.png', normalize=True)

    gen_face_image(0)
    gen_face_image(1)

    # ??????. fid??? ?????? ?????? ???????????? ??????
    # !python ./pytorch-frechet-inception-distance/fid.py --path1 ./results/custom/without_mask --path2 ./Face-Mask-Classification-20000-Dataset/test/without_mask --batch-size 32
    # !python ./pytorch-frechet-inception-distance/fid.py --path1 ./results/custom/with_mask --path2 ./Face-Mask-Classification-20000-Dataset/test/with_mask --batch-size 32


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    opt = parser.parse_args()
    # print(opt)

    train(opt)
    test(opt.latent_dim)