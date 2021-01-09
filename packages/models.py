import torch
from torch import nn

L1 = nn.L1Loss()
MSE = nn.MSELoss()

def init_weights(x, m=0.0, s=0.02):
    if type(x) == nn.Conv2d:
        torch.nn.init.normal(x.weight, m, s)

class ResnetBlock(nn.Module):
    def __init__(self, ch, normalize = True, dropout = 0):
        super().__init__()
        self.normalize = normalize
        self.conv1 = nn.Conv2d(ch, ch, kernel_size = 3, padding = 1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(ch, ch, kernel_size = 3, padding = 1, padding_mode='reflect')
        if self.normalize:
            self.in1 = nn.InstanceNorm2d(ch)
            self.in2 = nn.InstanceNorm2d(ch)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        res = self.conv1(x)
        res = torch.relu(res)
        if self.normalize:
            res = self.in1(res)
        res = self.dropout(res)
        res = self.conv2(res)
        res = torch.relu(res)
        if self.normalize:
          res = self.in2(res)
        res = self.dropout(res)
        return res + x


class Generator(nn.Module):
    def __init__(self, resnet_layers = 9, dim = 32, dropout = 0):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, dim, 7, padding = 3, padding_mode = 'reflect'),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.ReLU(),

        )

        self.transformer = nn.Sequential(
            *[ResnetBlock(dim * 4, dropout) for i in range(resnet_layers)]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim * 4, dim * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(dim * 2, dim, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, 3, 7, padding = 3, padding_mode = 'reflect'),
            nn.Tanh(),
        )

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.transformer.apply(init_weights)


    def forward(self, x):
        coded = self.encoder(x)
        transformed = self.transformer(coded)
        decoded = self.decoder(transformed)
        return decoded



def Descriminator(dim = 64):
    result = nn.Sequential(
        nn.Conv2d(3, dim, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2),

        nn.Conv2d(dim, dim*2, 4, stride=2, padding=1),
        nn.InstanceNorm2d(dim*2),
        nn.LeakyReLU(0.2),

        nn.Conv2d(dim*2, dim*4, 4, stride=2, padding=1),
        nn.InstanceNorm2d(dim*4),
        nn.LeakyReLU(0.2),

        nn.Conv2d(dim*4, dim*8, 4, padding=1),
        nn.InstanceNorm2d(dim*8),
        nn.LeakyReLU(0.2),

        nn.Conv2d(dim*8, 1, 4, padding=1),
    )

    result.apply(init_weights)
    return result

class GAN(nn.Module):

    def __init__(self, buffer, device = 'cpu', resnet_layers=9, transform = None, dim_gen = 64, dim_descr = 64, dropout = 0):
        super().__init__()
        self.transform = transform
        self.generator = Generator(dim = dim_gen, resnet_layers = resnet_layers, dropout = dropout)
        self.descriminator = Descriminator(dim = dim_descr)
        self.buffer = buffer
        self.device = device
        # self.descriminator = torchvision.models.vgg16(pretrained=False)
        # self.descriminator.classifier = nn.AdaptiveAvgPool2d(1)
        # self.descriminator = nn.Sequential(self.descriminator,
        #                                    nn.Flatten(),
        #                                    nn.Sigmoid())

    def descriminator_loss(self, false, true, return_image = False):
        prob_false = self.descriminator(self.buffer.get_batch(self.generator(false).detach()))
        prob_true = self.descriminator(true)
        if return_image:
            return MSE(prob_true, torch.ones_like(prob_true).to(self.device)) + MSE(prob_false, torch.zeros_like(prob_false).to(self.device)), img
        else:
            return MSE(prob_true, torch.ones_like(prob_true).to(self.device)) + MSE(prob_false, torch.zeros_like(prob_false).to(self.device))

    def generator_loss(self, false, return_image = False):
        img = self.generator(false)
        if self.transform is not None:
            prob_false = self.descriminator(self.transform(img))
        else:
            prob_false = self.descriminator(img)
        if return_image:
            return MSE(prob_false, torch.ones_like(prob_false).to(self.device)), img
        else:
            return MSE(prob_false, torch.ones_like(prob_false).to(self.device))

    def forward(self, x):
        self.eval()
        with torch.no_grad():
            return self.generator(x)
            
class CycleGAN(nn.Module):

    def __init__(self, buffer_images, buffer_targets, device = 'cpu', resnet_layers=9, transform = None, lambd = 10, dim_gen = 64, dim_descr = 64, dropout = 0):
        super().__init__()
        self.gan_straight = GAN(buffer_images, device = device, resnet_layers=resnet_layers, transform=transform, dim_gen=dim_gen, dim_descr=dim_descr, dropout = dropout)
        self.gan_inverse = GAN(buffer_targets, device = device, resnet_layers=resnet_layers, transform=transform, dim_gen=dim_gen, dim_descr=dim_descr, dropout = dropout)
        self.lambd = lambd

    def forward(self, x, inverse = False):
        if inverse:
            return self.gan_inverse(x)
        else:
            return self.gan_straight(x)

    def generator_loss(self, x, y):
        loss_straight, img_straight = self.gan_straight.generator_loss(x, True)
        loss_inverse, img_inverse = self.gan_inverse.generator_loss(y, True)
        reversed_x = self.gan_inverse.generator(img_straight)
        reversed_y = self.gan_straight.generator(img_inverse)
        return loss_straight + loss_inverse + self.lambd *( L1(x, reversed_x) + L1(y, reversed_y))
        # (self.lambd / 2) * (torch.mean(torch.abs(x - img_straight)) + torch.mean(torch.abs(y - img_inverse)))
    
    def descriminator_loss(self, x, y):
        return self.gan_straight.descriminator_loss(x, y) + self.gan_inverse.descriminator_loss(y, x)


class Buffer:
    def __init__(self, device, cap=50):
        self.memory = torch.empty(cap,3,256,256).to(device)
        self.device = device
        self.len = 0
        
    def get_batch(self,batch):
        result = torch.empty(batch.shape).to(self.device)
        for i, img in enumerate(batch):
            if self.len < self.memory.shape[0]:
                self.memory[self.len] = img
                result[i] = img
                self.len += 1
            else:
                if torch.rand(1) < 0.5:
                    index = torch.randint(low=0, high=self.memory.shape[0]-1, size=(1,))
                    result[i] = self.memory[index].clone()
                    self.memory[index] = img
                else:
                    result[i] = img
        return result