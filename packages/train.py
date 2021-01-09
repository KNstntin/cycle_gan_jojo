from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torchvision
import numpy as np

def de_norm(tensor):
    return (tensor + 1)/2
    
def plot_history(history_a, history_b, val_history, label_a = 'generator loss', label_b = 'descriminator_loss'):
    plt.figure(figsize=(10,5))
    plt.plot(history_a, label=label_a, zorder=1)
    plt.plot(history_b, label=label_b, zorder=1)
    points = np.array(val_history)
    steps = list(range(0, len(history_a) + 1, int(len(history_a) / len(val_history))))[1:]
    plt.scatter(steps, val_history, marker='+', s=180, c='orange', label='epochs', zorder=2)
    plt.legend(loc='best')
    plt.show()

def train(model, dataloader_images, dataloader_target, dataloader_test, opt_generator, opt_descriminator, device = 'cpu', sch_gen = None, sch_descr = None, epochs=1):
    losses_generator = []
    losses_descriminator = []
    val_losses = []
    for i in range(epochs):
        model.train()
        buffer_straight = []
        buffer_inverse = []
        for image, target in tqdm(zip(dataloader_images, dataloader_target), total = min(len(dataloader_images), len(dataloader_target))):
            image = image.to(device)
            target = target.to(device)

            if image.shape[0] > target.shape[0]:
                image = image[:target.shape[0]]
            elif image.shape[0] < target.shape[0]:
                target = target[:image.shape[0]]

            loss_generator = model.generator_loss(image, target)
            opt_generator.zero_grad()
            loss_generator.backward()
            opt_generator.step()
            
            loss_descriminator = model.descriminator_loss(image, target)
            opt_descriminator.zero_grad()
            loss_descriminator.backward()
            opt_descriminator.step()


            losses_descriminator.append(float(loss_descriminator))
            losses_generator.append(float(loss_generator))
        
        if sch_gen is not None:
            sch_gen.step()
        if sch_descr is not None:
            sch_descr.step()
        
        img = 0
        for img in dataloader_test:
            img = img.to(device)
            break
        val_losses.append(0)

        clear_output()
        print('Epoch: {}'.format(i+1))
        f, ax = plt.subplots(2, figsize = (5, 10)) 
        ax[0].imshow(torchvision.transforms.ToPILImage()(de_norm(img[0].cpu())))
        ax[1].imshow(torchvision.transforms.ToPILImage()(de_norm(model(img)[0].cpu())))
        plot_history(losses_generator, losses_descriminator, val_losses)