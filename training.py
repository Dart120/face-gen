import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from Discriminator import Discriminator
from Generator import Generator
from train_fn import train_fn

torch.backends.cudnn.benchmarks = True

device = 'cuda'
lst = [256,256,256,256]
batch_size = [32,32,32,32]
e_limit = [4,8,8,12]
print(f'Optimisation steps {sum(e_limit) * 1563}')


def main():
    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    # but really who cares..
    gen = Generator(lst).to(device)
    critic = Discriminator(lst).to(device)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(gen.parameters(), lr=0.003 * 0.01, betas=(0.0, 0.99))
    opt_gen = optim.Adam(gen.parameters(), lr=0.003, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=0.003, betas=(0.0, 0.99))
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()


    gen.train()
    critic.train()
    
    # start at step that corresponds to img size that we set in config
    for step,features in enumerate(lst):
        print(step)
     
        
        loader, dataset = get_loader(step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        f_noise = torch.randn(batch_size[step],lst[0], 1, 1).to(device)
        print(f"Current image size: {4 * 2 ** step}")
       
        train_fn(
            critic,
            gen,
            loader,
        
            step,
            opt_critic,
            opt_gen,
            scaler_gen,
            scaler_critic,'cuda',lst,f_noise,dataset,e_limit
        )



    
        save_checkpoint(gen, opt_gen, filename='config.CHECKPOINT_GEN')
        save_checkpoint(critic, opt_critic, filename='config.CHECKPOINT_CRITIC')



def get_loader(stage):
    image_size = 4*2**stage
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(3)],
                [0.5 for _ in range(3)],
            ),
        ]
    )
    
    dataset = datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size[stage],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    return loader, dataset
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

if __name__ == "__main__":
    main()