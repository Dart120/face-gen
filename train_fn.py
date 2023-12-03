import torch
from tqdm import tqdm
import wandb
import numpy as np

wandb.init(project='anime')



def train_fn(
    critic,
    gen,
    loader,
    step,
    
    opt_critic,
    opt_gen,
    scaler_gen,
    scaler_critic,
    device,
    lst,
    f_noise,
    dataset,
    e_limits
):
    images_shown = 0
    EPOCHS = 0
    alpha = 1e-5
    
    while EPOCHS < e_limits[step]:
        print(f"epoch: {EPOCHS}")
        loop = tqdm(loader, leave=True)

        for batch_idx, (real, _) in enumerate(loop):
        
            if step == 0:
                alpha = 1
     
            real = real.to(device)
          
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
            # which is equivalent to minimizing the negative of the expression
            noise = torch.randn(cur_batch_size,256, 1, 1).to(device)

            with torch.cuda.amp.autocast():
                fake = gen(noise, alpha, step)
                
                critic_real = critic(real, alpha, step)
           
                critic_fake = critic(fake.detach(), alpha, step)
               
                gp = gradient_penalty(critic, real, fake, alpha, step, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + 10 * gp
                    + (0.001 * torch.mean(critic_real ** 2))
                )

            opt_critic.zero_grad()
            scaler_critic.scale(loss_critic).backward()
            scaler_critic.step(opt_critic)
            scaler_critic.update()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            with torch.cuda.amp.autocast():
                gen_fake = critic(fake, alpha, step)
                loss_gen = -torch.mean(gen_fake)

            opt_gen.zero_grad()
            scaler_gen.scale(loss_gen).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()
            alpha += cur_batch_size / (
            (e_limits[step] * 0.5) * len(dataset)
            )
            alpha = min(alpha, 1)
            
            
    

            if batch_idx % 500 == 0:
                with torch.no_grad():
                    fixed_fakes = gen(f_noise, alpha, step) * 0.5 + 0.5
                    wandb.log({'images_sshown': images_shown, 'c_loss': loss_critic.item(),'g_loss':loss_gen.item()})
                    log_images(fixed_fakes)
                    log_images(real,name = 'reals')

            loop.set_postfix(
                gp=gp.item(),
                loss_critic=loss_critic.item(),
                alpha = alpha,
                loss_g = loss_gen.item()

            )
            images_shown += real.shape[0]
        EPOCHS += 1

    return alpha

def gradient_penalty(critic, real, fake, alpha, train_step, device="cuda"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
   
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    
    interpolated_images.requires_grad_(True)

    # Calculate critic scores

    mixed_scores = critic(interpolated_images, alpha, train_step)
  
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores).to(device),
        create_graph=True,
        retain_graph=True,

    )[0]
 

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def log_images(images_tensor, num_images=16,name='fake'):
    # Convert PyTorch tensor to NumPy array and move it to CPU
    images_np = images_tensor.cpu().numpy()

    # Log the images
    images_to_log = []
    for i in range(min(num_images, images_np.shape[0])):
        # Ensure that the image has the correct format (H, W, C) for W&B
        img = np.transpose(images_np[i], (1, 2, 0))

     

        # Create a wandb.Image object and append it to the list
        images_to_log.append(wandb.Image(img))

    # Log the images using wandb.log()

    wandb.log({name: images_to_log})