import torch
import torchvision.utils as vutils
import os
from architectures.cycle_GANs_blocks import ReplayBuffer
from tqdm import tqdm
import psutil
import time

def return_trained_model(save_dir, model_name, dataset_name, infer, run_id, train_dataloader, training_setup, model_config, dataset_config, device):
    
    if model_name == 'cgan256' or model_name == 'cgan64':
        trainer = train_cgan
    elif model_name == 'ganomaly':
        trainer = train_ganomaly
    elif model_name == 'patchcore':
        trainer = train_patchcore
    elif model_name == 'padim':
        trainer = train_padim
    else:
        raise ValueError('Unknown model name: {}'.format(model_name))

    return trainer(save_dir, model_name, dataset_name, infer, run_id, train_dataloader, training_setup, model_config, dataset_config, device)


def train_cgan(save_dir, model_name, dataset_name, infer, run_id, train_dataloader, training_setup, model_config, dataset_config, device):

    if infer:
        # Load the trained model(s)
        model = training_setup['models']['netG_abnormal2normal']
        model.load_state_dict(torch.load(os.path.join(save_dir, dataset_name, model_name, str(run_id), "netG_abnormal2normal.pth")))
        return model.to(device)
    
    # Training
    normal_folder = os.path.join(save_dir, dataset_name, model_name, str(run_id), "normal")
    if not os.path.exists(normal_folder):
        os.makedirs(normal_folder)
    abnormal_folder = os.path.join(save_dir, dataset_name, model_name, str(run_id), "abnormal")
    if not os.path.exists(abnormal_folder):
        os.makedirs(abnormal_folder)

    logfile = open(os.path.join(save_dir, dataset_name, model_name, str(run_id), "training_logs.txt"), "a", buffering=1)

    fake_normal_buffer = ReplayBuffer()
    fake_abnormal_buffer = ReplayBuffer()
    all_time = time.time()
    training_setup["models"]["netG_normal2abnormal"].train()
    training_setup["models"]["netG_abnormal2normal"].train()
    training_setup["models"]["netD_normal"].train()
    training_setup["models"]["netD_abnormal"].train()
    for epoch in range(dataset_config['epochs']):

        # Training
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        record_loss_D = 0
        record_loss_identity = 0
        record_loss_GAN = 0
        record_loss_cycle = 0
        start_time = time.time()
        for i, data in progress_bar:
            
            # get batch size data
            real_image_normal = data["normal"].to(device)
            real_image_abnormal = data["abnormal"].to(device)

            ##############################################
            # (1) Update G network: Generators A2B and B2A
            ##############################################

            # Set G_A and G_B's gradients to zero
            training_setup["optimizers"]["optimizer_G"].zero_grad()

            # Identity loss
            # G_B2A(A) should equal A if real A is fed
            identity_image_normal = training_setup["models"]["netG_abnormal2normal"](real_image_normal)
            loss_identity_normal = training_setup["losses"]["identity_loss"](identity_image_normal, real_image_normal) * 5.0

            # G_A2B(B) should equal B if real B is fed
            identity_image_abnormal = training_setup["models"]["netG_normal2abnormal"](real_image_abnormal)
            loss_identity_abnormal = training_setup["losses"]["identity_loss"](identity_image_abnormal, real_image_abnormal) * 5.0

            # GAN loss
            # GAN loss D_A(G_A(A))
            fake_image_normal = training_setup["models"]["netG_abnormal2normal"](real_image_abnormal)
            fake_output_normal = training_setup["models"]["netD_normal"](fake_image_normal)
            loss_GAN_abnormal2normal = training_setup["losses"]["adversarial_loss"](fake_output_normal, True)
            # GAN loss D_B(G_B(B))
            fake_image_abnormal = training_setup["models"]["netG_normal2abnormal"](real_image_normal)
            fake_output_abnormal = training_setup["models"]["netD_abnormal"](fake_image_abnormal)
            loss_GAN_normal2abnormal = training_setup["losses"]["adversarial_loss"](fake_output_abnormal, True)

            # Cycle loss
            recovered_image_normal = training_setup["models"]["netG_abnormal2normal"](fake_image_abnormal)
            loss_cycle_normal = training_setup["losses"]["cycle_loss"](recovered_image_normal, real_image_normal) * 10.0

            recovered_image_abnormal = training_setup["models"]["netG_normal2abnormal"](fake_image_normal)
            loss_cycle_abnormal = training_setup["losses"]["cycle_loss"](recovered_image_abnormal, real_image_abnormal) * 10.0

            # Combined loss and calculate gradients
            errG = loss_identity_normal + loss_identity_abnormal + loss_GAN_normal2abnormal + loss_GAN_abnormal2normal + loss_cycle_normal + loss_cycle_abnormal

            # Calculate gradients for G_A and G_B
            errG.backward()
            
            # Update G_A and G_B's weights
            training_setup["optimizers"]["optimizer_G"].step()

            ##############################################
            # (2) Update D network: Discriminator A
            ##############################################

            # Set D_A gradients to zero
            training_setup["optimizers"]["optimizer_D_normal"].zero_grad()

            # Real A image loss
            real_output_normal = training_setup["models"]["netD_normal"](real_image_normal)
            errD_real_normal = training_setup["losses"]["adversarial_loss"](real_output_normal, True)

            # Fake A image loss
            fake_image_normal = fake_normal_buffer.push_and_pop(fake_image_normal)
            fake_output_normal = training_setup["models"]["netD_normal"](fake_image_normal.detach())
            errD_fake_normal = training_setup["losses"]["adversarial_loss"](fake_output_normal, False)

            # Combined loss and calculate gradients
            errD_normal = (errD_real_normal + errD_fake_normal) / 2.

            # Calculate gradients for D_A
            errD_normal.backward()
            
            # Update D_A weights
            training_setup["optimizers"]["optimizer_D_normal"].step()

            ##############################################
            # (3) Update D network: Discriminator B
            ##############################################

            # Set D_B gradients to zero
            training_setup["optimizers"]["optimizer_D_abnormal"].zero_grad()

            # Real B image loss
            real_output_abnormal = training_setup["models"]["netD_abnormal"](real_image_abnormal)
            errD_real_abnormal = training_setup["losses"]["adversarial_loss"](real_output_abnormal, True)

            # Fake B image loss
            fake_image_abnormal = fake_abnormal_buffer.push_and_pop(fake_image_abnormal)
            fake_output_abnormal = training_setup["models"]["netD_abnormal"](fake_image_abnormal.detach())
            errD_fake_abnormal = training_setup["losses"]["adversarial_loss"](fake_output_abnormal, False)

            # Combined loss and calculate gradients
            errD_abnormal = (errD_real_abnormal + errD_fake_abnormal) / 2.

            progress_bar.set_description(
                f"[{epoch}/{dataset_config['epochs'] - 1}][{i}/{len(train_dataloader) - 1}] "
                f"Loss_D: {(errD_normal + errD_abnormal).item():.4f} "
                f"Loss_G: {errG.item():.4f} "
                f"Loss_G_identity: {(loss_identity_normal + loss_identity_abnormal).item():.4f} "
                f"loss_G_GAN: {(loss_GAN_normal2abnormal + loss_GAN_abnormal2normal).item():.4f} "
                f"loss_G_cycle: {(loss_cycle_normal + loss_cycle_abnormal).item():.4f}"
            )

            record_loss_D += (errD_normal + errD_abnormal).item() / len(train_dataloader)
            record_loss_identity += (loss_identity_normal + loss_identity_abnormal).item() / len(train_dataloader)
            record_loss_GAN += (loss_GAN_normal2abnormal + loss_GAN_abnormal2normal).item() / len(train_dataloader)
            record_loss_cycle += (loss_cycle_normal + loss_cycle_abnormal).item() / len(train_dataloader)

            if i == 0:
                vutils.save_image(real_image_normal,
                                  os.path.join(normal_folder, f"{epoch}_real_samples.png"),
                                  normalize=True)
                vutils.save_image(real_image_abnormal,
                                  os.path.join(abnormal_folder, f"{epoch}_real_samples.png"),
                                  normalize=True)

                fake_image_A = 0.5 * (training_setup["models"]["netG_abnormal2normal"](real_image_abnormal).data + 1.0)
                fake_image_B = 0.5 * (training_setup["models"]["netG_normal2abnormal"](real_image_normal).data + 1.0)

                vutils.save_image(fake_image_B.detach(),
                                  os.path.join(normal_folder, f"{epoch}_fake_samples.png"),
                                  normalize=True)
                vutils.save_image(fake_image_A.detach(),
                                  os.path.join(abnormal_folder, f"{epoch}_fake_samples.png"),
                                  normalize=True)
                    
            # Calculate gradients for D_B
            errD_abnormal.backward()

            # Update D_B weights
            training_setup["optimizers"]["optimizer_D_abnormal"].step()
        
        # do check pointing
        torch.save(training_setup["models"]["netG_normal2abnormal"].state_dict(), os.path.join(save_dir, dataset_name, model_name, str(run_id), "netG_normal2abnormal.pth"))
        torch.save(training_setup["models"]["netG_abnormal2normal"].state_dict(), os.path.join(save_dir, dataset_name, model_name, str(run_id), "netG_abnormal2normal.pth"))
        torch.save(training_setup["models"]["netD_normal"].state_dict(), os.path.join(save_dir, dataset_name, model_name, str(run_id), "netD_normal.pth"))
        torch.save(training_setup["models"]["netD_abnormal"].state_dict(), os.path.join(save_dir, dataset_name, model_name, str(run_id), "netD_abnormal.pth"))

        # Update learning rates
        training_setup["schedulers"]["lr_scheduler_G"].step()
        training_setup["schedulers"]["lr_scheduler_D_normal"].step()
        training_setup["schedulers"]["lr_scheduler_D_abnormal"].step()

        training_log = "Epoch: {} | Loss_D: {:.4f} | Loss_identity: {:.4f} | Loss_GAN: {:.4f} | Loss_cycle: {:.4f} | Time: {:.4f} | RAM (GB): {:.3f} | VRAM (GB): {:.3f}".format(epoch, 
                                                                                                                                                             record_loss_D, 
                                                                                                                                                             record_loss_identity, 
                                                                                                                                                             record_loss_GAN, 
                                                                                                                                                             record_loss_cycle, 
                                                                                                                                                             time.time() - start_time, 
                                                                                                                                                             psutil.virtual_memory()[3]/1024/1024/1024, 
                                                                                                                                                             torch.cuda.memory_allocated(device)/1024/1024/1024)
        logfile.write(training_log + "\n")
    logfile.write("Total training time: {:.4f}".format(time.time() - all_time) + "\n")
    logfile.close()

    return training_setup["models"]["netG_abnormal2normal"].to(device)


def train_ganomaly(save_dir, model_name, dataset_name, infer, run_id, train_dataloader, training_setup, model_config, dataset_config, device):

    if infer:
        model = training_setup['models']['model']
        model.load_state_dict(torch.load(os.path.join(save_dir, dataset_name, model_name, str(run_id), "ganomaly.pth")))
        return model.to(device)

    # Training
    normal_folder = os.path.join(save_dir, dataset_name, model_name, str(run_id), "normal")
    if not os.path.exists(normal_folder):
        os.makedirs(normal_folder)
    abnormal_folder = os.path.join(save_dir, dataset_name, model_name, str(run_id), "abnormal")
    if not os.path.exists(abnormal_folder):
        os.makedirs(abnormal_folder)

    logfile = open(os.path.join(save_dir, dataset_name, model_name, str(run_id), "training_logs.txt"), "a", buffering=1)

    all_time = time.time()
    training_setup["models"]["model"].train()
    for epoch in range(dataset_config['epochs']):

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        record_loss_D = 0
        record_loss_G = 0
        start_time = time.time()
        for i, data in progress_bar:

            # Get the data
            real_images = data['image'].to(device)
            
            # ---
            # Update the discriminator
            # ---
            
            training_setup["optimizers"]["optimizer_D"].zero_grad()
            
            padded, fake, latent_i, latent_o = training_setup["models"]["model"](real_images)
            
            pred_real, _ = training_setup["models"]["model"].discriminator(padded)
            pred_fake, _ = training_setup["models"]["model"].discriminator(fake)
            
            loss_D = training_setup["losses"]["loss_Discriminator"](pred_real, pred_fake)
            loss_D.backward()
            training_setup["optimizers"]["optimizer_D"].step()

            # ---
            # Update the generator
            # ---
            
            training_setup["optimizers"]["optimizer_G"].zero_grad()
            
            padded, fake, latent_i, latent_o = training_setup["models"]["model"](real_images)
            
            pred_real, _ = training_setup["models"]["model"].discriminator(padded)
            pred_fake, _ = training_setup["models"]["model"].discriminator(fake)
            
            loss_G = training_setup["losses"]["loss_Generator"](latent_i, latent_o, padded, fake, pred_real, pred_fake)
            loss_G.backward()
            training_setup["optimizers"]["optimizer_G"].step()

            record_loss_D += loss_D.item() / len(train_dataloader)
            record_loss_G += loss_G.item() / len(train_dataloader)
        
        # do check pointing
        torch.save(training_setup["models"]["model"].state_dict(), os.path.join(save_dir, dataset_name, model_name, str(run_id), "ganomaly.pth"))

        training_setup["schedulers"]["lr_scheduler_G"].step()
        training_setup["schedulers"]["lr_scheduler_D"].step()

        training_log = "Epoch: {} | Loss_D: {:.4f} | Loss_G: {:.4f} | Time: {:.4f} | RAM (GB): {:.3f} | VRAM (GB): {:.3f}".format(epoch, 
                                                                                                                                  record_loss_D, 
                                                                                                                                  record_loss_G, 
                                                                                                                                  time.time() - start_time, 
                                                                                                                                  psutil.virtual_memory()[3]/1024/1024/1024, 
                                                                                                                                  torch.cuda.memory_allocated(device)/1024/1024/1024)
        logfile.write(training_log + "\n")
    logfile.write("Total training time: {:.4f}".format(time.time() - all_time) + "\n")
    logfile.close()

    return training_setup["models"]["model"].to(device)


def train_patchcore(save_dir, model_name, dataset_name, infer, run_id, train_dataloader, training_setup, model_config, dataset_config, device):

    if infer:
        raise NotImplementedError("For patchcore, you have to also train the model.")
    
    # Training
    normal_folder = os.path.join(save_dir, dataset_name, model_name, str(run_id), "normal")
    if not os.path.exists(normal_folder):
        os.makedirs(normal_folder)
    abnormal_folder = os.path.join(save_dir, dataset_name, model_name, str(run_id), "abnormal")
    if not os.path.exists(abnormal_folder):
        os.makedirs(abnormal_folder)

    logfile = open(os.path.join(save_dir, dataset_name, model_name, str(run_id), "training_logs.txt"), "a", buffering=1)

    embeddings = []
    training_setup["models"]["model"].train()
    training_setup["models"]["model"].feature_extractor.eval()
    n = 0
    all_time = time.time()
    for i, data in enumerate(train_dataloader):

        # Get the data
        real_images = data['image'].to(device)

        embedding = training_setup["models"]["model"](real_images)
        embeddings.append(embedding)
        n += len(embedding)
        logfile.write("Batch: {} | {} data embedded\n".format(i, n))

    embeddings = torch.vstack(embeddings)
    training_setup["models"]["model"].subsample_embedding(embeddings, 0.1)
    logfile.write("Total training time: {:.4f}".format(time.time() - all_time) + "\n")
    logfile.close()

    return training_setup["models"]["model"].to(device)


def train_padim(save_dir, model_name, dataset_name, infer, run_id, train_dataloader, training_setup, model_config, dataset_config, device):

    if infer:
        raise NotImplementedError("For padim, you have to also train the model.")
    
    # Training
    normal_folder = os.path.join(save_dir, dataset_name, model_name, str(run_id), "normal")
    if not os.path.exists(normal_folder):
        os.makedirs(normal_folder)
    abnormal_folder = os.path.join(save_dir, dataset_name, model_name, str(run_id), "abnormal")
    if not os.path.exists(abnormal_folder):
        os.makedirs(abnormal_folder)

    logfile = open(os.path.join(save_dir, dataset_name, model_name, str(run_id), "training_logs.txt"), "a", buffering=1)

    embeddings = []
    training_setup["models"]["model"].train()
    training_setup["models"]["model"].feature_extractor.eval()
    n = 0
    all_time = time.time()
    for i, data in enumerate(train_dataloader):
            
        # Get the data
        real_images = data['image'].to(device)

        embedding = training_setup["models"]["model"](real_images)
        embeddings.append(embedding)
        n += len(embedding)
        logfile.write("Batch: {} | {} data embedded\n".format(i, n))

    embeddings = torch.vstack(embeddings)
    training_setup["models"]["model"].gaussian.fit(embeddings)
    logfile.write("Total training time: {:.4f}".format(time.time() - all_time) + "\n")
    logfile.close()

    return training_setup["models"]["model"].to(device)