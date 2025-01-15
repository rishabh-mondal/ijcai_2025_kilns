from dataset import HorseZebraDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image


import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def test_fn(gen_H, gen_Z, test_loader):
    loop = tqdm(test_loader, leave=True)
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        with torch.no_grad():
            fake_horse = gen_H(zebra)
            fake_zebra = gen_Z(horse)

        save_image(fake_horse, f"../data/cyclegan_data/test_output/fake_horse_{idx}.png")
        save_image(fake_zebra, f"../data/cyclegan_data/test_output/fake_zebra_{idx}.png")

def main():
    # Initialize models
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )






    # Load pre-trained models
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE)

    # Load test dataset
    test_dataset = HorseZebraDataset(
        root_horse=config.TEST_DIR + "/vlm_data/bihar_most_15/images",
        root_zebra=config.TEST_DIR + "/vlm_data/haryana_most_15/images",
        transform=config.transforms,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Test one image at a time for simplicity
        shuffle=False,
        pin_memory=True,
    )

    # Run the test function
    test_fn(gen_H, gen_Z, test_loader)

if __name__ == "__main__":
    main()
