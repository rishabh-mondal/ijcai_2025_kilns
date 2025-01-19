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
import os


def test_fn(gen_H, gen_Z, test_loader):
    loop = tqdm(test_loader, leave=True)
    for idx, (zebra, horse,zebra_filename,horse_filename) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        with torch.no_grad():
            fake_horse = gen_H(zebra)
            fake_zebra = gen_Z(horse)

        fake_zebra_files = zebra_filename[0].split(".")[0]
        fake_horse_files = horse_filename[0].split(".")[0]
        directory1 = "../data/cyclegan_data/test_output/fake_test_bihar_same_class_count_10_120_1000"
        # directory2 = "../data/cyclegan_data/test_output/fake_zebra"

        os.makedirs(directory1, exist_ok=True)
        # os.makedirs(directory2, exist_ok=True)


        save_image(fake_horse, f"{directory1}/{fake_horse_files}.png")
        # save_image(fake_zebra, f"{directory2}/{fake_zebra_files}.png")
        # save_image(fake_horse, f"../data/cyclegan_data/test_output/fake_horse/{base_name1}.png")
        # save_image(fake_zebra, f"../data/cyclegan_data/test_output/fake_zebra/{base_name2}.png")
        # save_image(fake_zebra, f"../data/cyclegan_data/test_output/fake_zebra/{base_name2}.png")

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
        root_horse=config.TEST_DIR + "/region_performance/test_bihar_same_class_count_10_120_1000/images",
        root_zebra=config.TEST_DIR + "/region_performance/bihar_same_class_count_10_120_1000/images",
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
