import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
import gym
import torchvision.utils as vutils

from Discriminator import Discriminator
from Generator import Generator
from InputWrapper import InputWrapper

log = gym.logger
log.set_level(gym.logger.INFO)

BATCH_SIZE = 16
LATENT_VECTOR_SIZE = 16
DISCR_FILTERS = 64
GENER_FILTERS = 64
IMAGE_SIZE = (64, 64)
LR = 1e-4
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1e3
ENV_NAMES = ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')


# Press the green button in the gutter to run the script.
def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_done:
            e.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda computation")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    envs = [InputWrapper(gym.make(name)) for name in ENV_NAMES]
    input_shape = envs[0].observation_space.shape

    net_discr = Discriminator(input_shape=input_shape, discr_filters=DISCR_FILTERS).to(device)
    net_gener = Generator(output_shape=input_shape, gener_filters=GENER_FILTERS, latent_vector=LATENT_VECTOR_SIZE)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LR, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LR, betas=(0.5, 0.999))
    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, dtype=torch.float32, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=device)

    for batch_v in iterate_batches(envs):
        # generate extra fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1).to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)

        # train dicsriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no +=1
        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e", iter_no, np.mean(gen_losses), np.mean(dis_loss))
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", vutils.make_grid(gen_output_v.data[:64], normalize=True), iter_no)
            writer.add_image("real", vutils.make_grid(batch_v.data[:64], normalize=True), iter_no)

