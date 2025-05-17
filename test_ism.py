from __future__ import annotations

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from hoa import real_sh
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from hoa import forward_hoa
from utils import sph2cart, cart2sph, fractional_delay_filter


class ISM_HOA(Dataset):

    def __init__(self):
        r"""Usa image source method data to construct HOA dataset."""
        self.c = 343
        self.sr = 24000
        self.duration = 0.1
        self.clip_samples = round(self.duration * self.sr)

        self.ism_order = 1
        self.hoa_order = 7

    def __getitem__(self, _):
        r"""Calculate an HOA data.

        c: channles, (hoa_order+1)^2
        t: samples_num
        n: image_sources_num

        Returns:
            hoa: (c, t)
        """

        t1 = time.time()

        # Build shoebox room
        corners = np.array([[0, 0], [0, 6], [6, 6], [6, 0]]).T
        room = pra.Room.from_corners(corners=corners, max_order=self.ism_order)
        room.extrude(height=4)

        # Microphone position
        mic_pos = np.array([3, 3, 2])

        # Source
        src = np.zeros(self.clip_samples)
        src[0] = 1

        # Sample source position
        azi = random.uniform(0, 2 * math.pi)
        col = random.uniform(np.deg2rad(30), np.deg2rad(150))
        direction = sph2cart(r=1., azi=azi, ele=math.pi / 2 - col)
        src_pos = mic_pos + direction  # (3,)

        # Add source and microphone
        room.add_source(src_pos)
        room.add_microphone(mic_pos)

        # Calculate image positions
        room.image_source_model()
        images = room.sources[0].images.T  # (n, 3)

        azis, cols, wavs = [], [], []

        for img in images:
            
            mic_to_img = img - mic_pos

            # Get azimuth and elevation of an image source
            r, azi, ele = cart2sph(mic_to_img)
            azi = azi * np.ones(self.clip_samples)  # (t,)
            col = (math.pi / 2 - ele) * np.ones(self.clip_samples)  # (t,)

            # Calculate delay IR of an image source
            delayed_samples = (r / self.c) * self.sr
            gain = 1. / r
            h_delay = gain * fractional_delay_filter(delayed_samples)
            
            # Wav of an image source
            wav = fftconvolve(in1=src, in2=h_delay, mode="same")  # (t,)

            wavs.append(wav)
            azis.append(azi)
            cols.append(col)

        wavs = np.stack(wavs, axis=0)  # (n, t)
        azis = np.stack(wavs, axis=0)  # (n, t)
        cols = np.stack(cols, axis=0)  # (n, t)

        # HOA coefficients along time axis
        hoa = forward_hoa(value=wavs, azi=azis, col=cols, order=self.hoa_order, reduction=None)  # (c, n, t)
        hoa = np.sum(hoa, axis=1)  # (c, t)

        print("Time: {:.3f}".format(time.time() - t1))

        return hoa

    def __len__(self):
        return 1000


if __name__ == '__main__':

    dataset = ISM_HOA()

    # Example of fetch a data
    print(dataset[0])

    # Example of dataloader
    B = 4
    dataloader = DataLoader(dataset=dataset, batch_size=B)

    for data in dataloader:
        print(data.shape)
        break

    hoa = data[0]
    fig, axs = plt.subplots(16, 4, figsize=(15, 15))
    for i in range(64):
        axs[i//4, i%4].stem(hoa[i])
        axs[i//4, i%4].set_xticklabels([])
        axs[i//4, i%4].set_yticklabels([])
    plt.savefig("sim_data.pdf")
    print("Write out to sim_data.pdf")
