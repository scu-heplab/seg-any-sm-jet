import os
import torch
import numpy as np
import torch.utils.data

from generate_mask import particle_to_image, image_to_mask, calc_mass, calc_pt_rapidity_phi


class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, dir_path, n_mix=50, n_mask=32, flat_prob=0.5):
        super(DatasetLoader, self).__init__()

        self._signal = np.load(os.path.join(dir_path, "event.npy"))
        _, signal_counts = np.unique(self._signal[:, 0], return_counts=True)

        self._n_signal = len(signal_counts)
        self._signal_end = np.cumsum(signal_counts)
        self._signal_start = self._signal_end - signal_counts

        self._pid_list = [6, 25, 23, 24, 5, 1, 2, 3, 4, 21]
        self._pid_to_id = {6: 1, 25: 2, 23: 3, 24: 3, 5: 4, 1: 5, 2: 5, 3: 5, 4: 5, 21: 5}

        self._pileup = np.load(os.path.join(dir_path, "pileup.npy"))
        _, pileup_counts = np.unique(self._pileup[:, 0], return_counts=True)

        self._n_mix = n_mix
        self._n_pileup = len(pileup_counts)
        self._pileup_end = np.cumsum(pileup_counts)
        self._pileup_start = self._pileup_end - pileup_counts

        self._n_mask = n_mask
        self._flat_prob = flat_prob

    def __getitem__(self, item):
        index = item

        mask_label = []
        class_label = []
        momentum_label = []
        events, particle_label, particle_momentum = self._assigned(self._signal[self._signal_start[index]:self._signal_end[index], 1:])

        prompt = np.zeros((0,), np.int32)
        # prompt = np.unique(np.random.randint(1, 6, (5,))) if np.random.rand(1)[0] < 0.8 else np.zeros((0,), np.int32)
        prompt = np.pad(prompt, (0, 5 - len(prompt)))

        for pl, pm in zip(particle_label, particle_momentum):
            tpl = calc_pt_rapidity_phi(pl[:, 1:])
            pid = self._pid_to_id[int(np.abs(pl[0, 0]))]
            if (pid in prompt or np.all(prompt == 0)) and (len(pm) > 0 and tpl[0, 0] > 50 and -np.pi < tpl[0, 1] < np.pi):
                target = image_to_mask(particle_to_image(pm[:, 1:]))
                mask_label.append(target)
                class_label.append(self._pid_to_id[int(np.abs(pl[0, 0]))] if np.sum(target) > 0.0 else 0)
                momentum_label.append(pl[0, 1:5] if np.sum(target) > 0.0 else np.zeros_like(pl[0, 1:5]))
        if len(mask_label) == 0:
            mask_label = np.zeros((self._n_mask, 315, 315))
            class_label = np.zeros((self._n_mask,))
            momentum_label = np.zeros((self._n_mask, 4))
        else:
            mask_label = np.pad(np.stack(mask_label, 0)[:self._n_mask], ((0, max(0, self._n_mask - len(mask_label))), (0, 0), (0, 0)))
            class_label = np.pad(np.array(class_label)[:self._n_mask], (0, max(0, self._n_mask - len(class_label))))
            momentum_label = np.pad(np.stack(momentum_label, 0)[:self._n_mask], ((0, max(0, self._n_mask - len(momentum_label))), (0, 0)))
        signal = np.transpose(particle_to_image(events[:, 1:]), (2, 0, 1))

        index = np.random.permutation(self._n_pileup)[:np.random.poisson(self._n_mix, (1,))[0]]
        pileup = np.transpose(particle_to_image(np.concatenate([self._pileup[self._pileup_start[idx]:self._pileup_end[idx]] for idx in index], 0)[:, 3:]), (2, 0, 1))

        image = np.concatenate([signal[:5] + pileup[:5], np.maximum(signal[5:], pileup[5:])], 0)
        momentum_label = np.concatenate([calc_pt_rapidity_phi(momentum_label), calc_mass(momentum_label)], -1)

        momentum_label[:, 1] = np.where(class_label > 0, momentum_label[:, 1] + np.pi, 0)
        momentum_label[:, 3] = np.where(np.logical_and(class_label > 0, momentum_label[:, 3] < 1e-3), 1e-3, momentum_label[:, 3])

        image = np.array(image, np.float32)
        prompt = np.array(prompt, np.int64)
        mask_label = np.array(mask_label, np.float32)
        class_label = np.array(class_label, np.int64)
        momentum_label = np.array(momentum_label, np.float32)

        if np.random.rand() < self._flat_prob:
            z = np.random.rand() * 0.8 + 0.1
            cut = int(np.ceil(2 * np.pi * z / 0.02))
            image = np.concatenate([image[:, cut:], image[:, :cut]], 1)
            mask_label = np.concatenate([mask_label[:, cut:], mask_label[:, :cut]], 1)

            mask = np.array(momentum_label.sum(-1) != 0, dtype=np.float32)
            momentum_label[:, 2] = np.where(momentum_label[:, 2] < 2 * np.pi * z, momentum_label[:, 2] + 2 * np.pi * (1 - z) * mask, momentum_label[:, 2] - 2 * np.pi * z)

        return image, prompt, mask_label, class_label, momentum_label

    @staticmethod
    def _assigned(events):
        particle_label = []
        particle_momentum = []
        for ev in events[events[:, 0] < 0]:
            particle_label.append(ev[None, 1:])
            particle_momentum.append(events[events[:, 0] == np.abs(ev[0]), 1:])
        particle = events[np.logical_and(events[:, 0] == 0, events[:, 1] == 0), 1:]

        return particle, particle_label, particle_momentum

    def __len__(self):
        return self._n_signal


def main():
    dataset = torch.utils.data.DataLoader(DatasetLoader("test"), 8, True)
    for i, data in enumerate(dataset):
        print(i + 1)


if __name__ == '__main__':
    main()
