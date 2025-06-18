import os
import pulp
import torch
import numpy as np
import torch.utils.data
import torch.multiprocessing
from generate_mask import calc_distance


class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, dir_path, include=".dat"):
        super(DatasetLoader, self).__init__()

        n = 0
        self._signal = []
        for path in [os.path.join(dir_path, path) for path in os.listdir(dir_path)]:
            if include in path:
                data = np.loadtxt(path)
                data[:, 0] = data[:, 0] + n
                n += len(np.unique(data[:, 0]))
                self._signal.append(data)
        self._signal = np.concatenate(self._signal, 0)
        np.save(os.path.join(dir_path, "signal.npy"), self._signal)

        # self._signal = np.load(os.path.join(dir_path, "signal.npy"))
        _, signal_counts = np.unique(self._signal[:, 0], return_counts=True)

        self._n_signal = len(signal_counts)
        self._signal_end = np.cumsum(signal_counts)
        self._signal_start = self._signal_end - signal_counts

        self._solver = pulp.CPLEX(threads=16, msg=False, maxMemory=20480)

    def __getitem__(self, item):
        index = item
        if index >= 0:
            events, particle_label, particle_momentum = self._assigned(self._signal[self._signal_start[index]:self._signal_end[index], 1:])

            if len(particle_label) > 0:
                particle_label = np.concatenate([np.concatenate([-(np.zeros((pl.shape[0], 1)) + i + 1), pl], -1) for i, pl in enumerate(particle_label)], 0)
                particle_momentum = np.concatenate([np.concatenate([np.zeros((pm.shape[0], 1)) + i + 1, pm], -1) for i, pm in enumerate(particle_momentum)], 0)
                
                events = np.concatenate([events[:, :1] + index, np.zeros((events.shape[0], 2)), events[:, 1:]], -1)
                particle_label = np.concatenate([np.zeros((particle_label.shape[0], 1)) + index, particle_label], -1)
                particle_momentum = np.concatenate([np.zeros((particle_momentum.shape[0], 1)) + index, particle_momentum], -1)
                events = np.concatenate([particle_label, particle_momentum, events])
                
                return np.array(events, np.float32)
            else:
                return 0
        else:
            return 0

    def _assigned(self, events):
        all_candidate = np.unique(np.abs(events[events[:, 0] < 0, 0]))

        particle = []
        particle_label = []

        nums = 10
        for n in range(all_candidate.shape[0] // nums if all_candidate.shape[0] % nums == 0 else all_candidate.shape[0] // nums + 1):
            id_candidate = all_candidate[n * nums: (n + 1) * nums]
            momentum_init = [np.sum(events[np.logical_and(events[:, 0] == idc, events[:, 1] == 0), 2:], 0) for idc in id_candidate]
            momentum_label = [np.sum(events[np.logical_and(events[:, 0] == idc, events[:, 1] != 0), 2:], 0) for idc in id_candidate]
            momentum_candidate = events[np.logical_and(events[:, 0] == 0, events[:, 1] == -1), 2:]
            distance = np.stack([calc_distance(mom_lab[None], momentum_candidate) for mom_lab in momentum_label], 0)

            task = pulp.LpProblem("task")
            error = {(x, y): pulp.LpVariable(f"error[{x}, {y}]", cat=pulp.LpContinuous) for x in range(distance.shape[0]) for y in range(4)}
            assign = {(x, y): pulp.LpVariable(f"assign[{x}, {y}]", cat=pulp.LpBinary) for x in range(distance.shape[0]) for y in range(distance.shape[1])}

            error_cost = sum(error[x, y] for x in range(distance.shape[0]) for y in range(4))
            assign_cost = sum(assign[x, y] * distance[x, y] for x in range(distance.shape[0]) for y in range(distance.shape[1]))

            task_label = np.array(momentum_label) - np.array(momentum_init)
            task_momentum = [[sum(assign[x, y] * momentum_candidate[y, i] for y in range(momentum_candidate.shape[0])) for i in range(4)] for x in range(distance.shape[0])]
            for i in range(len(task_momentum)):
                for j in range(4):
                    task.addConstraint(task_momentum[i][j] >= task_label[i, j] - error[i, j])
                    task.addConstraint(task_momentum[i][j] <= task_label[i, j] + error[i, j])

            task.setObjective(assign_cost + error_cost)

            if self._solver.solve(task) == 1:
                error = np.sum(np.array([[error[x, y].varValue for y in range(4)] for x in range(distance.shape[0])]), -1)
                assign = np.array([[assign[x, y].varValue for y in range(distance.shape[1])] for x in range(distance.shape[0])])

                momentum_assign = [momentum_candidate[param == 1] for param in assign]
                particle_momentum = [events[np.logical_and(events[:, 0] == idc, events[:, 1] == 0), 2:] for idc in id_candidate]

                particle_id = [events[np.logical_and(events[:, 0] == idc, events[:, 1] != 0), 1] for idc in id_candidate]
                particle_momentum = [np.concatenate([m, ma], 0) for m, ma in zip(particle_momentum, momentum_assign)]

                particle += [np.concatenate([np.tile(pid[None], [mom.shape[0], 1]), mom], -1) for pid, mom, err in zip(particle_id, particle_momentum, error) if mom.shape[0] > 0 and err < 5]
                particle_label += [events[np.logical_and(events[:, 0] == idc, events[:, 1] != 0), 1:] for idc, err in zip(id_candidate, error) if err < 5]

        mask_events = events[np.logical_and(events[:, 0] > 0, events[:, 1] != 0)]
        id_unique = np.unique(mask_events[[not cond for cond in np.isin(mask_events[:, 0], all_candidate)], 0])

        particle_id = [events[np.logical_and(events[:, 0] == idu, events[:, 1] != 0), 1] for idu in id_unique]
        particle_momentum = [events[np.logical_and(events[:, 0] == idu, events[:, 1] == 0), 2:] for idu in id_unique]

        particle += [np.concatenate([np.tile(pid[None], [mom.shape[0], 1]), mom], -1) for pid, mom in zip(particle_id, particle_momentum) if mom.shape[0] > 0]
        particle_label += [events[np.logical_and(events[:, 0] == idu, events[:, 1] != 0), 1:] for idu in id_unique]

        return events[np.logical_and(events[:, 0] == 0, events[:, 1] == 0), 1:], particle_label, particle

    def __len__(self):
        return self._n_signal


def trans(dir_name):
    result = []
    dataset = torch.utils.data.DataLoader(DatasetLoader(dir_name), 1, False, num_workers=16)
    for i, data in enumerate(dataset):
        if (i + 1) > 0:
            if torch.any(data):
                result.append(data[0].numpy())
                if (i + 1) % 1000 == 0:
                    result = np.concatenate(result, 0)
                    np.save(f"./{dir_name}/event_%d.npy" % ((i + 1) // 1000), result)
                    result = []
        print("\r%d/%d" % (i + 1, len(dataset)), end="")
    if len(result) > 0:
        result = np.concatenate(result, 0)
        np.save(f"./{dir_name}/event_%d.npy" % ((i + 1) // 1000), result)
    print(f"\n{dir_name}: Transform done.")


def merge(dir_name, nums):
    result = [np.load(f"./{dir_name}/event_{i + 1}.npy") for i in range(nums)]
    result = np.concatenate(result, 0)
    np.save(f"./{dir_name}/event.npy", np.array(result, np.float32))
    print(f"{dir_name}: Merge done.")


def main():
    trans("hp_h1wp_500")
    trans("hp_h1wp_1000")
    trans("hp_h1wp_1500")
    trans("hp_h1wp_500_wo_pt")
    trans("hp_h1wp_1000_wo_pt")
    trans("hp_h1wp_1500_wo_pt")
    merge("hp_h1wp_500", 10)
    merge("hp_h1wp_1000", 10)
    merge("hp_h1wp_1500", 10)
    merge("hp_h1wp_500_wo_pt", 10)
    merge("hp_h1wp_1000_wo_pt", 10)
    merge("hp_h1wp_1500_wo_pt", 10)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
