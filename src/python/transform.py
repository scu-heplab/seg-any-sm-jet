import os
import glob
import pulp
import torch
import argparse
import numpy as np
import torch.utils.data
import torch.multiprocessing
from tqdm import tqdm
from generate_mask import calc_distance


class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, dir_path, threads):
        super(DatasetLoader, self).__init__()

        n = 0
        self._signal = []
        paths = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".dat")])
        if not paths:
            raise FileNotFoundError(f"No '.dat' files found in the directory: {dir_path}")
        print(f"Found {len(paths)} '.dat' files to process in '{dir_path}'.")
        for path in paths:
            data = np.loadtxt(path)
            data[:, 0] = data[:, 0] + n
            n += len(np.unique(data[:, 0]))
            self._signal.append(data)
        self._signal = np.concatenate(self._signal, 0)
        # np.save(os.path.join(dir_path, "signal.npy"), self._signal)

        # self._signal = np.load(os.path.join(dir_path, "signal.npy"))
        _, signal_counts = np.unique(self._signal[:, 0], return_counts=True)

        self._n_signal = len(signal_counts)
        self._signal_end = np.cumsum(signal_counts)
        self._signal_start = self._signal_end - signal_counts

        self._solver = pulp.PULP_CBC_CMD(threads=threads, msg=False)

    def __getitem__(self, index):
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
            return None

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


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    else:
        return batch[0]


def run_transform(args):
    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else input_dir  # Default output to input dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Transform for directory: {input_dir} ---")

    dataset = DatasetLoader(dir_path=input_dir, threads=args.threads)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    chunk_index = 1
    result_chunk = []
    for i, data in enumerate(tqdm(dataloader, desc=f"Processing {os.path.basename(input_dir)}")):
        if data is not None:
            result_chunk.append(data)

        if len(result_chunk) % args.chunk_size == 0 or (i + 1) == len(dataloader):
            output_path = os.path.join(output_dir, f"{args.prefix}_{chunk_index}.npy")
            chunk_data = np.concatenate(result_chunk, 0)
            np.save(output_path, chunk_data)
            result_chunk = []
            chunk_index += 1

    print(f"--- Transform finished for {input_dir} ---")


def run_merge(args):
    input_dir = args.input_dir
    output_file = os.path.join(args.output_dir, "event.npy") if args.output_dir else os.path.join(input_dir, "event.npy")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Merge for directory: {input_dir} ---")

    chunk_files = sorted(glob.glob(os.path.join(input_dir, f"{args.prefix}_*.npy")))

    if not chunk_files:
        print(f"Error: No files found in '{input_dir}' with prefix '{args.prefix}_'. Nothing to merge.")
        return

    print(f"Found {len(chunk_files)} chunk files to merge.")

    all_data = [np.load(f) for f in tqdm(chunk_files, desc="Merging files")]

    final_data = np.concatenate(all_data, 0)

    print(f"Saving final merged file to: {output_file} ({np.unique(final_data[:, 0]).shape[0]} events)")
    np.save(output_file, final_data.astype(np.float32))
    print(f"--- Merge finished for {input_dir} ---")


def main():
    parser = argparse.ArgumentParser(description="Process '.dat' data generated by sajm. Use 'transform' then 'merge'.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # --- Parent Parser for shared arguments ---
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '-i', '--input-dir', type=str, required=True,
        help="Directory to process."
    )
    parent_parser.add_argument(
        '--prefix', type=str, default='event_chunk',
        help="Prefix for intermediate chunk files (default: event_chunk)."
    )

    # --- Transform Command ---
    parser_transform = subparsers.add_parser(
        'transform', parents=[parent_parser],
        help='Convert .dat files into chunked .npy files.'
    )
    parser_transform.add_argument(
        '-o', '--output-dir', type=str, default=None,
        help="Directory to save output chunks. Defaults to the input directory."
    )
    parser_transform.add_argument(
        '--chunk-size', type=int, default=1000,
        help="Number of events per chunk file (default: 1000)."
    )
    parser_transform.add_argument(
        '--workers', type=int, default=16,
        help="Number of DataLoader workers (default: 16)."
    )
    parser_transform.add_argument(
        '--threads', type=int, default=16,
        help="Number of threads for the MILP solver (default: 16)."
    )
    parser_transform.set_defaults(func=run_transform)

    # --- Merge Command ---
    parser_merge = subparsers.add_parser(
        'merge', parents=[parent_parser],
        help='Merge chunked .npy files into a single file.'
    )
    parser_merge.add_argument(
        '-o', '--output-dir', type=str, default=None,
        help="Full path for the final merged .npy file. Defaults to 'event.npy' inside the input directory."
    )
    parser_merge.set_defaults(func=run_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    # This setting is important for preventing "too many open files" errors
    # when using PyTorch's multiprocessing with a large number of workers.
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
