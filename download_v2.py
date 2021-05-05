"""
Usage:
python download_v2.py \
    --dataset-file /data/datasets/kinetics-mini/train_mini.json \
    --root-dir /data/datasets/kinetics-mini \
    --batch-size 1

ouroboros-train mpirun --hostfile hostfiles/hostfile-dennis-download-1.txt --run-command "\
        python download_v2.py \
            --dataset-file kinetics700_2020/train.json \
            --root-dir /mnt/fsx/datasets/kinetics700_2020 \
            "\
        --cross-node
"""
import argparse
import json
import logging
import os

import torch
from mpi4py import MPI
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler
from tqdm import tqdm

from lib.downloader import process_video

from detectron2.utils import comm

from tridet.utils.setup import setup_distributed

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger('tridet')


class KineticsDownloader(Dataset):
    def __init__(self, dataset_file, root_dir="/mnt/fsx/datasets/kinetics700_2020"):
        with open(dataset_file, 'r') as f:
            metadata = json.load(f)

        metadata_list = []
        for _id, data in tqdm(metadata.items()):
            x = data
            x.update({"id": _id})
            metadata_list.append(x)
        self.metadata = metadata_list
        self.root_dir = root_dir
        rank = comm.get_rank()
        self._log_file = f"logs/log.rank{rank}"
        os.makedirs("logs", exist_ok=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        x = self.metadata[idx]

        label = x['annotations']['label']
        label = label.replace(' ', '-')
        start, end = x['annotations']['segment']
        # url = x['url']
        _id = x['id']

        # def process_video(video_id, directory, start, end, video_format="mp4", compress=False, overwrite=False, log_file=None):
        directory = os.path.join(self.root_dir, label)
        os.makedirs(directory, exist_ok=True)
        success = process_video(_id, directory, start, end, log_file=self._log_file)
        return success


class TrivialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.FloatTensor([0.]))

    def forward(self, x):
        return self.theta * 0.


def setup(args):
    """
    Create configs and perform basic setups.
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."

    world_size = MPI.COMM_WORLD.Get_size()
    distributed = world_size > 1

    if distributed:
        rank = MPI.COMM_WORLD.Get_rank()
        setup_distributed(world_size, rank)


def main():
    parser = argparse.ArgumentParser(description="Download Kinetics dataset.")
    parser.add_argument("--dataset-file", required=True, help="Path to Kinetics dataset JSON file (e.g. 'train.json').")
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory of target dataset (e.g. '/mnt/fsx/datasets/kinetics700_2020')."
    )
    parser.add_argument(
        "--num-workers-per-gpu", required=False, default=8, type=int, help="Number of CPU workeres per GPU."
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        default=32,
        type=int,
        help="Batch size. This only determines how often dummy backward() is called."
    )
    args = parser.parse_args()

    setup(args)

    dataset = KineticsDownloader(args.dataset_file, args.root_dir)

    model = TrivialModel().to('cuda')
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
        )
        # sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers_per_gpu,
        sampler=sampler,
        batch_size=args.batch_size,
        collate_fn=lambda x: x,
        drop_last=False,
        # batch_sampler=batch_sampler,
        # collate_fn=trivial_batch_collator,
        # pin_memory=True
    )

    success = []
    for x in tqdm(dataloader):
        loss = model(x)
        loss.backward()
        success.extend(x)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
