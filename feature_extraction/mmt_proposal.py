# coding=utf-8

from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from glob import glob
from pathlib import Path
import argparse


class MMTDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_path_list = list(tqdm(image_dir.iterdir()))
        self.n_images = len(self.image_path_list)

        # self.transform = image_transform
        print('Load images from', image_dir)
        print('# Images:', self.n_images)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_id = image_path.stem

        img = cv2.imread(str(image_path))

        return {
            'img_id': image_id,
            'img': img
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--dataroot', type=str, default='/data/home/yc27434/projects/mmt/data/MuCoW')
    parser.add_argument('--dataset_name', type=str)

    args = parser.parse_args()

    data_root = Path(args.dataroot).resolve()
    dataset_name = args.dataset_name

    img_dir = data_root.joinpath(f'image')

    out_dir = data_root.joinpath(f'features')
    if not out_dir.exists():
        out_dir.mkdir()

    dataset = MMTDataset(img_dir)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{dataset_name}_boxes{NUM_OBJECTS}.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{(NUM_OBJECTS, DIM)}'

    extract(output_fname, dataloader, desc)
