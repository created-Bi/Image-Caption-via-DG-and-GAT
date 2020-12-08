import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder_captions, data_folder_ds, Capdata_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'DEV', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder_captions, self.split + '_IMAGES_' + Capdata_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder_captions, self.split + '_CAPTIONS_' + Capdata_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder_captions, self.split + '_CAPLENS_' + Capdata_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load triples (completely into memory)
        with open(os.path.join(data_folder_ds, self.split + '_DSTRIPLES_' + Capdata_name + '.json'), 'r') as j:
            self.triples_list = json.load(j)

        # Load distance (completely into memory)
        with open(os.path.join(data_folder_ds, self.split + '_DSRELES_' + Capdata_name + '.json'), 'r') as j:
            self.relations_list = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        triples_list = torch.LongTensor(self.triples_list[i])

        relations_list = torch.LongTensor(self.relations_list[i])

        if self.split is 'TRAIN':
            return img, caption, caplen, triples_list, relations_list
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions, triples_list, relations_list, i // self.cpi

    def __len__(self):
        return self.dataset_size
