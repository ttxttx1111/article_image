import torch
import torch.utils.data as data
import os
import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import copy

class CocoDataset(data.Dataset):
    pad_length=100
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, pad_len, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.imgids = list(self.coco.imgs.keys())
        # self.wrongimgIds = copy.deepcopy(self.imgids)
        # self.wrongimgIds = np.repeat(self.wrongimgIds, 5)
        # np.random.shuffle(self.wrongimgIds)
        self.imgLen = len(self.imgids)
        self.vocab = vocab
        self.transform = transform
        CocoDataset.pad_length = pad_len

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']

        seed = torch.LongTensor([0])
        seed.random_(0, self.imgLen)
        img_id_wrong = self.imgids[seed[0]]
        # if len(self.wrongimgIds) != 0:
        #     img_id_wrong = self.wrongimgIds[0]
        #     self.wrongimgIds = np.delete(self.wrongimgIds, [0])
        while img_id_wrong == img_id:
            seed.random_(0, self.imgLen)
            img_id_wrong = self.imgids[seed[0]]
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        path_wrong = coco.loadImgs(img_id_wrong)[0]['file_name']
        image_wrong = Image.open(os.path.join(self.root, path_wrong)).convert('RGB')
        if self.transform is not None:
            image_wrong = self.transform(image_wrong)
            

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        #caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        #caption.append(vocab('<end>'))
        caption_tensor = torch.Tensor(caption)
        return image, image_wrong, caption_tensor

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, images_wrong, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    images_wrong = torch.stack(images_wrong, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), CocoDataset.pad_length).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, images_wrong, targets, lengths


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers, pad_len=30):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform,
                       pad_len=pad_len)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader