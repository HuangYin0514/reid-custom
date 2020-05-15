import os.path as osp
import glob

from torch.utils.data import Dataset
from data_tools import read_image


class Occluded_REID(Dataset):

    def __init__(self, root='', part='', transform=None, **kwargs):
        # dir ----------------------------------------------------------------
        self.data_dir = osp.abspath(osp.expanduser(root))
        assert osp.isdir(
            self.data_dir), 'The current data structure is deprecated.'
        self.query_dir = osp.join(self.data_dir, 'occluded_body_images')
        self.gallery_dir = osp.join(self.data_dir, 'whole_body_images')
        # data ----------------------------------------------------------------
        assert part in {'train', 'query', 'gallery'}, 'part not in folders'
        self.data = []
        if part == 'query':
            query = self.process_dir(self.query_dir)
            self.data = query
        if part == 'gallery':
            gallery = self.process_dir(self.gallery_dir, is_query=False)
            self.data = gallery
        assert len(self.data) != 0, 'Data is None. please check data'
        # transform ------------------------------------------------------------
        self.transform = transform

    def process_dir(self, dir_path, relabel=False, is_query=True):
        img_paths = glob.glob(osp.join(dir_path, '*', '*.tif'))
        if is_query:
            camid = 0
        else:
            camid = 1
        pid_container = set()
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))
        return data

    def __getitem__(self, index):
        img_path, pid, camid = self.data[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

    def __len__(self):
        return len(self.data)
