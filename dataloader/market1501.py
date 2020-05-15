import re
import glob
import os.path as osp
import warnings
from torch.utils.data import Dataset
from .data_tools import read_image


class Market1501(Dataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'market1501'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', part='', transform=None, **kwargs):
        # dir ----------------------------------------------------------------
        self.data_dir = osp.abspath(osp.expanduser(root))
        assert osp.isdir(
            self.data_dir), 'The current data structure is deprecated.'

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        # data ----------------------------------------------------------------
        assert part in {'train', 'query', 'gallery'}, 'part not in folders'
        if part == 'train':
            train = self.process_dir(self.train_dir, relabel=True)
            self.data = train
            self.num_train_pids = self.get_num_pids(self.data)
            self.num_train_cams = self.get_num_cams(self.data)
        if part == 'query':
            query = self.process_dir(self.query_dir, relabel=False)
            self.data = query
        if part == 'gallery':
            gallery = self.process_dir(self.gallery_dir, relabel=False)
            self.data = gallery
        # transform ------------------------------------------------------------
        self.transform = transform

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
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

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.

        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(cams)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)[0]

    def get_num_cams(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[1]
