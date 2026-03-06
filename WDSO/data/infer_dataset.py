import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class InferDataset(BaseDataset):
    """only used for cyclegan test process
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # opt.phase == "train"
        
        # self.dir_A = os.path.join(opt.datarootA, f"{opt.phase}A")
        # self.dir_B = os.path.join(opt.datarootB, f"{opt.phase}B")
        self.dir_A = os.path.join(opt.datarootA, f"trainA")
        self.dir_B = os.path.join(opt.datarootB, f"trainB")
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        assert self.A_size > 0

        self.input_nc = 3
        self.output_nc = 3

    def __getitem__(self, index):
        A_path = self.A_paths[index]  # make sure index is within then range
        #B_path = self.B_paths[index]

        A = Image.open(A_path).convert('RGB')
        #B = Image.open(B_path).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        #B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        #B = B_transform(B)


        return {'A': A, 'B': A, 'A_paths': A_path, 'B_paths': A_path}# A和B长度不一样，B测试的时候用不上，给的A的patch

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size
