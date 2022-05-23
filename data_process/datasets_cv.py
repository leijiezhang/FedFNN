import torch.utils.data as data


class Data(data.Dataset):
    def __init__(self, data, target, data_idxs=None,
                 transform=None, target_transform=None):
        self.data_idxs = data_idxs
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.target = self.__build_truncated_dataset__(data, target)

    def __build_truncated_dataset__(self, data_para, target_para):
        data_return = data_para
        target_return = target_para
        if self.data_idxs is not None:
            data_return = data_para[self.data_idxs]
            target_return = target_para[self.data_idxs]
        return data_return, target_return

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data_return, target = self.data[index], self.target[index]

        if self.transform is not None:
            data_return = self.transform(data_return)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data_return, target

    def __len__(self):
        return len(self.data)
