import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path


class UFC101Dataset(Dataset):
    def __init__(self, directory: Path, transform=None, num_frames=None, num_samples_use: int = 50,
                 classes_to_use: list = None, mode: str = 'train'):
        """

        :param directory:
        :param transform:
        :param num_features:
        """
        self.video_dir = directory
        self.transform = transform
        self.num_frames = num_frames
        self.classes_use = classes_to_use
        self.file_ind = os.path.join(os.path.dirname(video_dir), 'ucfTrainTestlist/classInd.txt')
        self.train_list = os.path.join(os.path.dirname(video_dir), 'ucfTrainTestlist/trainlist01.txt')
        self.num_samples_use = num_samples_use
        self.class_indixes = self.load_class_indices(self.file_ind)
        self.video_paths, self.labels = self.load_video_list(self.train_list, mode)

    def load_class_indices(self, file_ind):
        class_indices = {}
        with open(file_ind, 'r') as f:
            for line in f:
                split_line = line.strip().split(' ')
                class_indices[int(split_line[0])] = split_line[1]
        return class_indices

    def load_video_list(self, trainlist, mode):
        video_paths = []
        labels = []
        class_sample_count = {c: 0 for c in self.classes_use}
        with open(trainlist, 'r') as f:
            for line in f:
                video_path, class_index = line.strip().split(' ')
                class_index = int(class_index)

                if self.classes_use is None or class_index in self.classes_use:
                    if class_sample_count[class_index] < self.num_samples_use:
                        # Correct the path by not adding the redundant folder name
                        corrected_video_path = os.path.join(self.video_dir, video_path.split('/')[1])
                        video_paths.append(corrected_video_path)
                        labels.append(class_index)
                        class_sample_count[class_index] += 1
        return video_paths, labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx] - 1
        frames = self.load_frames_video(video_path, self.num_frames)
        if self.transform:
            frames = [self.transform(Image.fromarray(frame)) for frame in frames]
        frames = torch.stack(frames)
        sample = {'frames': frames, 'label': label}
        return sample

    def load_frames_video(self, video_path, num_frames):
        """
        Load specified number of frames from the video
        :param video_name:
        :param num_frames:
        :return:
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()
        num_extracted = len(frames)
        if num_extracted == 0:
            print(f"No frames extracted from {video_path}. Returning black frames.")
        if num_extracted < num_frames:
            repeat_times = num_frames // num_extracted + 1
            frames = (frames * repeat_times)[:num_frames]
        frame_indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
        frames = [frames[i] for i in frame_indices]
        return frames


# if __name__ == '__main__':
#     video_dir = '/home/iamshri/Documents/Dataset/UCF/Videos'
#     # file_ind_path = '/home/iamshri/Documents/Dataset/UCF/ucfTrainTestlist/classInd.txt'
#     # train_list_path = '/home/iamshri/Documents/Dataset/UCF/ucfTrainTestlist/trainlist01.txt'
#     classes_to_use = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
#     num_samples_use = 5
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor()
#     ])
#     ucf_dataset = UFC101Dataset(video_dir, transform=transform,
#                                 num_frames=16, classes_to_use=classes_to_use, mode='train')
#     batch_size = 4
#     ucf_dataloader = DataLoader(ucf_dataset, batch_size=batch_size, shuffle=True)
#     for i, batch in enumerate(ucf_dataloader):
#         frames = batch['frames']
#         labels = batch['label']
#
#         print(f'Batch {i}:')
#         print(f'Frames Shape: {frames.shape}')
#         print(f'Labels:{labels}')
