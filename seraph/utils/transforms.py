# 일단 복붙.

class VideoDataset(Dataset):
    def __getitem__(self, idx):
        video_path, start_frame, clip_label, frame_labels = self.clips[idx]

        # 저장된 클립 없이 실시간으로 읽기
        frames = read_frames(video_path, start_frame, start_frame+16)
        frames = self.transform(frames)

        return frames, clip_label, frame_labels
    


    