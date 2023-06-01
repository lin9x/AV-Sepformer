import math
from typing import Dict, List, Union

import numpy as np
import torch
import torch.utils.data as tdata
import soundfile

class Vox2_Dataset(tdata.Dataset):
    def __init__(self, scp_file, dur_file, visual_embed_type: str = 'resnet', batch_size: int = 4, max_duration: int = 6, sr: int = 16000):
        self._dataframe=[]
        self.batch_size = batch_size
        with open(scp_file, 'r') as f:
            with open(dur_file, 'r') as g:
                id2dur = {}
                for line in g.readlines():
                    line=line.strip().split(' ')
                    id2dur[line[0]]=int(line[1])
                for line in f.readlines():
                    line=line.strip().split(' ')
                    uid=line[0]
                    base_path = line[1]
                    mixture_path= base_path + '/mixture.wav'
                    s0_path = base_path + '/s0.wav'
                    s1_path = base_path + '/s1.wav'
                    if visual_embed_type == 'resnet':
                        c0_path = base_path + '/v0_embedding.npy'
                        c1_path = base_path + '/v1_embedding.npy'
                    elif visual_embed_type == 'avhubert':
                        c0_path = base_path + '/v0_avhubert_embed.npy'
                        c1_path = base_path + '/v1_avhubert_embed.npy'
                    else:
                        raise ValueError()
                    self._dataframe.append({'uid': uid, 'mixture': mixture_path, 's0': s0_path, 's1': s1_path, 'c0': c0_path, 'c1': c1_path, 'dur': id2dur[uid]})
        self.visual_embed_type = visual_embed_type
        self._dataframe = sorted(self._dataframe, key=lambda d: d['dur'], reverse=True)
        self._minibatch = []
        start = 0
        while True:
            end = min(len(self._dataframe), start + self.batch_size)
            self._minibatch.append(self._dataframe[start: end])
            if end == len(self._dataframe):
                break
            start = end
        self.len = len(self._minibatch)
        self._sr=sr
        self.max_duration = max_duration
        self.max_duration_in_samples = int(max_duration * sr)
        self.max_duration_in_frames = int(max_duration * 25)
        
        self.spkmap = {}
        index = 0
        for item in self._dataframe:
            spkid1, spkid2 = item['uid'].split('#')
            spkid1 = spkid1.split('+')[0]
            spkid2 = spkid2.split('+')[0]
            if spkid1 not in self.spkmap:
                self.spkmap[spkid1] = index
                index += 1
            if spkid2 not in self.spkmap:
                self.spkmap[spkid2] = index
                index += 1
           
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        batch_list = self._minibatch[index]
        min_length = batch_list[-1]['dur']
        min_length_in_second = min_length / 16000.0
        min_length_in_frame = math.floor(min_length_in_second * 25)
        mixtures = []
        sources = []
        conditions = []
        spkids = []
        for meta_info in batch_list:
            mixture, _ = soundfile.read(meta_info['mixture'], dtype='float32')
            s0, _ = soundfile.read(meta_info['s0'], dtype='float32')
            s1, _ = soundfile.read(meta_info['s1'], dtype='float32')
            c0 = np.load(meta_info['c0'])
            c1 = np.load(meta_info['c1'])
            if self.visual_embed_type == 'avhubert':
                c0 = c0.transpose((1,2,0))
                c1 = c1.transpose((1,2,0))
            mixture = mixture[:min_length]    
            s0 = s0[:min_length]
            s1 = s1[:min_length]
            mixture = np.divide(mixture, np.max(np.abs(mixture)))
            s0 = np.divide(s0, np.max(np.abs(s0)))
            s1 = np.divide(s1, np.max(np.abs(s1)))
            c0 = c0[:min_length_in_frame]
            c1 = c1[:min_length_in_frame]
            if self.visual_embed_type == 'resnet':
                if c0.shape[0] < min_length_in_frame:
                    c0 = np.pad(c0, ((0, min_length_in_frame - c0.shape[0]), (0, 0)), mode = 'edge')
                if c1.shape[0] < min_length_in_frame:
                    c1 = np.pad(c1, ((0, min_length_in_frame - c1.shape[0]), (0, 0)), mode = 'edge')
            else:
                if c0.shape[0] < min_length_in_frame:
                    c0 = np.pad(c0, ((0, min_length_in_frame - c0.shape[0]), (0, 0), (0, 0)), mode = 'edge')
                if c1.shape[0] < min_length_in_frame:
                    c1 = np.pad(c1, ((0, min_length_in_frame - c1.shape[0]), (0, 0), (0, 0)), mode = 'edge')
            '''
            if c0.shape[0] < 25 * self._duration:
                c0 = np.pad(c0, ((0, int(25 * self._duration - c0.shape[0])),(0,0),(0,0)), mode='edge')
            if c1.shape[0] < 25 * self._duration:
                c1 = np.pad(c1, ((0, int(25 * self._duration - c1.shape[0])),(0,0),(0,0)), mode='edge')
            '''
            assert c0.shape[0] == c1.shape[0]
            assert s0.shape[0] == s1.shape[0]  and mixture.shape[0] == s0.shape[0]
            spkid1, spkid2 = meta_info['uid'].split('#')
            spkid1 = self.spkmap[spkid1.split('+')[0]]
            spkid2 = self.spkmap[spkid2.split('+')[0]]

            mixtures.append(mixture[:self.max_duration_in_samples])
            mixtures.append(mixture[:self.max_duration_in_samples])
            sources.append(s0[:self.max_duration_in_samples])
            sources.append(s1[:self.max_duration_in_samples])
            conditions.append(c0[:self.max_duration_in_frames])
            conditions.append(c1[:self.max_duration_in_frames])
            spkids.append(spkid1)
            spkids.append(spkid2)
        mixtures = torch.tensor(np.array(mixtures))
        sources = torch.tensor(np.array(sources))
        conditions = torch.tensor(np.array(conditions))
        spkids =torch.tensor(spkids)
        return mixtures, sources, conditions, spkids

def dummy_collate_fn(x):
    if len(x) == 1:
        return x[0]
    else:
        return x

