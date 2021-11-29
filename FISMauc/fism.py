import os
import json
import torch
import numpy as np
from tqdm import trange


class FISMauc:
    def __init__(self, lamb1=0.01, lamb2=0.01, lr=0.01, K=5, alpha=1):
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.lr = lr
        self.alpha = alpha
        self.beta = torch.zeros(len(TrackID), requires_grad=True)
        self.gammaP = torch.zeros((len(TrackID), K), requires_grad=True)
        self.gammaQ = torch.zeros((len(TrackID), K), requires_grad=True)
        torch.nn.init.normal_(self.gammaP, 0, 0.1)
        torch.nn.init.normal_(self.gammaQ, 0, 0.1)
        self.optimizer = torch.optim.Adam([self.beta, self.gammaP, self.gammaQ], lr=self.lr)

    def score_one(self, tracks, i):
        # given tracks(in ID), predict relative score for track i
        if i in tracks:
            tracks.remove(i)
        beta_i = self.beta[i]
        gamma_p = self.gammaP[tracks]
        gamma_q = self.gammaQ[i]
        score = beta_i + torch.sum(gamma_p * gamma_q) * len(tracks) ** (-self.alpha)
        reg = self.lamb1 * beta_i ** 2 + self.lamb2 * ((gamma_p ** 2).sum(axis=1).mean() + torch.sum(gamma_q ** 2))
        return score, reg

    def calc_loss(self, sample_playlists, sample_pos, sample_neg):
        loss = torch.tensor(0.0)
        auc = 0
        for tracks, pos, neg in zip(sample_playlists, sample_pos, sample_neg):
            score1, reg1 = self.score_one(tracks, pos)
            score2, reg2 = self.score_one(tracks, neg)
            if score1.item() > score2.item():
                auc += 1
            loss += - torch.log(torch.sigmoid(score1 - score2)) + reg1 + reg2
        return loss, auc / len(sample_playlists)

    def train(self, Nsamples=64, Nsteps=1000):
        smoothed_train_auc = smoothed_valid_auc = 0.5
        with trange(Nsteps) as t:
            for step in t:
                idxs = np.random.randint(0, len(train_data), size=Nsamples)
                sample_playlists, sample_pos, sample_neg = [], [], []
                for i in idxs:
                    tracks = train_data[i]['visible_tracks'] + train_data[i]['withheld_tracks']
                    tracks = [TrackID[t] for t in tracks]  # map to ID
                    sample_playlists.append(tracks)
                    positive_sample_idx = np.random.randint(0, len(tracks))
                    sample_pos.append(tracks[positive_sample_idx])
                    while (negative_sample_idx := np.random.randint(0, len(all_tracks))) in tracks:
                        continue
                    sample_neg.append(negative_sample_idx)

                self.optimizer.zero_grad()
                loss, train_auc = self.calc_loss(sample_playlists, sample_pos, sample_neg)
                loss.backward()
                self.optimizer.step()

                valid_auc = self.valid()
                smoothed_valid_auc = smoothed_valid_auc * 0.99 + 0.01 * valid_auc
                smoothed_train_auc = smoothed_train_auc * 0.99 + 0.01 * train_auc
                t.set_description(f'step {step + 1}')
                t.set_postfix(train_auc=smoothed_train_auc, valid_auc=smoothed_valid_auc)

    def valid(self):
        auc = 0
        cnt = 0
        idxs = np.random.randint(0, len(valid_data), size=10)
        for i in idxs:
            visible_tracks = valid_data[i]['visible_tracks']
            withheld_tracks = valid_data[i]['withheld_tracks']
            filtered_visible_tracks = [TrackID[t] for t in visible_tracks if t in all_tracks]
            filtered_withheld_tracks = [TrackID[t] for t in withheld_tracks if t in all_tracks]
            if len(filtered_visible_tracks) == 0 or len(filtered_withheld_tracks) == 0:
                continue
            positive_sample_idx = np.random.randint(0, len(filtered_withheld_tracks))
            pos = filtered_withheld_tracks[positive_sample_idx]
            while (neg := np.random.randint(0, len(all_tracks))) in filtered_visible_tracks + filtered_withheld_tracks:
                continue
            if self.score_one(filtered_visible_tracks, pos)[0] - self.score_one(filtered_visible_tracks, neg)[0] > 0:
                auc += 1
            cnt += 1
        return auc / cnt


if __name__ == '__main__':
    path_home = os.path.expanduser('~')
    path_data = os.path.join(path_home, '258/project/processed/')  # change this to fit your machine
    train_path = path_data + 'train.json'
    valid_path = path_data + 'valid.json'
    test_path = path_data + 'test.json'

    with open(train_path, 'r') as file_train:
        train_data = json.load(file_train)

    all_tracks = set()
    for playlist in train_data:
        tracks = playlist['visible_tracks'] + playlist['withheld_tracks']
        all_tracks |= set(tracks)

    with open(valid_path, 'r') as file_valid:
        valid_data = json.load(file_valid)
    with open(test_path, 'r') as file_test:
        test_data = json.load(file_test)

    all_tracks_valid = set()
    for playlist in valid_data:
        tracks = playlist['visible_tracks'] + playlist['withheld_tracks']
        all_tracks_valid |= set(tracks)

    all_tracks_test = set()
    for playlist in test_data:
        tracks = playlist['visible_tracks'] + playlist['withheld_tracks']
        all_tracks_test |= set(tracks)

    # map track uri to trackID
    TrackID = {uri: i for i, uri in enumerate(all_tracks)}
    model = FISMauc()
    model.train()
