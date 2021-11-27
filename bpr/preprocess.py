import os
import json
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def preprocess_bpr(path_data, train_valid_split=[8,2]):
    """ Preprocess the Spotify mpd dataset.
        Currently using only the first 1/10 of the whole dataset (100,000 playlists).

        This function also convert the track_uri into indices.
    """

    assert(len(train_valid_split) == 2)

    # build dictionary of playlist and their tracks to sample for pairs
    track_per_playlist = defaultdict(set)
    all_tracks = set()

    print("Loading data playlist from .json files...")
    for filename in tqdm(os.listdir(path_data)[:100]):
        full_path = os.path.join(path_data, filename)
        with open(full_path, 'r') as file:
            mpd_slice = json.load(file)

        for playlist in mpd_slice['playlists']:
            pid = playlist['pid']

            for track in playlist['tracks']:
                track_uri = track['track_uri']

                track_per_playlist[pid].add(track_uri)
                all_tracks.add(track_uri)


    all_track_list = list(all_tracks)
    track_uri_to_id = {uri: i for i, uri in enumerate(all_track_list)}

    # save to file to reuse the indexing in the training step
    np_all_track = np.array(all_track_list, dtype=str)

    # "negatives" are sampled in random
    print("Building negative data...")
    dataset = []
    for pl in tqdm(track_per_playlist):
        for track in track_per_playlist[pl]:
            neg = random.choice(all_track_list) # not guranteeing negative

            track_id = track_uri_to_id[track]
            neg_id = track_uri_to_id[neg]

            dataset.append([pl, track_id, neg_id])

    np_dataset = np.array(dataset)
    np_train = np_dataset[:int(len(dataset) * train_valid_split[0] / sum(train_valid_split))]
    np_valid = np_dataset[int(len(dataset) * train_valid_split[0] / sum(train_valid_split)):]

    return np_all_track, np_train, np_valid


if __name__ == '__main__':
    path_home = os.path.expanduser('~')
    path_data = os.path.join(path_home, 'spotify_mpd/data') # change this to fit your machine

    all_tracks, train, valid = preprocess_bpr(path_data, [8, 2])

    np.savetxt('track_ids.txt', all_tracks, fmt='%s')
    np.savetxt('train.txt', train, fmt='%u')
    np.savetxt('valid.txt', valid, fmt='%u')