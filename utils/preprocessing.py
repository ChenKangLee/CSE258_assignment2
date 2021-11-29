import os
import json
import random


def filter_data(path_data, splits, keeping_lengths, path_output, holdout=10, dump=False):
    """ Preprocess data to match the format for our problem
        Filter the data to keep only playlists with length in the range `keeping_length`.
        Also generates first_n and random_n holdouts.
    """

    filtered_seq = []
    filtered_rand = []

    for filename in tqdm(os.listdir(path_data)):
        full_path = os.path.join(path_data, filename)
        with open(full_path, 'r') as file:
            mpd_slice = json.load(file)
            
        for playlist in mpd_slice['playlists']:
            pid = playlist['pid']
            
            if len(playlist['tracks']) in keeping_lengths:
                tracks = [t['track_uri'] for t in playlist['tracks']]
                
                seq_entry = {
                    'pid': pid,
                    'visible_tracks': list(tracks[:holdout]),
                    'withheld_tracks': list(tracks[holdout:]),
                }
                
                random.shuffle(tracks)
                
                rand_entry = {
                    'pid': pid,
                    'visible_tracks': list(tracks[:holdout]),
                    'withheld_tracks': list(tracks[holdout:]),
                }
            
                filtered_seq.append(seq_entry)
                filtered_rand.append(rand_entry)
    
    if dump:
        for suffix, dataset in zip(["", "_rand"], [filtered_seq, filtered_rand]):
            splits = [8, 1, 1]
            breakpoints = [len(dataset) // sum(splits) * n for n in splits]

            train = dataset[:breakpoints[0]]
            valid = dataset[breakpoints[0]:breakpoints[0] + breakpoints[1]]
            test = dataset[-breakpoints[2]:]
            
            for s, d in zip(['train', 'valid', 'test'], [train, valid, test]):
                path_dataset = os.path.join(path_output, f'{s}{suffix}.json')
                with open(path_dataset, 'w') as f:
                    json.dump(d, f)
    
    return filtered_seq, filtered_rand


if __name__ == '__main__':
    random.seed(2021)

    path_home = os.path.expanduser('~')
    path_data = os.path.join(path_home, 'spotify_mpd/data')
    path_output = os.path.join(path_home, 'spotify_mpd/processed')

    splits = [8, 1, 1]
    keeping_lengths = [25, 26, 27, 28, 29, 30]

    filter_data(path_data, splits, keeping_lengths, path_output, holdout=10, dump=True)
    return