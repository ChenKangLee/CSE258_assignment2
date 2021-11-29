import os
import json

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
    print(len(train_data), len(all_tracks))
