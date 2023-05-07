from typing import Optional
import os
import pretty_midi
import pandas as pd
import collections
import numpy as np
from matplotlib import pyplot as plt


def midi_to_notes(pm) -> pd.DataFrame:
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'First {count} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(
        plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    _ = plt.title(title)


def vectorize(in_path: str):
    loading_bar_dict = {
        0: "--------->",
        1: "=-------->",
        2: "==------->",
        3: "===------>",
        4: "====----->",
        5: "=====---->",
        6: "======--->",
        7: "=======-->",
        8: "========->",
        9: "=========>"

    }
    

    pm = pretty_midi.PrettyMIDI(in_path)
    tempo = pm.estimate_tempo()
    end_time = pm.get_end_time()

    vectorized_music = np.zeros((600, 88,), dtype=np.float32)
    notes = midi_to_notes(pm)

    t = 1
    dt = (60 / tempo) / 4
    loading_portion = end_time / 10
    portion = -1
    while t <= end_time:
        if portion != loading_bar_dict[int(t // loading_portion)]:
            print(loading_bar_dict[int(t // loading_portion)])
        portion = loading_bar_dict[int(t // loading_portion)]
        for index, pitch, start, end in zip(range(len(notes['pitch'])), notes['pitch'], notes['start'],
                                            notes['end']):
            if index == 600:
                break
            if start <= t <= end:
                vectorized_music[index][pitch - 20] = 1
            if t > end_time:
                break

        t += dt
    return vectorized_music


def read_vec(vectorized):
    running_notes = []
    for time_point in vectorized:
        for i in running_notes:
            if time_point[i] == 0:
                print(time_point[i])
                running_notes.remove(i)

        for index, key in enumerate(time_point):
            if key == 1:
                running_notes.append(index)
        if running_notes == []:
            continue
        print(running_notes)



def plot_vec(vectorized):
    print("test")
    all_running_notes = []
    running_notes = []
    for beats, time_point in enumerate(vectorized):
        for i in running_notes:
            if time_point[i] == -1:
                print("test")
            if time_point[i] == 0 or time_point[i] == -1:
                running_notes.remove(i)

        for index, key in enumerate(time_point):
            if key == 1:
                running_notes.append(index)
        all_running_notes += running_notes

    beats = [i for i in range(0, len(all_running_notes))]

    fig, ax = plt.subplots(1, 1)

    plt.rcParams["figure.figsize"] = (20, 5)
    ax.plot(beats, all_running_notes, 'bo')
    
def plot_vec_old(vectorized):
    
    all_running_notes = []
    running_notes = []
    for beats,time_point in enumerate(vectorized):
        for i in running_notes:
            if time_point[i][0] == 0:

                running_notes.remove(i)
            
        for index,key in enumerate(time_point):
            if key[0] == 1:
                running_notes.append(index)
        all_running_notes += running_notes

    beats = [i for i in range(0,len(all_running_notes))]
    
    fig,ax = plt.subplots(1,1)
    
    plt.rcParams["figure.figsize"]=(25, 5)
    ax.plot(beats,all_running_notes,'bo')
    ax.set_xlabel("beats")
    ax.set_ylabel("pitch")

def plot_vec(vectorized):
    all_running_notes = []
    running_notes = []
    for beats, time_point in enumerate(vectorized):
        for i in running_notes:
            if time_point[i] == 0 or time_point[i] == -1:
                running_notes.remove(i)

        for index, key in enumerate(time_point):
            if key == 1:
                running_notes.append(index)
        all_running_notes += running_notes

    beats = [i for i in range(0, len(all_running_notes))]

    fig, ax = plt.subplots(1, 1)

    plt.rcParams["figure.figsize"] = (20, 5)
    ax.plot(beats, all_running_notes, 'bo')

def make_segments(vectorized_music, grain=1):
    for i in range(vectorized):
        pass


def cut_to_size(vectorized_music, beats_amnt):
    vectorized_music.resize((beats_amnt, 88), refcheck=False)


def fix_issue_sample(sample):
    new_sample = np.zeros((600,88,),dtype=np.float32)
    for time_ind in range(len(sample)):
        for key in range(0,88):
            new_sample[time_ind][key] = sample[time_ind][key][0]
    return new_sample



if __name__ == '__main__':
    count = 0
    midi_list = os.listdir('midi_files')
    for index, mid in enumerate(midi_list):
        print(f"progress on document {index}: ")
        try:
            vectorized = vectorize(f'midi_files/{mid}')
            count += 1
        except Exception as e:
            print(e)
            print("skipped")
            continue
        np.save(f'np_music_files/np_full_{count}.npy', vectorized)
        del vectorized
