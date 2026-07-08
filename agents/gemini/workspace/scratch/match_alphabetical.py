import os
import numpy as np
import pandas as pd

# Load YIN results
summary = pd.read_csv("agents/gemini/workspace/pitch_yin_mapping.csv")

reliable = summary[summary['std_midi'] < 0.1]
print(f"Number of reliable classes: {len(reliable)}")
print(reliable[['Pitch_ID', 'median_midi', 'std_midi']])

# Let's write a script to find if there is a permutation of MIDI notes that matches
# standard note names sorted alphabetically.
note_names_sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
note_names_flat = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Let's try many different naming styles
naming_styles = []

# Style 1: standard sharp/flat with optional separator
for sep in ["", "_", " "]:
    for flat in [False, True]:
        naming_styles.append(('standard', flat, sep))

# Style 2: just notes (no octave) - wait, if there are 82 classes, they must have octaves
# Style 3: SciPy/Librosa note names
# Librosa note names use sharp by default, e.g. "C#1", "D-1" (for negative octaves)

for start_midi in range(12, 48): # MIDI 12 is C0, MIDI 48 is C3
    end_midi = start_midi + 82
    midi_range = list(range(start_midi, end_midi))
    
    for style_type, flat, sep in naming_styles:
        # Generate names
        names = []
        for m in midi_range:
            octave = (m // 12) - 1
            note_idx = m % 12
            note = note_names_flat[note_idx] if flat else note_names_sharp[note_idx]
            names.append(f"{note}{sep}{octave}")
            
        # Sort names alphabetically
        sorted_indices = np.argsort(names)
        
        diffs = []
        for _, row in reliable.iterrows():
            pid = int(row['Pitch_ID'])
            est_midi = row['median_midi']
            alpha_midi = midi_range[sorted_indices[pid]]
            diffs.append(abs(est_midi - alpha_midi))
            
        mean_diff = np.mean(diffs)
        if mean_diff < 1.5:
            print(f"Start: {start_midi}, flat: {flat}, sep: '{sep}', Mean Diff: {mean_diff:.4f}")
