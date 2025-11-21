import essentia.standard as es
import pandas as pd
import os
import numpy as np

# Beginner-friendly feature mapping
FEATURE_MAPPING = [
    ("song_name", "song_name"),
    ("metadata.audio_properties.length", "duration"),
    ("rhythm.bpm", "tempo"),
    ("rhythm.danceability", "danceability"),
    ("lowlevel.average_loudness", "loudness"),
    ("lowlevel.dynamic_complexity", "dynamic_complexity"),
    ("tonal.key_edma.key", "key"),
    ("tonal.key_edma.scale", "mode"),
    ("tonal.key_edma.strength", "key_strength"),
    ("lowlevel.mfcc.mean", "mfcc_mean"),
    ("lowlevel.melbands.mean", "melbands_mean")
]

def extract_features(file_path):
    extractor = es.MusicExtractor()
    features, _ = extractor(file_path)

    flat_feats = {}
    for ess_name, col_name in FEATURE_MAPPING:
        if ess_name == "song_name":
            flat_feats[col_name] = os.path.basename(file_path)
            continue

        # Safely access Pool elements
        try:
            val = features[ess_name]
        except KeyError:
            val = None

        # Convert arrays/lists to a single scalar
        if isinstance(val, (list, np.ndarray)):
            if len(val) == 0:
                val = None
            elif len(val) == 1:
                val = val[0]
            else:
                try:
                    val = float(np.mean(val))
                except:
                    val = str(val[0])

        # Keep scalars as is
        elif isinstance(val, (int, float, str)):
            val = val
        else:
            val = str(val)

        flat_feats[col_name] = val

    return flat_feats

def main(audio_folder="/Users/...../SongsFolder", out_csv="Viral_Music_Dataset.csv"):
    dataset = []

    for file in os.listdir(audio_folder):
        if not file.lower().endswith((".mp3", ".wav")):
            continue
        path = os.path.join(audio_folder, file)
        print(f"Processing {file}...")
        feats = extract_features(path)
        dataset.append(feats)

    df = pd.DataFrame(dataset)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Selected features saved to {out_csv}")

if __name__ == "__main__":
    main()
