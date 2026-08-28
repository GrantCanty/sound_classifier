
import pandas as pd
import os
from pathlib import Path
import librosa
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import hashlib
import logging
import argparse
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser(
    prog='audio metadata training file generator',
    description='generate training csv file from the audio directory and a reference file. contains labels/metadata for audio files that are used for training a ML model'
)
parser.add_argument('--directory')
parser.add_argument('--reference')
parser.add_argument('--output')

args = parser.parse_args()
dir = args.directory
ref = args.reference
out = args.output

dir_path = Path(dir)
dir_path = (Path('~')/ dir_path).expanduser()
if not dir_path.exists():
    logger.error(f'--directory path {dir_path} does not exist. Exiting program')
    sys.exit()
lower_dir_path = str(dir_path).lower()

ref_path = Path(ref)
ref_path = (Path('~')/ ref_path).expanduser()
if not ref_path.exists():
    logger.error(f'--reference path {ref_path} does not exist. Exiting program')
    sys.exit()

out_path = Path("data" , "csv" , out)
if out_path.suffix.lower() != ".csv":
    if out_path.suffix:
        logger.warning(f"--output path {out} is not a .csv file. Updating extension")
    out_path = out_path.with_suffix('.csv')
    
if out_path.exists():
    response = input(f"--output file '{out_path}' already exists. Overwrite? (y/n): ").strip().lower()
    if response not in ('y', 'yes'):
        logger.info("Overwrite cancelled by user. Exiting program.")
        sys.exit()

file_paths = []
total_byte_count = 0

logger.info("Finding files in directory")
for root, dirs, files in os.walk(dir_path):
    for file in files:
        path = Path(root) / file
        path_lower = str(path).lower()
        if path_lower.endswith(('.wav', '.aif', '.aiff')):
          file_paths.append(str(path))

          total_byte_count += path.stat().st_size

logger.info(f"Total bytes of all files in directory: {total_byte_count}")

failed_files = []
divided_by_zero = []

def process_audio(audio, sr):
    rs = librosa.resample(audio, orig_sr=sr, target_sr=44100)
    if rs.max() == 0:
        divided_by_zero.append(sr)
    else:
        return rs / abs(rs.max())
    return rs

def hash_waveform_with_id(row):
    waveform = row
    waveform_hash = hashlib.sha256(waveform.tobytes()).hexdigest()
    return waveform_hash

def extract_metadata(index, file):
    """Load an audio file and extract its waveform, sample rate, and label."""
    try:
        audio_waveform, sr = librosa.load(file, sr=44100)  # Load without resampling
        audio = process_audio(audio_waveform, sr)
        audio = hash_waveform_with_id(audio)
        file = str(file)
        file_name = file.split('/')[-1]
        return [index, file, file_name, audio]
    except Exception as e:
        failed_files.append(file)
        return None

# Use parallel processing to load audio files
def get_audio_and_metadata(files):
    meta_results = []

    with ThreadPoolExecutor() as executor:
        results = tqdm(executor.map(lambda p: extract_metadata(*p), enumerate(files)), total=len(files))
        
        for res in results:
            if res is not None:
                meta_data = res
                meta_results.append(meta_data)
    
    return meta_results

logger.info("Loading files in directory")
metadataResults = get_audio_and_metadata(file_paths)

#### STATS ON LOADED FILES ####
failed_len = len(failed_files)
divide_by_zero_len = len(divided_by_zero)
total_file_paths = len(file_paths)
loaded_files = len(metadataResults)

logger.warning(f"Count of files that failed to load: {failed_len}")
logger.warning(f"Count of empty files: {divide_by_zero_len}")
logger.info(f"Total files found: {total_file_paths}")
logger.info(f"Total files loaded: {loaded_files}")
#### END STATS ON LOADED FILES ####

# Convert results to DataFrame
df_md = pd.DataFrame(metadataResults, columns=["id", "file_path", "file_name", "audio_hash"])

# load old csv file with categorizations
logger.info("Loading reference file")
reference_df = pd.read_csv('ref_2.csv')

# only keep data that has been classified
filtered_results = reference_df[reference_df['group'].notna()].copy()
filtered_results['is_used'] = True

# split the path and make it all lower case to match the paths in filtered results
df_md['file_path'] = df_md['file_path'].apply(lambda x: x.lower().split(lower_dir_path)[1])

# join data together. adds the is used column
logger.info("Merging Loaded and reference files")
joined_df = df_md.merge(filtered_results, on='audio_hash', how='left')
joined_df.drop(columns=[col for col in joined_df.columns if '_y' in col], inplace=True)
joined_df.rename(columns={col: col.split('_x')[0] for col in joined_df.columns if '_x' in col}, inplace=True)
joined_df_len = len(joined_df)
logger.info(f"Joined file count: {joined_df_len}")

# fill in missing values
joined_df['is_used'].astype(bool)
joined_df.loc[joined_df['is_used'].isna(), 'is_used'] = False

# sort df ensuring that used files are first, drop any duplicates while keeping the first listed audio 
# file (ensures that any duplicates that are in use are kept while all other are deleted)
logger.info("Removing duplicated Loaded audio files. Keeping used files first")
joined_df = joined_df.sort_values(by='is_used', ascending=False).drop_duplicates(subset='audio_hash', keep='first', inplace=False)
logger.info(f"Joined file final count: {len(joined_df)}")

try:
    joined_df.drop(columns=['Unnamed: 52'], inplace=True)
except Exception as e:
    None

assert len(set(df_md['audio_hash'])) == len(joined_df)
joined_df.drop(columns=['is_used'], inplace=True)
joined_df.sort_values(by='file_path').to_csv(out_path, header=True, index=False, encoding='utf-8-sig')
logger.info(f"Successfully saved file to {out_path}")