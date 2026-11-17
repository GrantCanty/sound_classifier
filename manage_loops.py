import pandas as pd
import numpy as np
import librosa
import h5py
import math
import warnings
import json
from loop_generator import LoopGenerator
from pathlib import Path
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime

one_to_loop_ratio = 1
loop_unit_multi_sample_pct = 0.2

# import current metadata file
import_path = 'audio_metadata - filtered'
meta_data = pd.read_csv(import_path + '.csv', sep=',', index_col=False)
print(f'{datetime.now().isoformat(' ')} [MANAGE LOOPS] imported {import_path}.csv')

# import patterns for different drum types
f = open('patterns.json')
looping_patterns = json.load(f)
print(f'{datetime.now().isoformat(' ')} [MANAGE LOOPS] imported patterns.json')

# find all available paths for audio samples
FOLDER_PATH = "/Users/cheoso/Documents/samples/Packs/"
file_paths = []
total = 0

for root, dirs, files in os.walk(FOLDER_PATH):
    for file in files:
        path = Path(root) / file
        path_lower = str(path).lower()
        if path_lower.endswith(('.wav', '.aif', '.aiff')):
          file_paths.append(str(path))

          total += path.stat().st_size

print(f'{datetime.now().isoformat(' ')} [MANAGE LOOPS] found {len(file_paths)} audio file(s)')

# creating paths variable
meta_data['file_path'] = meta_data['file_path'].apply(lambda x: '/users/cheoso/documents/samples/packs/' + x)
paths = np.array(meta_data['file_path']).flatten()
paths_set = set(paths)

paths_to_import = [
    any_path for any_path in file_paths
    if any_path.lower() in paths_set
]

# load the files
failed_files = []
divided_by_zero = []

class SilenceError(Exception):
    """Raised when an audio file is silent (max amplitude is 0)."""
    pass

def process_audio(file, audio, sr):
    try:
        rs = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        if abs(rs.max()) == 0:
            raise SilenceError(f'Silent file detected: {file}')
        return rs / abs(rs.max()), 44100
    except Exception as e:
        print(f'Error processing file {file}: {e}')
        raise

def hash_waveform_with_id(row):
    waveform = row
    waveform_hash = hashlib.sha256(waveform.tobytes()).hexdigest()
    return waveform_hash

def extract_metadata(index, file):
    """Load an audio file and extract its waveform, sample rate, and label."""
    try:
        audio, sr = librosa.load(file, sr=None)  # Load without resampling
        audio, sr = process_audio(file, audio, sr)
        audio_hash = hash_waveform_with_id(audio)
        file = str(file).lower()
        #file_name = file.split('/')[-1]
        return [file, audio_hash, audio, sr, index]
    except SilenceError:
        divided_by_zero.append([index, file])
        return None
    except Exception as e:
        failed_files.append(file)
        return None

# Use parallel processing to load audio files
# add tqdm counter
def get_audio_and_metadata(files):
    file_path = []
    audio_hash = []
    wavs = []
    srs = []
    ids = []

    with ThreadPoolExecutor() as executor:
        results = tqdm(executor.map(lambda p: extract_metadata(*p), enumerate(files)), total=len(files))
        
        for res in results:
            if res is not None:
                #meta_data = res
                file_path.append(res[0])
                audio_hash.append(res[1])
                wavs.append(res[2])
                srs.append(res[3])
                ids.append(res[4])
    
    samples = pd.DataFrame({'file_path': file_path, 'audio_hash': audio_hash, 'waveform': wavs, 'sample_rate': srs, 'id': ids})
    return samples

# call the function to load the files
samples = get_audio_and_metadata(paths_to_import)
print(f'{datetime.now().isoformat(' ')} [MANAGE LOOPS] loaded {len(samples)} samples')

# merge samples with metadata
samples = samples.loc[:, ['file_path', 'audio_hash', 'waveform', 'sample_rate']]
samples = pd.merge(samples, meta_data, on='file_path')

# function to get the sample counts
def get_sample_counts(samples):
    category_counts = {}
    sample_categories = set(samples['category'])
    for cat in sample_categories:
        category_counts[cat] = {}
        category_counts[cat]["loop_count"] = len(samples[(samples['category'] == cat) & (samples['One_Shot'] == 0)])
        category_counts[cat]["one_shot_count"] = len(samples[(samples['category'] == cat) & (samples['One_Shot'] == 1)])
        if 'One_Shot_Intent' and 'Loop_id' in samples.columns:
            category_counts[cat]['one_shot_intent_count'] = len(samples[(samples['category'] == cat) & (samples['One_Shot'] == 1) & (samples['One_Shot_Intent'] == 1)])
            category_counts[cat]['loop_intent_count'] = len(samples[(samples['category'] == cat) & (samples['Loop_Intent'] == 1) & (samples['One_Shot'] == 1) ])
        else:
            category_counts[cat]['one_shot_intent_count'] = 0
            category_counts[cat]['loop_intent_count'] = 0
        
        category_counts[cat]['all'] = category_counts[cat]["one_shot_count"] + category_counts[cat]["loop_count"]
        category_counts[cat]['samples_to_convert_to_loops'] = math.floor((category_counts[cat]["one_shot_count"] - category_counts[cat]["loop_count"]) *  (1-(1/(2+loop_unit_multi_sample_pct))) )
        category_counts[cat]['required_one_shot_count'] = category_counts[cat]["one_shot_count"] - category_counts[cat]['samples_to_convert_to_loops']
        category_counts[cat]['required_loop_count'] = category_counts[cat]["loop_count"] + math.floor(category_counts[cat]['samples_to_convert_to_loops'] * (1 / (1+loop_unit_multi_sample_pct)))
        category_counts[cat]['required_loop_count_sample'] = category_counts[cat]['samples_to_convert_to_loops'] + category_counts[cat]["loop_count"]
        category_counts[cat]['new_loops_to_add'] = math.floor(category_counts[cat]['samples_to_convert_to_loops'] / (1+loop_unit_multi_sample_pct))
        category_counts[cat]['unused_with_loop_intent'] = len(samples[(samples['category'] == cat) & (samples['Loop_Intent'] == 1) & (samples['Loop_id'].isnull()) & (samples['One_Shot'] == 1)])

        category_counts[cat]['count_of_samples_for_loop_generation'] = category_counts[cat]['samples_to_convert_to_loops'] - (category_counts[cat]['loop_intent_count'] - category_counts[cat]['unused_with_loop_intent'])
    
    return category_counts

category_counts = get_sample_counts(samples)

# clean merge function that is used when merging multiple dataframes together
def clean_merge(df1, df2):
    merged = df1.merge(
            df2[['id', 'One_Shot_Intent', 'Loop_Intent', 'Loop_id']],
            on='id',
            how='left',
            suffixes=('', '_new')
        )

    for col in ['One_Shot_Intent', 'Loop_Intent', 'Loop_id']:
        merged[col] = merged[f'{col}_new'].combine_first(merged[col])
        merged.drop(columns=[f'{col}_new'], inplace=True)

    return merged

# create instance of loop generator and copy the samples variable so data is not overwritten accidentally
gen = LoopGenerator(save_loops=False)
samples_copy = samples.copy()

# get necessary counts of samples from ratios
num_of_loop_units = int(one_to_loop_ratio / loop_unit_multi_sample_pct)
num_of_loop_samples = math.ceil((one_to_loop_ratio + loop_unit_multi_sample_pct) / loop_unit_multi_sample_pct)

def generate_samples(category_counts, samples_copy, num_of_loop_samples):
    updated_rows = []
    # if one shot count is greater than needed and 
    for cat in category_counts:
        # count for intents meet the required counts - no edits need to be made
        if (category_counts[cat]['samples_to_convert_to_loops'] == 0) and (category_counts[cat]['one_shot_count'] >= category_counts[cat]['required_one_shot_count']) and (category_counts[cat]['loop_count'] >= category_counts[cat]['required_loop_count']):
            print(f'one shot intent and loop intent counts for {cat} meet requirements')
        # 0 count for both intents - just created intent col
        elif category_counts[cat]['one_shot_count'] > category_counts[cat]['required_one_shot_count'] and category_counts[cat]['one_shot_intent_count'] == 0 and category_counts[cat]['loop_intent_count'] == 0 :
            print(f'one shot intent and loop intent counts for {cat} do not meet requirements')
            '''tmp = samples_copy[(samples_copy['category'] == cat) & (samples_copy['One_Shot'] == 1)].copy()
            
            ones = tmp.iloc[:category_counts[cat]['required_one_shot_count']].copy()
            ones['One_Shot_Intent'] = 1
            ones['Loop_Intent'] = 0
            
            existing_loops = samples_copy[(samples_copy['category'] == cat) & (samples_copy['One_Shot'] == 0)].copy()
            loops = tmp.iloc[category_counts[cat]['required_one_shot_count']:(category_counts[cat]['required_one_shot_count']+category_counts[cat]['required_loop_count_sample'])+1].copy()

            existing_loops['One_Shot_Intent'] = 0
            existing_loops['Loop_Intent'] = 1
            
            loops['One_Shot_Intent'] = 0
            loops['Loop_Intent'] = 1
            loops_no_idx = loops.reset_index(drop=False)
            
            gen.generate_loops(loops_no_idx, looping_patterns, num_of_loop_samples, cat)
            tmp = clean_merge(tmp, gen.old_files)

            tmp = clean_merge(tmp, ones)

            tmp = tmp[~(
                (tmp['One_Shot_Intent'].isna() | (tmp['One_Shot_Intent'] == '')) &
                (tmp['Loop_id'].isna() | (tmp['Loop_id'] == '')) &
                (tmp['Loop_Intent'].isna() | (tmp['Loop_Intent'] == ''))
            )]

            tmp = pd.concat([tmp, existing_loops], ignore_index=True, axis=0, join='outer')
            updated_rows.append(tmp)'''

        # non 0 count exists for intents. intents already existing - dataset has done some loop generation already and new files were probably added
        elif (category_counts[cat]['one_shot_count'] > category_counts[cat]['required_one_shot_count']) and (category_counts[cat]['one_shot_intent_count'] < category_counts[cat]['required_one_shot_count']) and (category_counts[cat]['loop_intent_count'] < category_counts[cat]['required_loop_count']):
            tmp = samples_copy[samples_copy['category'] == cat].copy()
            
            # get ones that have not been used for anything. get number of samples needed to create a loop
            unused_ones = tmp[(tmp['One_Shot'] == 1) & (tmp['One_Shot_Intent'].isnull())].copy()
            ones_used_for_loops = unused_ones.copy()
            ones_used_for_loops = ones_used_for_loops.reset_index(drop=False)
            ones_used_for_loops = ones_used_for_loops.iloc[:category_counts[cat]['count_of_samples_for_loop_generation'] - category_counts[cat]['unused_with_loop_intent']]

            # get created loops
            used_loop_intent = tmp[(tmp['Loop_Intent'] == 1) & (tmp['Loop_id'].notna())]

            # get any loops that are set to be created but have not been created
            unused_loop_intent = tmp[(tmp['Loop_Intent'] == 1) & (tmp['Loop_id'].isnull()) & (tmp['One_Shot'] == 1)].copy()
            unused_loop_intent = unused_loop_intent.reset_index(drop=False)

            # variable for all loops to be created. check that it is equal to the number of samples needed for loop creation
            create_new_loops = pd.concat([unused_loop_intent, ones_used_for_loops])
            assert len(create_new_loops) == category_counts[cat]['count_of_samples_for_loop_generation']
            
            # assign correct intent to the loops
            create_new_loops['One_Shot_Intent'] = 0
            create_new_loops['Loop_Intent'] = 1
            
            # generate the new loops and merge the loops to the tmp variable
            gen.generate_loops(create_new_loops, looping_patterns, num_of_loop_samples, cat)
            tmp = clean_merge(tmp, gen.old_files)

            # one shot samples that are used as a one shot
            used_ones = tmp[(tmp['One_Shot'] == 1) & (tmp['One_Shot_Intent'] == 1)]

            # get the rest of the one shots that were unused and use for them as one shots
            remaining_ones = unused_ones.copy()
            remaining_ones = remaining_ones.reset_index(drop=False)
            remaining_ones = remaining_ones.iloc[category_counts[cat]['count_of_samples_for_loop_generation'] - category_counts[cat]['unused_with_loop_intent']:]
            
            # assign one shot intent. merge to tmp variable
            remaining_ones['One_Shot_Intent'] = 1
            remaining_ones['Loop_Intent'] = 0
            tmp = clean_merge(tmp, remaining_ones)

            # check that the newly added one shots and existing one shots equal the required amount of one shot samples
            assert (len(remaining_ones) + len(used_ones)) == category_counts[cat]['required_one_shot_count']
            #all_ones = pd.concat([used_ones, remaining_ones], ignore_index=True, axis=0, join='outer')

            # get existing loops and ensure they have the correct intent. merge them to tmp. check that loop sample count equal require loop sample count
            existing_loops = samples_copy[(samples_copy['category'] == cat) & (samples_copy['One_Shot'] == 0)].copy()
            existing_loops['One_Shot_Intent'] = 0
            existing_loops['Loop_Intent'] = 1
            tmp = clean_merge(tmp, existing_loops)
            assert (len(used_loop_intent) + len(create_new_loops) + len(existing_loops)) == category_counts[cat]['required_loop_count_sample']

            updated_rows.append(tmp)
            
        elif category_counts[cat]['required_one_shot_count'] > category_counts[cat]['one_shot_count']:
            warnings.warn(f"not enough one shots to reach {one_to_loop_ratio}:1 ratio")
        
        else:
            print('in else statement')


    if updated_rows:
        all_updates = pd.concat(updated_rows)
        samples_copy = clean_merge(samples_copy, all_updates)

        samples_copy = samples_copy[~(
                (samples_copy['One_Shot_Intent'].isna() | (samples_copy['One_Shot_Intent'] == '')) &
                (samples_copy['Loop_id'].isna() | (samples_copy['Loop_id'] == '')) &
                (samples_copy['Loop_Intent'].isna() | (samples_copy['Loop_Intent'] == ''))
            )]
    
    return samples_copy

samples_copy = generate_samples(category_counts, samples_copy, num_of_loop_samples)
print(f'{datetime.now().isoformat(' ')} [MANAGE LOOPS] generated loops. {len(samples_copy)} samples left')

#gen.generated_files_metadata
#samples_copy

from pprint import pprint

category_counts = {}
sample_categories = set(samples_copy['category'])
for cat in sample_categories:
    category_counts[cat] = {}
    category_counts[cat]["loop_count"] = len(samples_copy[(samples_copy['category'] == cat) & (samples_copy['One_Shot'] == 0)])
    category_counts[cat]["one_shot_count"] = len(samples_copy[(samples_copy['category'] == cat) & (samples_copy['One_Shot'] == 1)])
    if 'One_Shot_Intent' and 'Loop_id' in samples_copy.columns:
        category_counts[cat]['one_shot_intent_count'] = len(samples_copy[(samples_copy['category'] == cat) & (samples_copy['One_Shot'] == 1) & (samples_copy['One_Shot_Intent'] == 1)])
        category_counts[cat]['loop_intent_count'] = len(samples_copy[(samples_copy['category'] == cat) & (samples_copy['Loop_Intent'] == 1) & (samples_copy['One_Shot'] == 1) ])
    else:
        category_counts[cat]['one_shot_intent_count'] = 0
        category_counts[cat]['loop_intent_count'] = 0
    
    category_counts[cat]['all'] = category_counts[cat]["one_shot_count"] + category_counts[cat]["loop_count"]
    category_counts[cat]['samples_to_convert_to_loops'] = math.floor((category_counts[cat]["one_shot_count"] - category_counts[cat]["loop_count"]) *  (1-(1/(2+loop_unit_multi_sample_pct))) )
    category_counts[cat]['required_one_shot_count'] = category_counts[cat]["one_shot_count"] - category_counts[cat]['samples_to_convert_to_loops']
    category_counts[cat]['required_loop_count'] = category_counts[cat]["loop_count"] + math.floor(category_counts[cat]['samples_to_convert_to_loops'] * (1 / (1+loop_unit_multi_sample_pct)))
    category_counts[cat]['new_loops_to_add'] = math.floor(category_counts[cat]['samples_to_convert_to_loops'] / (1+loop_unit_multi_sample_pct))
    category_counts[cat]['unused_with_loop_intent'] = len(samples_copy[(samples_copy['category'] == cat) & (samples_copy['Loop_Intent'] == 1) & (samples_copy['Loop_id'].isnull()) & (samples_copy['One_Shot'] == 1)])

    category_counts[cat]['count_of_samples_for_loop_generation'] = category_counts[cat]['samples_to_convert_to_loops'] - (category_counts[cat]['loop_intent_count'] - category_counts[cat]['unused_with_loop_intent'])

pprint(category_counts)

def save_changes(gen, samples_copy):
    gen.save_updates()
    print(f'{datetime.now().isoformat(' ')} [MANAGE LOOPS] generated new loops metadata and h5 files')

    non_gen_loops = samples_copy.dropna(subset=['Loop_id'])
    non_gen_loops = samples_copy.drop(non_gen_loops.index).reset_index(drop=True)

    # non_gen_loops.head()

    dt = h5py.string_dtype(encoding='utf-8')
    h5_path = "filtered_audio_data.h5"

    with h5py.File(h5_path, "w") as f:
        h5_meta_data = f.create_group("meta_data")
        h5_meta_data.create_dataset('index', data=np.array(non_gen_loops['id'], dtype='int'))
        h5_meta_data.create_dataset('path', data=np.array(non_gen_loops['file_path'], dtype=dt))
        h5_meta_data.create_dataset('name', data=np.array(non_gen_loops['file_name'], dtype=dt))

        # Get/create audio_data group
        audio_data = f.create_group("audio_data")

        def save_sample(row):
            sample_id = str(row['id'])
            sample_group = audio_data.create_group(sample_id)
            sample_group.create_dataset('waveform', data=row['waveform'], compression='gzip')
            sample_group.create_dataset('sample_rate', data=row['sample_rate'])

        # Use ThreadPoolExecutor for speed
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(
                executor.map(save_sample, [row for _, row in non_gen_loops.iterrows()]),
                total=len(non_gen_loops),
                desc="Saving new audio to HDF5"
            ))

    f.close()

    samples_copy.drop(columns=['waveform', 'sample_rate'], inplace=True)
    samples_copy.to_csv(import_path + '_copy.csv', sep=',', index=False, encoding='utf-8-sig')
    print(f'{datetime.now().isoformat(' ')} [MANAGE LOOPS] generated new one shots metadata and h5 files')


# save_changes(gen, samples_copy)
