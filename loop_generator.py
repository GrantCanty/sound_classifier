import pandas as pd
import numpy as np
import h5py
import scipy.io.wavfile as wavf
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class LoopGenerator:
    def __init__(self, swing=.55, velocity_var=.1, note_var=2/12, dir_path="loops", bpm_min=70, bpm_max=180): # swing of .5 is neutral
        self.swing = swing
        self.velocity_var = velocity_var
        self.note_var = note_var
        self.dir_path = dir_path
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self.cat_and_sub_cat_id = int(0)
        self.new_sub_cat_files = []
        self.new_sub_cat_audio = []
        self.generated_files_metadata = pd.DataFrame()
        self.generated_files_audio = pd.DataFrame()
        self.old_files = pd.DataFrame()

        try:
            self.existing_files_df = pd.read_csv('audio_metadata - loops.csv', sep=',')
        except:
            self.existing_files_df = pd.DataFrame()
            self.existing_files = False
            self.id = int(0)
        else:
            self.existing_files = True
            self.id = int(self.existing_files_df['id'].max())


    def safety_guard(self, pattern: list, audio: list, sr: list, category: list, sub_category: list):
        # count of audio files should match count of sample rates
        if len(audio) != len(sr):
            raise Exception('Different counts of audio and sample rates')
        
        # different sample rates
        if len(np.unique(sr)) > 1:
            raise Exception('Differing sample rates detected')
        sample_rate = sr[0]

        if (len(np.unique(category))) >1:
            raise Exception('Differing categories detected')
        cat = category[0]
        
        if (len(np.unique(sub_category))) >1:
            raise Exception('Differeing sub categories detected')
        sub_cat = sub_category[0]
        
        # pattern cannot be empty
        if len(pattern) < 8:
            raise Exception(f'Pattern is too small. Length of {len(pattern)} is < 8 steps')
        
        return sample_rate, cat, sub_cat
    

    def bpm(self):
        return np.random.randint(self.bpm_min, self.bpm_max)


    def generate_loops(self, samples, patterns, num_of_loop_samples, cat):
        self.old_files = pd.DataFrame()
        self.new_sub_cat_files= []
        self.new_sub_cat_audio = []
        pat_type = [cat, "Global"]
        
        sub_cats = samples['sub_category'].unique()
        for sub_cat in sub_cats:
            self.cat_and_sub_cat_id = 0
            if pd.isna(sub_cat):
                samples_split = samples[samples['sub_category'].isna()].copy()
                if self.existing_files:
                    vals = self.existing_files_df[self.existing_files_df['file_path'].str.contains(os.path.join(self.dir_path, cat), case=False, na=False)]
            else:
                samples_split = samples[samples['sub_category'] == sub_cat].copy()
                if self.existing_files:
                    vals = self.existing_files_df[self.existing_files_df['file_path'].str.contains(os.path.join(self.dir_path, cat, sub_cat), case=False, na=False)]
            
            samples_split.reset_index(drop=False, inplace=True)
            if self.existing_files:
                vals = vals['file_path'].str.split('_', expand=True)
                vals = vals[len(vals.columns)-1].str.extract(r'(\d+)\.')[0].dropna().astype(int)
                self.cat_and_sub_cat_id = vals.max() if not vals.empty else 0
            
            nums = []
            for index, _ in samples_split.iterrows():
                nums.append(index)
                if index % num_of_loop_samples != 0:
                    pat_choice = np.random.randint(len(pat_type))
                    which = pat_type[pat_choice]

                    pattern = patterns[which][index % len(patterns[which])]
                    samples_split.loc[nums[0]:nums[-1]+1, 'Loop_id'] = int(self.id)
                    rows = samples_split[nums[0]:nums[-1]+1]

                    self.generate_loop(pattern, rows)
                    nums = []
            
            self.old_files = pd.concat([self.old_files, samples_split], ignore_index=True)
        
        if isinstance(self.new_sub_cat_files, list):
            self.new_sub_cat_files = pd.DataFrame(self.new_sub_cat_files)
        if isinstance(self.new_sub_cat_audio, list):
            self.new_sub_cat_audio = pd.DataFrame(self.new_sub_cat_audio)

        self.generated_files_metadata = pd.concat([self.generated_files_metadata, self.new_sub_cat_files], ignore_index=True)
        self.generated_files_audio = pd.concat([self.generated_files_audio, self.new_sub_cat_audio], ignore_index=True)
    

    # edit og dataframe to show the loop id of each sample
    def generate_loop(self, pattern, rows):
        audio = rows['waveform'].to_list()
        sr = rows['sample_rate'].to_list()
        cats = rows['category'].to_list()
        sub_cats = rows['sub_category'].to_list()
        
        sample_rate, cat, sub_cat = self.safety_guard(pattern, audio, sr, cats, sub_cats)
        
        num_of_beats = 2 ** np.random.randint(2, 5)
        bpm = self.bpm()
        beat_time = 60 / bpm  # seconds per beat
        loop_duration = beat_time * num_of_beats
        loop_samples = int(np.ceil(loop_duration * sample_rate))
        new_loop = np.zeros(loop_samples, dtype=np.float32)

        for bar in range(int(num_of_beats/4)):
            for i, hit in enumerate(pattern):
                if hit > 0:
                    for j in range(0, hit):
                        velocity = (1-self.velocity_var) * np.random.random_sample() + self.velocity_var
                        sample_idx = np.random.randint(len(audio))
                        sample = audio[sample_idx]
                        sample = sample * velocity
                        
                        total_step = i + (bar * len(pattern))
                        start_idx = int((total_step * sample_rate * beat_time / 4) + ((sample_rate / hit) * j))
                        end_idx = start_idx + len(sample)
                        
                        if end_idx > len(new_loop):
                            diff = end_idx - len(new_loop)
                            z = np.zeros(diff)
                            new_loop = np.append(new_loop, z)

                        new_loop[start_idx:end_idx] += sample
        
        new_loop = self.soft_limit(new_loop)
        rows_copy = rows.drop(columns=['waveform', 'sample_rate'], inplace=False)
        self.save_loop(sample_rate, new_loop, cat, sub_cat, rows_copy)


    def soft_limit(self, loop, threshold=.96, max_val=1):
        abs_x = np.abs(loop)
        sign_x = np.sign(loop)
        
        over = abs_x > threshold
        zeroed_vals = abs_x[over] - threshold
        scaled_vals = 1 - np.exp(1)**-zeroed_vals
        scaled_vals = scaled_vals * (max_val - threshold)
        scaled_vals = scaled_vals + threshold
        
        loop[over] = sign_x[over] * scaled_vals
        return loop
    

    def save_loop(self, sr, loop, category, sub_category, rows):
        os.makedirs(self.dir_path, exist_ok=True)
        os.makedirs(os.path.join(self.dir_path, category), exist_ok=True)
        
        loop = (loop * (2 ** 15 - 1)).astype("<h")
        if pd.isna(sub_category):
            file_name = f'{category}_{self.cat_and_sub_cat_id}.wav'
            file_path = os.path.join(self.dir_path, category, file_name)
        else:
            os.makedirs(os.path.join(self.dir_path, category, sub_category), exist_ok=True)
            file_name = f'{sub_category}_{category}_{self.cat_and_sub_cat_id}.wav'
            file_path = os.path.join(self.dir_path, category, sub_category, file_name)
        wavf.write(file_path, sr, loop)
        
        new_row = rows.iloc[0].to_dict()
        new_row['id'] = self.id
        new_row['file_path'] = file_path
        new_row['file_name'] = file_name
        new_row['One_Shot'] = 0
        new_row['Loop'] = 1
        for key in ['Loop_id', 'One_Shot_Intent', 'level_0', 'index']:
            new_row.pop(key, None)
        
        new_audio_row = {}
        new_audio_row['id'] = self.id
        new_audio_row['sample_rate'] = sr
        new_audio_row['waveform'] = loop

        self.new_sub_cat_files.append(new_row)
        self.new_sub_cat_audio.append(new_audio_row)

        self.id += 1
        self.cat_and_sub_cat_id += 1

    
    def save_updates(self):
        if len(self.generated_files_metadata) >0:
            if self.existing_files and not self.generated_files_metadata.empty:
                self.generated_files_metadata = pd.concat([self.existing_files_df, self.generated_files_metadata], ignore_index=True)
            self.generated_files_metadata.to_csv('audio_metadata - loops.csv', sep=',', index=False)
            
            if not self.generated_files_metadata.empty:
                dt = h5py.string_dtype(encoding='utf-8')
                h5_path = "generated_loop_data.h5"
                
                mode = "r+" if os.path.exists(h5_path) else "w"
                with h5py.File(h5_path, mode) as f:
                    if "meta_data" not in f:
                        meta_data = f.create_group("meta_data")
                        meta_data.create_dataset('index', data=np.array(self.generated_files_metadata['id'], dtype='int'))
                        meta_data.create_dataset('path', data=np.array(self.generated_files_metadata['file_path'], dtype=dt))
                        meta_data.create_dataset('name', data=np.array(self.generated_files_metadata['file_name'], dtype=dt))
                    else:
                        # Optional: could update existing metadata datasets here if desired
                        pass

                    # Get/create audio_data group
                    if "audio_data" not in f:
                        audio_data = f.create_group("audio_data")
                    else:
                        audio_data = f["audio_data"]

                    def save_sample(row):
                        sample_id = str(row['id'])
                        if sample_id in audio_data:
                            # Avoid overwriting existing data (can log if needed)
                            return
                        sample_group = audio_data.create_group(sample_id)
                        sample_group.create_dataset('waveform', data=row['waveform'], compression='gzip')
                        sample_group.create_dataset('sample_rate', data=row['sample_rate'])

                    # Use ThreadPoolExecutor for speed
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        list(tqdm(
                            executor.map(save_sample, [row for _, row in self.generated_files_audio.iterrows()]),
                            total=len(self.generated_files_audio),
                            desc="Saving new audio to HDF5"
                        ))