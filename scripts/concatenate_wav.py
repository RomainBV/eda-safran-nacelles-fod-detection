#%%
import os
import pandas as pd
from wavely.edge_metadata import db
from wavely.edge_metadata.models import Recording
import logging
import warnings
warnings.filterwarnings('ignore')
os.environ["WAVELEAKS_SETTINGS"] = 'settings.yaml'
logger = logging.getLogger("waveleaks")
logging.basicConfig(level=logging.INFO)
import datetime
import zipfile
import rarfile
import gzip
import shutil
import re 
from tqdm import tqdm
from pydub import AudioSegment
import pytz
from collections import defaultdict


path = '/media/rbeauvais/Elements/romainb/2025-n05-Safran-nacelles-FoD/'
os.chdir(path)

script_dir = path
os.chdir(script_dir)
print(script_dir)

date = '2025-04-01'
sensor = 'magneto'
sensor_id = '626'
start_time = '01:00:00'
end_time = '23:59:00'
gain = 0
timezone = 'Europe/Paris'
data_path = os.path.join('data', sensor, sensor_id, date)
metadata_path = os.path.join(data_path, 'metadata.db')

start = datetime.datetime.strptime(date+' '+start_time, '%Y-%m-%d %H:%M:%S')
end = datetime.datetime.strptime(date+' '+end_time, '%Y-%m-%d %H:%M:%S')

# recording_type = 'continuous'
recording_type = 'continuous'


def estimate_wav_size(audio):
    """Estime la taille du fichier WAV exporté en octets."""
    num_samples = len(audio.get_array_of_samples())  # Nombre total d'échantillons
    num_channels = audio.channels  # Mono (1) ou Stéréo (2)
    sample_width = audio.sample_width  # Largeur d'échantillon (ex: 2 octets pour 16-bit)

    return num_samples * num_channels * sample_width  # Taille estimée en octets


def get_recordings_from_time_range(metadata_path: str, start_time: datetime, end_time: datetime,recording_type: str = 'continuous') -> list:
    """
    Get filenames and start_time of recordings in a time range.

    Args:
        metadata_path (str): Path of the sqlite db.
        start_time (datetime): Start time from where to get the recordings.
        end_time (datetime): End time from where to get the recordings.

    Returns:
        List[Recording]: Recordings list with infos from metadata.db.
    """
    db_url = "sqlite:///{}".format(metadata_path)
    db.initialize(url=db_url)

    with db.create_session() as session:
        logger.debug(f"Getting recordings from {start_time} to {end_time}")
        recordings_query = session.query(
            Recording
        ).filter(
            Recording.start_time > start_time
        ).filter(
            Recording.start_time < end_time
        ).filter(
            Recording.recording_type == recording_type,
        )

        t0 = [recording.start_time for recording in recordings_query]
        start_times = pd.DataFrame(index=[recording.filename for recording in recordings_query], data={"start_times": t0})

    recordings = []
    valid_filenames = []

    for recording in recordings_query:
        wav_file_path = os.path.join(os.path.dirname(metadata_path), recording.filename)
        if os.path.exists(wav_file_path):
            recordings.append(recording)
            valid_filenames.append(recording.filename)
        else:
            logger.debug(f"File {wav_file_path} not found, removing from the list.")

    start_times = start_times.loc[valid_filenames]

    return recordings, db_url, start_times

def decompress_and_remove_files(folder_path):
    """
    Decompresses all zip, rar, and gz files in the given folder and deletes them after decompression.
    Continues to the next file if an error occurs during decompression.

    Args:
        folder_path (str): Path to the folder containing the compressed files.
    """
    decompressed_files = []
    
    for filename in tqdm(os.listdir(folder_path), desc='Decompressing files'):
        file_path = os.path.join(folder_path, filename)
        
        try:
            if filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(folder_path)
                    decompressed_files.extend(zip_ref.namelist())
                os.remove(file_path)

            elif filename.endswith('.gz'):
                decompressed_file_path = os.path.splitext(file_path)[0]
                with gzip.open(file_path, 'rb') as f_in:
                    with open(decompressed_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        decompressed_files.append(decompressed_file_path)
                os.remove(file_path)

            elif filename.endswith('.rar'):
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    rar_ref.extractall(folder_path)
                    decompressed_files.extend(rar_ref.namelist())
                os.remove(file_path)

        except Exception as e:
            print(f"Error decompressing file {filename}: {e}")
            continue

    print(f"Decompression completed. Total files processed: {len(decompressed_files)}")
    return decompressed_files

def sort_key(path):
    # Extract sequence number if exists
    match = re.search(r'_(\d+)\.wav$', path)
    if match:
        return (path.rsplit('_', 1)[0], int(match.group(1)))
    else:
        return (path, 0)


def concatenate_wav_files_by_group(output_dir, input_paths, start_times, gain, timezone='Europe/Paris'):
    """
    Concatène des fichiers WAV par groupes basés sur un préfixe de nom commun,
    applique un gain, convertit les fichiers 32-bit en 16-bit, et crée des batches
    respectant une durée maximale par groupe.

    Args:
        input_paths (list): Liste des chemins des fichiers WAV d'entrée.
        start_times (DataFrame): DataFrame contenant les temps de début des fichiers (indexés par noms de fichiers).
        output_dir (str): Répertoire où exporter les fichiers batchs.
        max_duration_ms (int): Durée maximale en millisecondes pour un batch.
        gain (float): Gain à appliquer aux fichiers audio.
        timezone (str): Fuseau horaire pour les noms de fichiers exportés.
    """
    tz = pytz.timezone(timezone)

    # Grouper les fichiers par préfixe commun
    groups = defaultdict(list)
    for path in input_paths:
        base_name = os.path.basename(path)
        match = re.match(r'(\d+_\d+)', base_name)
        if match:
            group_name = match.group(1)
            groups[group_name].append(path)

    # Traiter chaque groupe séparément
    for group_name, files in groups.items():
        combined = AudioSegment.empty()
        current_batch_duration = 0
        batch_files = []
        batch_count = 0

        # Trier les fichiers par leur temps de début
        files = sorted(files, key=lambda p: start_times.loc[os.path.basename(p), 'start_times'])

        for path in tqdm(files, desc=f'Processing group {group_name}'):
            try:
                sound = AudioSegment.from_wav(path)
                current_sample_width = sound.sample_width
                file_duration = len(sound)

                if current_sample_width == 4:
                    sound = sound.apply_gain(gain)
                    sound = sound.set_sample_width(2)
                else:
                    print(f"File {path} is not in 32-bit format. Ignored.")
                    continue

                # Vérifier si l'ajout dépasse la durée maximale
                if current_batch_duration + file_duration > MAX_DURATION_MS:
                    if batch_files:
                        export_batch(output_dir, batch_files, combined, start_times, tz, group_name, batch_count)
                        batch_count += 1

                    # Réinitialiser le batch
                    combined = AudioSegment.empty()
                    current_batch_duration = 0
                    batch_files = []

                combined += sound
                current_batch_duration += file_duration
                batch_files.append(path)

            except Exception as e:
                print(f"Error processing file {path}: {e}")

        # Exporter le dernier batch du groupe
        if batch_files:
            export_batch(output_dir, batch_files, combined, start_times, tz, group_name, batch_count)

def concatenate_wav_files_by_start_times(output_dir,input_paths, start_times, gain, tz):
    """
    Découpe et exporte les fichiers WAV par batchs de 15, en positionnant sur une base silencieuse.

    :param input_paths: Liste des chemins des fichiers WAV
    :param start_times: DataFrame contenant les temps de début des fichiers
    :param output_dir: Répertoire de sortie
    :param gain: Gain audio appliqué aux fichiers
    """
    os.makedirs(output_dir, exist_ok=True)

    # Trier les fichiers par start_time
    input_paths = sorted(input_paths, key=lambda p: start_times.loc[os.path.basename(p), 'start_times'])

    batch_count = 0
    num_files = len(input_paths)
    
    for i in tqdm(range(0, num_files, BATCH_SIZE), desc="Processing batches"):
        batch_files = input_paths[i:i+BATCH_SIZE]  

        # Déterminer la durée du segment silencieux
        start_time_1 = start_times.loc[os.path.basename(batch_files[0]), 'start_times'].timestamp() * 1000
        end_time_last = start_times.loc[os.path.basename(batch_files[-1]), 'start_times'].timestamp() * 1000
        duration_last = len(AudioSegment.from_wav(batch_files[-1]))
        batch_duration = (end_time_last + duration_last) - start_time_1

        # Créer un segment de silence avec cette durée
        combined = AudioSegment.silent(duration=int(batch_duration))

        for path in batch_files:
            try:
                sound = AudioSegment.from_wav(path)
                current_sample_width = sound.sample_width
                file_start_time = start_times.loc[os.path.basename(path), 'start_times']

                if current_sample_width == 4:
                    sound = sound.apply_gain(gain)
                    sound = sound.set_sample_width(2)
                else:
                    print(f"File {path} is not in 32-bit format. Ignored.")
                    continue

                # Positionner le fichier dans `combined`
                position = file_start_time.timestamp() * 1000 - start_time_1
                combined = combined.overlay(sound, position=int(position))

            except Exception as e:
                print(f"Erreur avec le fichier {path}: {e}")

        # Exporter le batch
        export_batch(output_dir, batch_files, combined, start_times, tz, None, None)
        batch_count += 1

    # max_start_time = start_times['start_times'].max().timestamp() * 1000  
    # min_start_time = start_times['start_times'].min().timestamp() * 1000 
    # max_duration_ms = int((max_start_time - min_start_time) + (3 * 60 * 1000)) 

    # for path in tqdm(input_paths, desc='Processing file', unit='file'):
    #     combined = AudioSegment.silent(duration=max_duration_ms)  # Création d'un segment silencieux

    #     sound = AudioSegment.from_wav(path)
    #     current_sample_width = sound.sample_width
    #     file_start_time = start_times.loc[os.path.basename(path), 'start_times']

    #     if current_sample_width == 4:
    #         sound = sound.apply_gain(gain)
    #         sound = sound.set_sample_width(2)
    #     else:
    #         print(f"File {path} is not in 32-bit format. Ignored.")
    #         continue

    #     # Position files on the right time scale
    #     position = int(file_start_time.timestamp()* 1000  - min_start_time)
    #     combined = combined.overlay(sound, position=position)

    #     print(path)

    # export_batch(output_dir, input_paths, combined, start_times, tz, None, None)





def export_batch(output_dir, batch_files, combined, start_times, tz, group_name, batch_count):
    first_file_path = batch_files[0]
    first_start_time = start_times.loc[os.path.basename(first_file_path), 'start_times']
    first_start_time = first_start_time.astimezone(tz)
    if batch_count:
        output_filename = f"{first_start_time.strftime('%Y-%m-%d_%H:%M:%S')}_batch_{batch_count + 1}.wav"
    else:
        output_filename = f"{first_start_time.strftime('%Y-%m-%d_%H:%M:%S')}.wav"
    output_path = os.path.join(output_dir, output_filename)

    # Créer le dossier pour le batch
    batch_name = os.path.splitext(output_filename)[0]
    data_folder = os.path.join(output_dir, 'labels', batch_name + '_data')
    os.makedirs(data_folder, exist_ok=True)

    # Exporter le fichier WAV
    combined.export(output_path, format='wav')
    if batch_count:
        print(f"Exported batch {batch_count + 1} of group {group_name} to {output_path}")
    else:
        print(f"Exported file {output_filename}")

if not recording_type ==' continuous':
    BATCH_SIZE = 10 
MAX_DURATION_MS = 1800000

decompress_and_remove_files(data_path)

recordings, db_url, start_times = get_recordings_from_time_range(metadata_path, start, end, recording_type)

# Obtenez les chemins de fichiers à partir des enregistrements
input_paths = [os.path.join(os.path.dirname(metadata_path), recording.filename) for recording in recordings]
output_dir = os.path.join(data_path,'long_format')
os.makedirs(output_dir, exist_ok=True)
if recording_type == 'continuous':
    concatenate_wav_files_by_group(output_dir, input_paths, start_times, gain, timezone)
else:
    concatenate_wav_files_by_start_times(output_dir,input_paths, start_times,gain, timezone)

# %%