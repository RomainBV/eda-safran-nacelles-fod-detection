#%%
import os
import glob
import re 
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
from matplotlib.ticker import EngFormatter, MaxNLocator, LogLocator, LogFormatter
from matplotlib.dates import date2num
import soundfile as sf
import seaborn as sns

from scipy.signal import butter, sosfiltfilt
from scipy.signal import find_peaks
from scipy.stats import pearsonr


from wavely.edge_metadata import db
from wavely.edge_metadata.models import Recording
from wavely.signal.preprocessing import preprocessing as sp
from wavely.signal.features.features import FeaturesComputer
from wavely.signal.units.helpers import split_signal
from wavely.waveleaks.offline_evaluation import helpers as offline_evaluation_helpers
from wavely.waveleaks.conf import settings
from wavely.ml import features as ml_features
import logging

from typing import Optional, Sequence, Tuple, Union
import datetime
import pytz
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')
os.environ["WAVELEAKS_SETTINGS"] = 'settings.yaml'
logger = logging.getLogger("waveleaks")
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)


formatter_freq = EngFormatter(unit="Hz")
Scalar = Union[int, float]
_NOMINAL_FREQ = {
    band_type: sp.octave_bands_frequencies(band_type)
    for band_type in ["octave", "third"]
}


# start_times_list = ['11:28:00']
start_times_list = ['11:28:00','12:28:00','12:58:00','14:45:00','15:15:00']

###################
### User inputs ###
###################
for start_time in start_times_list:
    path = '/media/rbeauvais/Elements/romainb/2025-n05-Safran-nacelles-FoD/'
    date = '2025-04-01'
    frequency_scale = 'log'
    tz = 'Europe/Paris'
    recording_type = 'continuous'
    compute_leak_detection = False
    audacity_label_name = 'Labels 1.txt'

    # channel = 1
    # sensor = 'magneto'
    # sensor_id = '626'
    channel = 3
    sensor = 'zoom-f4'
    sensor_id = ''
    start_time = start_time
    end_time = '15:50:00'
    window_rolling = 1
    block_duration = 0.12
    format_long = True

    features_list = ['spectralcentroid', 'spectralspread', 'highfrequencycontent', 'spectralcrest', 'spectralentropy', 'spectralflatness', 'spectralirregularity', 'spectralkurtosis', 'spectralskewness', 'spectralrolloff', 'zero_crossing_rate', 'ultrasoundlevel','spectralflux', 'temporal_kurtosis', 'audiblelevel', 'crest_factor', 'rms', 'peakfreq']

    campaign_name = 'SAFRAN Nacelles - FoD'
    band_freq = 50
    freq_limits = [1000,96000]
    emergence = False
    Q_factor = False
    display_features = False
    display_leak = False
    display_spectra = False
    save_data = True
    bandleq_features = {}



    #filtering
    apply_filter = False
    btype = 'high'  # Type de filtre ('low', 'high', 'band')features
    cutoff_freqs = [1000]  # Fréquences de coupure haute et basse (Hz)
    order = 2  # Ordre du filtre

    term_replacement = []
    label_removals = {}    
    ##################
    ### Processing ###
    ##################

    os.chdir(path)
    print(path)


    #######################################
    ### Functions for datasets-analysis ###
    #######################################

    def audacity_label_csv_formatting(label_file_path: str,
                                    start_times: pd.Series,
                                    term_replacements: list[tuple[str, str]],
                                    label_removals: list[str],
                                    format_long: bool = False,
                                    timezone: str = 'Europe/Paris') -> pd.DataFrame:
        """
        Formats a label file from Audacity into a CSV format and applies various transformations to labels.

        Args:
            label_file_path (str): Path to the input label file.
            start_times (pd.Series): Series containing start times with the file name as index.
            term_replacements (list[tuple[str, str]]): List of tuples containing old and new terms for label replacement.
            label_removals (list[str]): List of substrings to be removed from labels.
            format_long (bool): Whether to format the timestamps using the long format (with date and time). Default is False.
            timezone (str): The timezone to be applied for the timestamps. Default is 'Europe/Paris'.

        Returns:
            pd.DataFrame: A DataFrame containing the formatted labels.
        """

        label_output_path = os.path.join(os.path.split(label_file_path)[0], "labels.csv")

        with open(label_file_path, 'r') as file:
            lines = file.readlines()

        filtered_lines = [line for line in lines if not line.startswith('\\')]
        data = [line.strip().split('\t') for line in filtered_lines]
        df_labels = pd.DataFrame(data, columns=["Start", "End", "Label"])

        df_labels["Start"] = df_labels["Start"].astype(float)
        df_labels["End"] = df_labels["End"].astype(float)


        # Determine the starting time t0 because Audacity annotations is in relative time
        folder_path = os.path.dirname(label_file_path)
        if format_long:
            pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})'
            match = re.search(pattern, folder_path)
            datetime_str = match.group(0)
            t0 = datetime.datetime.strptime(datetime_str, "%Y-%m-%d_%H:%M:%S")
        else:
            datetime_str = folder_path.split('/')[-1].split("_data")[0] + '.wav'

            if datetime_str not in start_times.index:
                print(f"Warning: '{datetime_str}' not found in start_times. Skipping this file.")
                return None  
            t0 = start_times.loc[datetime_str, 'start_times']

        # Localize timezone if not already set
        if t0.tzinfo is None:
            t0 = pd.Timestamp(t0).tz_localize(timezone)

        df_labels["Start"] = df_labels["Start"].apply(lambda x: t0 + datetime.timedelta(seconds=x))
        df_labels["End"] = df_labels["End"].apply(lambda x: t0 + datetime.timedelta(seconds=x))

        if term_replacements:
            for old_term, new_term in term_replacements:
                df_labels["Label"] = df_labels["Label"].str.replace(old_term, new_term)

        if label_removals:
            mask = ~df_labels["Label"].apply(lambda x: any(removal in x for removal in label_removals))
            df_labels = df_labels[mask]

        df_labels.to_csv(label_output_path, index=False)
        print(f"Updated label file: {label_output_path}")

        return df_labels

    def butter_filter(data: np.ndarray,
                    cutoff_freqs: list[float],
                    fs: float,
                    btype: str = 'low',
                    order: int = 5) -> np.ndarray:
        """
        Applies a Butterworth filter to a signal.

        Args:
            data (np.ndarray): The input signal, typically a 1D or 2D array.
            cutoff_freqs (list[float]): Cutoff frequency or frequencies. If 'btype' is 'low' or 'high', 
                                        provide a single cutoff frequency [fc]. If 'btype' is 'band', 
                                        provide two frequencies [f_low, f_high].
            fs (float): The sampling frequency of the signal in Hz.
            btype (str, optional): The type of filter: 'low' for low-pass, 'high' for high-pass, 
                                'band' for band-pass. Default is 'low'.
            order (int, optional): The order of the filter. A higher order results in a steeper cutoff slope. 
                                Default is 5.

        Returns:
            np.ndarray: The filtered signal as a NumPy array.
        """
        # Nyquist frequency (half the sampling rate)
        nyquist = 0.5 * fs

        # Normalization
        if btype == 'band' and len(cutoff_freqs) != 2:
            raise ValueError("For a band-pass filter ('band'), 'cutoff_freqs' must contain two frequencies.")
        
        normal_cutoff = [freq / nyquist for freq in cutoff_freqs]

        # Butterworth filter
        sos = butter(order, normal_cutoff, btype=btype, output='sos')

        # Apply the filter using sosfiltfilt to avoid phase distortion
        filtered_data = sosfiltfilt(sos, data, axis=0)

        return filtered_data

    def calculate_q_factor(frequencies: np.ndarray, 
                        magnitudes: np.ndarray, 
                        peak_idx: int, 
                        band_freq: float) -> float:
        """
        Calculates the Q-factor of a peak in a frequency spectrum.

        Args:
            frequencies (np.ndarray): Array of frequency values.
            magnitudes (np.ndarray): Array of magnitude values corresponding to the frequencies.
            peak_idx (int): Index of the peak in the magnitude array.
            band_freq (float): Bandwidth frequency for Q-factor calculation.

        Returns:
            float: The Q-factor of the peak.
        """
        peak_mag = magnitudes[peak_idx]
        half_max = peak_mag / 2

        # Find the index to the left of the peak where the amplitude falls below half-height
        left_idx = peak_idx
        while left_idx > 0 and magnitudes[left_idx] >= half_max:
            left_idx -= 1

        # Linear interpolation to find the exact frequency at half-height to the left
        if left_idx < peak_idx:
            f_left = np.interp(
                half_max, 
                magnitudes[left_idx:left_idx + 2][::-1], 
                frequencies[left_idx:left_idx + 2][::-1]
            )
        else:
            return np.nan  # Not enough data to the left for interpolation

        # idem on the right side
        right_idx = peak_idx
        while right_idx < len(magnitudes) - 1 and magnitudes[right_idx] >= half_max:
            right_idx += 1


        if right_idx > peak_idx:
            f_right = np.interp(
                half_max, 
                magnitudes[right_idx - 1:right_idx + 1], 
                frequencies[right_idx - 1:right_idx + 1]
            )
        else:
            return np.nan  

        # Calculate bandwidth Δf
        bandwidth = f_right - f_left  # Δf
        
        # Calculate Q-factor
        q_factor = band_freq / bandwidth

        return q_factor

    # function from "plot" in the "datasets-analysis" module (modified because of poor legend management)
    def plot_feature_dispersion(
        dataframes: Sequence[pd.DataFrame],
        feature: str,
        min: float = 0.25,
        max: float = 0.75,
        mid: float = 0.5,
        logx: bool = False,
        colors: Sequence[str] = None,
        labels: Sequence[str] = None,
        grid: bool = True,
        fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
        language: str = "french",
        output_file: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a feature's dispersion across time.

        The function plots the variability of a feature over a certain amount of time by
        computing percentile values. The dispersion is computed frequency-wise, that is,
        the percentile is computed over all feature values for each frequency point.
        Time and frequency steps are inherited from the DataFrame ones.

        Args:
            dataframes: A list of 2d dataframes indexed by "time" and "freq".
            feature: The name of the feature whose dispersion will be plotted.
            min: Float in [0-1), it is the p-th percentile, used to define the lower bound
                of the dispersion. Defaults to .25.
            max: Float in (0-1], it is the p-th percentile, used to define the upper bound
                of the dispersion. Defaults to .75.
            mid: Float in (0-1), it is the p-th percentile, used to define an "average"
                spectrum value. Defaults to .5.
            logx: Plot with log scaling on the x axis if True.
            colors: An sequence containing curves' plot colors.
            labels: An sequence containing curves' plot labels.
            grid: Show the grid lines if True.
            fig_ax: A (fig, ax) pyplot tuple.
            language: Language for axis annotations. Defaults to "french".
            output_file: Path to save the figure to.

        Returns:
            A 2-tuple containing

            - A figure containing the Axes plot.
            - An array (as created by `plt.subplots`) of Axes. One
            Axes is created for each feature.

        """
        if fig_ax is not None:
            fig, ax = fig_ax
        else:
            fig, ax = plt.subplots()

        for i, dataframe in enumerate(dataframes):
            dataframe_groupby = dataframe[feature].groupby(level="freq")
            df_upper_percentile = dataframe_groupby.quantile(max)
            df_middle_percentile = dataframe_groupby.quantile(mid)
            df_low_percentile = dataframe_groupby.quantile(min)

            if colors is None:
                color = None
            else:
                color = colors[i]

            df_middle_percentile.plot(ax=ax, color=color, logx=logx, label=labels[i])
            freqs = dataframe.index.get_level_values("freq").unique().values
            ax.fill_between(
                freqs,
                df_upper_percentile,
                df_low_percentile,
                facecolor=ax.get_lines()[-1].get_color(),
                alpha=0.2,
            )
        ax.xaxis.set_major_formatter(formatter_freq)
        ax.set_ylabel(STRINGS[language]["amplitude_db"])
        ax.set_xlabel(STRINGS[language]["frequency"])
        ax.set_title(STRINGS[language]["dispersion"].format(feature))
        ax.grid(grid)
        if labels is not None:
            ax.legend()
        if output_file is not None:
            fig.savefig(output_file, bbox_inches="tight")
        return fig, ax

    # function from "plot" in the "datasets-analysis" module (modified to display normalized spectra vertically)
    def modified_plot_feature_dispersion(
        dataframes: Sequence[pd.DataFrame],
        global_dataframes : Sequence[pd.DataFrame],
        feature: str,
        min: float = 0.25,
        max: float = 0.75,
        mid: float = 0.5,
        logx: bool = False,
        colors: Sequence[str] = None,
        labels: Sequence[str] = None,
        grid: bool = True,
        fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
        language: str = "french",
        output_file: Optional[str] = None,
        idx: int = 0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a feature's dispersion across time.

        The function plots the variability of a feature over a certain amount of time by
        computing percentile values. The dispersion is computed frequency-wise, that is,
        the percentile is computed over all feature values for each frequency point.
        Time and frequency steps are inherited from the DataFrame ones.

        Args:
            dataframes: A list of 2d dataframes indexed by "time" and "freq".
            feature: The name of the feature whose dispersion will be plotted.
            min: Float in [0-1), it is the p-th percentile, used to define the lower bound
                of the dispersion. Defaults to .25.
            max: Float in (0-1], it is the p-th percentile, used to define the upper bound
                of the dispersion. Defaults to .75.
            mid: Float in (0-1), it is the p-th percentile, used to define an "average"
                spectrum value. Defaults to .5.
            logx: Plot with log scaling on the x axis if True.
            colors: An sequence containing curves' plot colors.
            labels: An sequence containing curves' plot labels.
            grid: Show the grid lines if True.
            fig_ax: A (fig, ax) pyplot tuple.
            language: Language for axis annotations. Defaults to "french".
            output_file: Path to save the figure to.

        Returns:
            A 2-tuple containing

            - A figure containing the Axes plot.
            - An array (as created by `plt.subplots`) of Axes. One
            Axes is created for each feature.

        """
        if fig_ax is not None:
            fig, ax = fig_ax
        else:
            fig, ax = plt.subplots()

        min_value = global_dataframes.min()
        max_value = global_dataframes.max()


        

        dataframes = [dataframes.transform(normalize_between_0_and_1,idx=idx,min_value=min_value,max_value=max_value)]

        for i, dataframe in enumerate(dataframes):
            dataframe_groupby = dataframe[feature].groupby(level="freq")
            df_upper_percentile = dataframe_groupby.quantile(max)
            df_middle_percentile = dataframe_groupby.quantile(mid)
            df_low_percentile = dataframe_groupby.quantile(min)

            if colors is None:
                color = None
            else:
                color = colors[i]

            # Get the unique amplitude values instead of freqs for x-axis
            freqs = dataframe.index.get_level_values("freq").unique().values

            # Plot using amplitudes for x-axis and freqs for y-axis
            ax.plot(df_middle_percentile,freqs, color=color)
            ax.fill_betweenx(
                freqs,
                df_upper_percentile,
                df_low_percentile,
                facecolor=ax.get_lines()[-1].get_color(),
                alpha=0.2,
            )

        # Set the formatter and labels accordingly
        ax.xaxis.set_major_formatter(formatter_freq)
        ax.set_xlabel(STRINGS[language]["amplitude_db"])  # Assuming this is now amplitude label
        ax.set_ylabel(STRINGS[language]["frequency"])  # Assuming this is now frequency label
        ax.set_title(STRINGS[language]["dispersion"].format(feature))
        ax.grid(grid)
        if labels is not None:
            ax.legend(labels=labels)
        if output_file is not None:
            fig.savefig(output_file, bbox_inches="tight")

        return fig, ax

    def normalize_between_0_and_1(x,idx,min_value,max_value):
            # Normalisation Min-Max
            normalized = (x - min_value) / (max_value - min_value)
            normalized -= 0.25 - idx
            return normalized
        
    # function from "plot" in the "datasets-analysis" module (modified to correctly handle the display 
    # of spectrograms with a non-linear time axis - useful, for example, when time jumps occur due to 
    # repeated switching on and off of the measuring sensor)
    # -> Use of meshgrid and pcolormesh, instead of imshow
    def plot_2d_feature(
        dataframe: pd.DataFrame,
        feature: str,
        fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
        x_unit: str = "absolute",
        y_unit: str = "frequency",
        color_map: str = "inferno",
        add_color_bar: bool = True,
        color_limits: Optional[Tuple[Scalar, Scalar]] = None,
        language: str = "french",
        log_scale: bool = False,
        output_file: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a 2D feature.

        Args:
            dataframe: A DataFrame containing 2D features, indexed by wav filenames,
                frequency and DateTime indexes.
            feature: The name of the feature to be plotted.
            fig_ax: A 2-tuple containing a matplotlib Figures and Axes.
            x_unit: xticks labels unit. Defaults to "absolute" where xticks labels are set
                corresponding to dataframe's time index. Other choice is "relative" where
                the signal's duration in seconds is set as xticks labels.
            y_unit: yticks labels unit. Defaults to "frequency" where yticks labels are
                set corresponding to dataframe's frequency index. Other choice is "bands"
                where band frequency is recognized as octave bands or third octave bands.
                Number of bands is then set as yticks labels.
            color_map: Figure's colormap. Defaults to a perceptually unifrom colormap
                "inferno".
            add_color_bar: Add a colorbar to the figure. Defaults to True.
            color_limits: Set the color limits of the plotted figure.
            language: Language for axis annotations. Defaults to "french".
            output_file: Path to save the figure to.

        Returns:
            A 2-tuple containing

            - A figure containing the Axes plot.
            - An array (as created by `plt.subplots`) of Axes. One
            Axes is created for each feature.

        """
        if fig_ax is not None:
            fig, ax = fig_ax
        else:
            fig, ax = plt.subplots()

        data = dataframe[feature]
        time_ix = data.index.get_level_values("time").unique()
        time = (time_ix - time_ix[0]).total_seconds()
        freqs = data.index.get_level_values("freq").unique().values

        vals = data.values.reshape((time.size, freqs.size))
        norm = colors.Normalize(vals.min(), vals.max())

        if x_unit == "absolute":
            time = time_ix# mdates.date2num(time_ix)
            ax.xaxis_date()

        y_ticks_unit = "frequency"
        if y_unit == "bands":
            for band_type, freq_array in _NOMINAL_FREQ.items():
                if set(freqs) == set(freq_array):
                    y_ticks_unit = band_type
                    n_bands = len(freqs)
                    freqs = np.arange(n_bands + 1)

        # Create a meshgrid for time and frequency
        X, Y = np.meshgrid(time, freqs)

        # Use pcolormesh instead of imshow
        im = ax.pcolormesh(X, Y, vals.T, norm=norm, cmap=color_map, shading='auto')

        if color_limits is not None:
            im.set_clim(color_limits)

        if add_color_bar:
            cb = fig.colorbar(im, ax=ax)
            cb.set_label(STRINGS[language]["amplitude_db"])

        if log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            ax.yaxis.set_major_formatter(LogFormatter())
        else:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if y_unit == "frequency":
                ax.yaxis.set_major_formatter(formatter_freq)

        # Set x-axis label using `x_unit` and `STRINGS` dictionary
        ax.set_xlabel(STRINGS[language][x_unit])
        ax.set_ylabel(STRINGS[language][y_ticks_unit])
        ax.set_title(feature)

        if output_file is not None:
            fig.savefig(output_file, bbox_inches="tight")

        return fig, ax

    ##########################################
    ### Functions dedicated to the studies ###
    ##########################################

    def sort_key(path: str) -> Tuple[str, int]:
        """
        Generate a sort key based on the file path.

        This function extracts a sequence number from the file path if it exists,
        and returns a tuple containing the path without the sequence number and
        the sequence number itself. If no sequence number is found, the function
        returns the path with a default sequence number of 0.

        Args:
            path (str): The file path to extract the sort key from.

        Returns:
            Tuple[str, int]: A tuple where the first element is the file path without
            the sequence number and the second element is the sequence number (or 0 if not present).
        """
        # Extract the sequence number if it exists
        match = re.search(r'_(\d+)\.wav$', path)
        
        if match:
            # Return a tuple with the prefix and the sequence number as an integer
            return (path.rsplit('_', 1)[0], int(match.group(1)))
        else:
            # If no sequence number is found, return the full path with a default sequence number of 0
            return (path, 0)

    def get_recordings_from_time_range(metadata_path: str, 
        start: datetime, 
        end: datetime,
        timezone: str = 'Europe/Paris',
        recording_type: str = 'continuous',
        channel: int = 1,
        sensor: str = 'magneto',
        ) -> list:
        """
        Getting filenames and start_time of recordings in a time range.

        Args:
            metadata_path (str): Path of the sqlite db.
            start_time (datetime): Start time from where to get the recordings.
            end_time (datetime): End time from where to get the recordings.

        Returns:
            List[Recording]: Recordings list with infos from metadata.db.
        """

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Non existing path : {metadata_path}")

        db_url = "sqlite:///{}".format(metadata_path)
        db.initialize(url=db_url)


        with db.create_session() as session:
            logger.debug(f"getting recordings from {start} to {end}")
            recordings_query = session.query(
                Recording
            ).filter(
                Recording.start_time > start - datetime.timedelta(minutes=2)
            ).filter(
                Recording.start_time < end + datetime.timedelta(minutes=2)
            ).filter(
                Recording.recording_type == recording_type,
            )
            
            # t0 = [recording.start_time.replace(tzinfo=pytz.timezone(timezone) ) for recording in recordings_query]
            t0 = [recording.start_time.astimezone(pytz.timezone(timezone)) for recording in recordings_query]
            start_times = pd.DataFrame(index=[recording.filename for recording in recordings_query], data={"start_times": t0})

        recordings = []
        valid_filenames = []

        for recording in recordings_query:
            # Vérifier si le fichier wav existe
            wav_file_path = os.path.join(os.path.dirname(metadata_path), recording.filename)
            if os.path.exists(wav_file_path):
                if sensor == 'magneto':
                    recordings.append(recording)
                    valid_filenames.append(recording.filename)
                else:
                    if ('_Tr'+str(channel) in recording.filename or '_channel'+str(channel) in recording.filename):
                        recordings.append(recording)
                        valid_filenames.append(recording.filename)                
            else:
                logger.debug(f"File {wav_file_path} not found, removing from the list.")

        if not sensor == 'magneto':    
            channel = 1

        # Filtrer start_times pour ne conserver que les enregistrements valides
        start_times = start_times.loc[valid_filenames]

        # Groupement des fichiers par préfixe
        file_groups = {}
        for file_name in valid_filenames:
            prefix = file_name.split('_')[0]
            
            if prefix not in file_groups:
                file_groups[prefix] = []
            
            # Stocker le fichier avec son start_time
            file_groups[prefix].append((file_name))
        
        # Trier les fichiers dans chaque groupe par start_time
        for key in file_groups:
            file_groups[key].sort(key=lambda x: x[1])

        return recordings,db_url,start_times,file_groups,channel

    def should_filter(metadata_df, new_metadata):
        # Si dataframe vide, filtrage nécessaire
        if metadata_df.empty:
            return True

        old_metadata = metadata_df.iloc[0]

        # Vérifier les clés simples
        keys_to_check = ['cutoff_freqs', 'btype', 'order']
        for key in keys_to_check:
            if old_metadata[key] != new_metadata[key]:
                return True

        # Vérifier la liste des fichiers (ordre ou contenu différent)
        old_files = old_metadata['files']
        new_files = new_metadata['files']

        if old_files != new_files:
            return True

        return False

    # Fonction pour décider si un fichier doit être filtré
    def needs_filtering(row, cutoff_freqs, btype, order):
        if row is None:
            return True  # pas dans metadata
        # Comparaison simple, cutoff_freqs en string pour homogénéité
        return (row['cutoff_freqs'] != str(cutoff_freqs) or
                row['btype'] != btype or
                row['order'] != order)

    def apply_filter_and_save(file_paths: list[str],
                            cutoff_freqs: list[float],
                            btype: str = 'low',
                            order: int = 5) -> pd.DataFrame:
        """
        Applies a filter to WAV files and saves the filtered versions.
        
        Args:
            file_paths (list[str]): List of WAV file paths.
            cutoff_freqs (list[float]): Cutoff frequencies for the filter.
            btype (str, optional): Filter type ('low', 'high', 'band'). Defaults to 'low'.
            order (int, optional): Filter order. Defaults to 5.
        
        Returns:
            pd.DataFrame: DataFrame containing the filter metadata.
        """

        # def should_filter(metadata_df, new_metadata):
        #     # Si dataframe vide, filtrage nécessaire
        #     if metadata_df.empty:
        #         return True

        #     old_metadata = metadata_df.iloc[0]

        #     # Vérifier les clés simples
        #     keys_to_check = ['cutoff_freqs', 'btype', 'order']
        #     for key in keys_to_check:
        #         if old_metadata[key] != new_metadata[key]:
        #             return True

        #     # Vérifier la liste des fichiers (ordre ou contenu différent)
        #     old_files = old_metadata['files']
        #     new_files = new_metadata['files']

        #     # old_files dans CSV est une string, on convertit en liste
        #     import ast
        #     try:
        #         old_files_list = ast.literal_eval(old_files)
        #     except Exception:
        #         old_files_list = []

        #     if old_files_list != new_files:
        #         return True

        #     return False


        # # ---

        output_dir = os.path.join(os.path.dirname(file_paths[0]), 'filtered_data', f'ch{channel}')
        metadata_path = os.path.join(output_dir, "metadata_filter.csv")
        os.makedirs(output_dir, exist_ok=True)

        # Chargement ou création du metadata_df
        if os.path.exists(metadata_path):
            metadata_df = pd.read_csv(metadata_path, index_col=0)
        else:
            metadata_df = pd.DataFrame(columns=['cutoff_freqs', 'btype', 'order'])
            metadata_df.index.name = 'filename'



        # Filtrage des fichiers qui en ont besoin
        files_to_filter = []
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            if filename in metadata_df.index:
                if needs_filtering(metadata_df.loc[filename], cutoff_freqs, btype, order):
                    files_to_filter.append(file_path)
            else:
                files_to_filter.append(file_path)

        if files_to_filter:
            with tqdm(total=len(files_to_filter), desc="Filtering files", dynamic_ncols=True) as pbar:
                for file_path in files_to_filter:
                    filename = os.path.basename(file_path)
                    output_path = os.path.join(output_dir, filename)

                    if not os.path.exists(file_path):
                        print(f"File {file_path} does not exist. Skipping...")
                        pbar.update(1)
                        continue

                    data, fs = sf.read(file_path, dtype='float32')

                    # Appel de ta fonction de filtrage (à adapter)
                    filtered_data = butter_filter(data, cutoff_freqs, fs, btype=btype, order=order)
                    filtered_data = filtered_data.astype(np.float32)

                    sf.write(output_path, filtered_data, fs, subtype='FLOAT')

                    # Mise à jour metadata_df avec les paramètres actuels
                    metadata_df.loc[filename] = [str(cutoff_freqs), btype, order]

                    pbar.set_description(f"Filtering file: {filename}")
                    pbar.update(1)

            # Sauvegarde du metadata mis à jour
            metadata_df.to_csv(metadata_path)
        else:
            print("Tous les fichiers ont déjà été filtrés avec ces paramètres. Rien à faire.")
        # # Comparison
        # if (metadata_df.empty or
        #     metadata_df.iloc[0]['cutoff_freqs'] != new_metadata['cutoff_freqs'] or
        #     metadata_df.iloc[0]['btype'] != new_metadata['btype'] or
        #     metadata_df.iloc[0]['order'] != new_metadata['order']):

        #     with tqdm(total=len(file_paths), desc="Filtering files", dynamic_ncols=True) as pbar:
        #         for file_path in file_paths:
        #             base_name = os.path.basename(file_path)
        #             output_path = os.path.join(output_dir, base_name)

        #             if not os.path.exists(file_path):
        #                 print(f"File {file_path} does not exist. Skipping...")
        #                 pbar.update(1)  
        #                 continue

        #             data, fs = sf.read(file_path, dtype='float32')  
                    
        #             filtered_data = butter_filter(data, cutoff_freqs, fs, btype=btype, order=order)
        #             filtered_data = filtered_data.astype(np.float32)

        #             sf.write(output_path, filtered_data, fs, subtype='FLOAT')

        #             pbar.set_description(f"Filtering file: {base_name}")
        #             pbar.update(1)

        #     # Save to a CSV file
        #     metadata_df = pd.DataFrame([new_metadata])
        #     metadata_df.to_csv(metadata_path, index=False)

        return metadata_df

    def process_data(recordings: list[str], 
                    channel: int, 
                    start_times: pd.DataFrame, 
                    file_groups: dict, 
                    freq_limits: list[float], 
                    block_duration: int = 1, 
                    band_freq: int = 50, 
                    features_list: list = None,
                    compute_leak_detection: bool=False,
                    bandleq_features: dict = None):  # Ajout d'un paramètre pour les nouvelles fonctionnalités bandleq
        """
        Processes data to compute the spectrogram and features.

        Args:
            recordings (list[str]): List of WAV file paths to process.
            channel (int): Channel to process (1-based index).
            start_times (pd.DataFrame): DataFrame containing start times.
            file_groups (dict): Dictionary containing grouped files by prefix.
            freq_limits (list[float]): Frequency limits for the spectrogram.
            block_duration (int, optional): Duration of blocks for spectrogram computation. Default is 1.
            band_freq (int, optional): Frequency of the band for spectrogram computation. Default is 50.
            features_list (list, optional): List of features to compute. Default is None.
            compute_leak_detection (bool,optional) : computation of leak detection.
            bandleq_features (dict, optional): Dictionnaire avec des clés comme les noms de fonctionnalités et des tuples
                                                représentant les limites de fréquence (min, max).

        Returns:
            pd.DataFrame: Combined spectrogram data.
            pd.DataFrame: Combined feature data.
            pd.DataFrame: Combined Q-factors data.
            pd.DataFrame: Combined leak_detection data.
            bool: Indicator of whether the 'rms' feature exists.
            list[float]: Updated frequency limits.
        """
        # Initialisation
        combined_spectrograms = pd.DataFrame()
        combined_features = pd.DataFrame()
        combined_q_factors = pd.DataFrame()
        combined_leak_detection = pd.DataFrame()
        # combined_onsets = pd.DataFrame()
        rms_exists = True
        if features_list is None:
            features_list = []

        if compute_leak_detection:
            mel_feat = ["melspectrogram{}".format(i) for i in range(128)]
            labels = ["leak", "noise"]
            
            model = offline_evaluation_helpers.get_model(
                mel_feat, labels, in_prod=True)

        # Assurez-vous que la fonctionnalité 'rms' est incluse pour le calcul SNR
        if 'rms' not in features_list:
            rms_exists = False
            features_list.append('rms')

        if not recordings:
            logging.error("The 'recordings' list is empty. Stopping processing.")
            raise ValueError("Error: 'recordings' list is empty.")
        else:
            dirname = os.path.dirname(recordings[0])

        # Process each group of files (various files in a group when "continuous recording")
        total_files = sum(len(files) for files in file_groups.values())  # Total number of files
        with tqdm(total=total_files, desc="Calculating features & spectrograms", dynamic_ncols=True) as pbar:
            for prefix, files in file_groups.items():
                for i, filename in enumerate(files):
                    file_path = os.path.join(dirname,filename)
                    if not os.path.exists(file_path):
                        if apply_filter:
                            raise FileNotFoundError(f"File {file_path}  not found. The filtered folder may have top be cleared.")
                        else:
                            raise FileNotFoundError(f"File {file_path}  not found.")
                        
                    # try:
                    wav, fs = sf.read(file_path)

                    # Handle multichannel data (extract selected channel)
                    if wav.ndim > 1:
                        wav = wav.T[channel - 1, :]
                    wav = wav.reshape(-1)

                    # If the group contains only a single file (e.g., periodic recording), use its start time.
                    datetime_str = os.path.basename(filename)
                    if i == 0:
                        t0 = start_times.loc[datetime_str, 'start_times'] if datetime_str in start_times.index else 0
                    else:
                        last_index = combined_features.index[-1]
                        t0 = last_index + pd.Timedelta(seconds=block_duration)

                    # Compute [spectrogram, features, Q-factors]
                    spectrogram, features, q_factors, freq_idx = compute_features_bandleq(
                        wav, fs, block_duration, float(band_freq), t0, features_list, Q_factor, freq_limits
                    )

                    # Calculer les fonctionnalités de bandleq si spécifié
                    if bandleq_features:
                        for feature_name, (freq_min, freq_max) in bandleq_features.items():
                            
                            freq_mask = (spectrogram.index.get_level_values('freq') >= freq_min) & \
                                        (spectrogram.index.get_level_values('freq') <= freq_max)

                            if freq_mask.any():
                                selected_data = spectrogram[freq_mask]
                                features[feature_name] = selected_data.groupby(level='time').mean()
                            else:
                                logging.warning(f"Aucune fréquence trouvée pour {feature_name} dans les limites ({freq_min}, {freq_max})")
                    if compute_leak_detection:
                        blocks = mel_in_blocks(file_path, t0, channel)
                        preds = write_output_for_blocks(blocks, model)
                        combined_leak_detection = pd.concat([combined_leak_detection, preds])     


                    combined_spectrograms = pd.concat([combined_spectrograms, spectrogram])
                    combined_features = pd.concat([combined_features, features])



                    combined_q_factors = pd.concat([combined_q_factors, q_factors])

                        

                    # except Exception as e:
                    #     logging.error(f"Error processing file {file_path}: {e}")
                    #     print(f"Error processing file {file_path}: {e}")

                    pbar.update(1)

        return combined_spectrograms, combined_features, combined_q_factors, combined_leak_detection, rms_exists, freq_limits

    def Q_factors(
        spectrogram: pd.DataFrame,
        freq_limits: Optional[np.ndarray],
        band_freq: float
    ) -> pd.DataFrame:
        """
        Calculate Q-factors from the spectrogram data.

        Args:
            spectrogram (pd.DataFrame): DataFrame containing the spectrogram with MultiIndex (time, freq).
            freq_limits (Optional[np.ndarray]): Frequency limits for filtering the data. If None, no filtering is applied.
            band_freq (float): Frequency bandwidth for the Q-factor calculation.

        Returns:
            pd.DataFrame: DataFrame containing the Q-factors and other related metrics.
        """
        all_q_factors = {
            'time': [],
            'peak freq.': [],
            'peak mag.': [],
            'Q-factor': [],
            'Q-sum': [],
            'spectrum correlation': [],
            'Q-prod': []
        }

        time_index = spectrogram.index.get_level_values('time').unique()
        previous_spectrum = None
        previous_peaks = None
        
        for time in time_index:
            spectrum = spectrogram.loc[time]
            y_log = spectrum['bandleq'].values
            freq = spectrum.index.astype(float)
            y_lin = 10 ** (y_log / 20)

            if freq_limits is not None:
                mask = (freq > freq_limits[0]) & (freq < freq_limits[1])
                freq = freq[mask]
                y_lin = y_lin[mask]
                y_log = y_log[mask]

            peaks, _ = find_peaks(y_lin, prominence=0.0001)
            q_factors = [calculate_q_factor(freq, y_lin, idx, band_freq) for idx in peaks]

            df_Q_factor = pd.DataFrame({
                'peak freq.': freq[peaks],
                'peak mag.': y_log[peaks],
                'Q-factor': q_factors
            })
            df_Q_factor = df_Q_factor.sort_values(by='Q-factor', ascending=False)
            df_Q_factor = df_Q_factor.head(10)
            df_Q_factor = df_Q_factor[df_Q_factor['Q-factor'] >= 0.05]
            
            # Calculate correlation with the previous spectrum
            if previous_spectrum is not None:
                correlation, _ = pearsonr(y_lin, previous_spectrum)
            else:
                correlation = None
            
            # Compare peaks with the previous iteration
            if previous_peaks is not None:
                common_peaks = np.intersect1d(previous_peaks, df_Q_factor['peak freq.']).size
            else:
                common_peaks = 0
            
            # Calculate Q-prod : Q prod is the product of the sum of the 10 largest Q factor values in the spectrum (expressing the fineness of the spectral lines) 
            # and the number of lines in common with the previous spectrum (expressing line continuity).
            q_sum = np.sum(q_factors)
            q_prod = common_peaks * q_sum

            all_q_factors['time'].append(time)
            all_q_factors['peak freq.'].append(df_Q_factor['peak freq.'].tolist())
            all_q_factors['peak mag.'].append(df_Q_factor['peak mag.'].tolist())
            all_q_factors['Q-factor'].append(df_Q_factor['Q-factor'].tolist())
            all_q_factors['Q-sum'].append(q_sum)
            all_q_factors['spectrum correlation'].append(correlation)
            all_q_factors['Q-prod'].append(q_prod)

            previous_spectrum = y_lin
            previous_peaks = df_Q_factor['peak freq.'].values

        Q_factors_df = pd.DataFrame(all_q_factors)
        Q_factors_df.set_index('time', inplace=True)

        return Q_factors_df
        
    def compute_features_bandleq(
        wav: np.ndarray, 
        fs: int, 
        block_duration: float, 
        band_freq: float, 
        t0: datetime.datetime, 
        features_list: List[str], 
        q_factor: bool, 
        freq_limits: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract features and bandleq from .wav data.

        Args:
            wav (np.ndarray): NumPy array containing .wav data.
            fs (int): Sampling frequency in Hz.
            block_duration (float): Duration of blocks for bandleq calculation in seconds.
            band_freq (float): Frequency bandwidth in Hz for the calculation.
            t0 (datetime.datetime): Start time of the recording.
            features_list (List[str]): List of features to compute.
            q_factor (bool): Flag indicating whether to compute Q-factors.
            freq_limits (np.ndarray): Frequency limits for Q-factor calculation.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing:
                - spectrogram DataFrame with bandleq values.
                - features DataFrame with computed features.
                - q_factors DataFrame with computed Q-factors (if applicable).
        """

        block_size = int(fs * block_duration)
        freq = np.arange(band_freq, int(fs / 2), band_freq)
        fc = FeaturesComputer(block_size=block_size, rate=fs, features=['bandleq'], band_freq=band_freq)
        blocks = split_signal(wav, fs, block_size / fs)
        features = fc.compute(blocks)
        n_blocks = features['bandleq'].shape[0]
        time_array = np.arange(n_blocks) * block_size / fs
        time = [t0 + datetime.timedelta(seconds=value) for value in time_array]

        # Create MultiIndex for the spectrogram DataFrame
        time_idx = pd.DatetimeIndex(time, name='time')
        freq_idx = pd.Index(freq, name='freq')
        midx = pd.MultiIndex.from_product([time_idx, freq_idx], names=["time", "freq"])

        # Create the spectrogram DataFrame
        spectrogram = pd.DataFrame(data=features["bandleq"].flatten(), columns=["bandleq"], index=midx)

        # Compute Q-factors if required
        q_factors = pd.DataFrame()
        if q_factor:
            q_factors = Q_factors(spectrogram, freq_limits, band_freq)

        # if onset_detection:
        #     onset_results = [
        #         1 if len(librosa.onset.onset_detect(y=block, sr=fs)) > 1 else 0
        #         for block in blocks
        #     ]
        #     onsets = pd.DataFrame({'onsets': onset_results})
                
        # # Prendre les 10 derniers blocs
        # last_block_start_index = max(0, n_blocks - 10)
        # last_blocks = blocks[-10:]

        # # Préparer les subplots
        # fig, axs = plt.subplots(10, 1, figsize=(12, 20), sharex=False)
        # fig.suptitle("Détection d'onsets sur les 10 derniers blocs", fontsize=16)

        # for i, (block, ax) in enumerate(zip(last_blocks, axs)):
        #     block_index = last_block_start_index + i
        #     block_start_time = block_index * block_duration

        #     # Onsets
        #     onset_env = librosa.onset.onset_strength(y=block, sr=fs, hop_length=512)
        #     onset_frames = librosa.onset.onset_detect(
        #         onset_envelope=onset_env,
        #         sr=fs,
        #         hop_length=512,
        #         backtrack=False,       # pas besoin de reculer : garde la crête
        #         pre_max=10,            # chercher des pics plus saillants
        #         post_max=10,
        #         pre_avg=20,            # lisser davantage avant/après
        #         post_avg=20,
        #         delta=0.2,             # seuil plus élevé → moins de détection
        #         wait=5                 # éviter plusieurs détections rapprochées
        #     )
        #     onset_times = librosa.frames_to_time(onset_frames, sr=fs)

        #     # Tracer la forme d’onde du bloc
        #     t = np.linspace(0, len(block)/fs, num=len(block))
        #     ax.plot(t, block, label=f'Bloc {block_index}')
        #     ax.set_ylabel("Amplitude")
        #     ax.set_title(f"Bloc {block_index} (t = {block_start_time:.1f}s à {block_start_time + block_duration:.1f}s)")

        #     # Tracer les onsets (relatifs au bloc)
        #     for ot in onset_times:
        #         ax.axvline(x=ot, color='r', linestyle='--')

        #     ax.legend(loc='upper right')

        # plt.xlabel("Temps (s)")
        # plt.tight_layout(rect=[0, 0, 1, 0.97])
        # plt.show()
            

        

        # Initialize FeaturesComputer for additional features
        fc = FeaturesComputer(block_size=block_size, rate=fs, features=features_list, band_freq=band_freq)
        features = pd.DataFrame(fc.compute(blocks))
        features.index = time_idx
        features.loc[features['rms'] == 0] = np.nan

        return spectrogram, features, q_factors, freq_idx

    def mel_in_blocks(file_path: str, start_time: float,channel : int):
        """compute dataframe containing the timestamps and melspectrograms
        """
        n_frames_per_example = 50
        rate = 192000
        block_size = round(0.02*rate*block_duration)
        n_mels = 128
        

        features_array, dt = ml_features.extract_features_from_file(
            path=file_path,
            target_sample_rate=rate,
            feature_extraction_fn=ml_features.extract_melspectrogram,
            n_fft=block_size,
            hop_length=block_size,
            n_mels=n_mels,
        )

        n_frames = features_array.shape[0]
        time_step = block_size / rate
        time_ix = pd.date_range(
            start=start_time, periods=n_frames, freq="%.3fS" % time_step
        )
        features_array = features_array[:,:,channel-1]

        df_mel = pd.DataFrame(
            features_array.squeeze(),
            columns=["melspectrogram{}".format(i) for i in range(n_mels)],
            index=time_ix
        )
        file_name = os.path.basename(file_path)

        index = pd.MultiIndex.from_tuples(
            [
            (file_name, time_ix)
            for time_ix in df_mel.index
        ],
            names=["filename", "time"]
        )
        df_mel.index = index

        blocks = []
        for i in range(len(df_mel)//n_frames_per_example):
            blocks.append(
                df_mel.iloc[i*n_frames_per_example: (i+1)*n_frames_per_example])
        
        return blocks

    def write_output_for_blocks(blocks, model):
        preds = []
        for block in blocks:
            pred = model.predict_proba(block)
            print(block.index.get_level_values("time")[-1])
            pred["time"] = block.index.get_level_values("time")[-1]
            preds.append(pred)

        preds = pd.concat(preds)
        preds.set_index("time", inplace=True)
        print(preds)
        return preds

    def plot_periods(df: pd.DataFrame, 
                    labeled_features: pd.DataFrame, 
                    ax: plt.Axes, 
                    last_plot: bool, 
                    label_color_map: dict):
        """
        Plots periods as shaded areas on a given axis.

        Args:
            df (pd.DataFrame): DataFrame containing 'Start', 'End', and 'Label' columns.
            labeled_features (pd.DataFrame): DataFrame containing the features with time index.
            ax (plt.Axes): Matplotlib axis to plot on.
            last_plot (bool): Flag to determine if the legend should be added.
            label_color_map (dict): Dictionary mapping labels to colors.

        Returns:
            None
        """
        # Convert 'Start' and 'End' to datetime if they are not already
        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])

        # Determine the time range for plotting
        start_time = labeled_features.index.get_level_values('time').min()
        end_time = labeled_features.index.get_level_values('time').max()

        # Filter the DataFrame to include only rows within the time range
        filtered_df = df[(df['Start'] <= end_time) & (df['End'] >= start_time)]

        # Plot the periods
        for idx, row in filtered_df.iterrows():
            start = row["Start"]
            end = row["End"]
            label = row["Label"]

            ax.axvspan(date2num(start), date2num(end), 
                    label=label, color=label_color_map.get(label, 'grey'), alpha=0.3)

        # Add legend if this is the last plot
        if last_plot:
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = {label: handle for handle, label in zip(handles, labels)}
            ax.legend(unique_labels.values(), unique_labels.keys(), 
                    ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.5))

    def find_uncovered_intervals(covered_intervals: List[Tuple[float, float]], 
                                full_range: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Find intervals within a full range that are not covered by any of the given intervals.
        Use to identify intervals labelled “other”.

        Args:
            covered_intervals (List[Tuple[float, float]]): List of covered intervals, where each interval is a tuple (start, end).
            full_range (Tuple[float, float]): The full range (start, end) to check for uncovered intervals.

        Returns:
            List[Tuple[float, float]]: List of uncovered intervals within the full range, where each interval is a tuple (start, end).
        """
        uncovered_intervals = []
        current_start = full_range[0]
        
        for start, end in sorted(covered_intervals):
            if current_start < start:
                uncovered_intervals.append((current_start, start))
            current_start = max(current_start, end)
        
        if current_start < full_range[1]:
            uncovered_intervals.append((current_start, full_range[1]))
        
        return uncovered_intervals

    def label_and_concat(df: pd.DataFrame, 
                        label: str, 
                        start: pd.Timestamp, 
                        end: pd.Timestamp) -> pd.DataFrame:
        """
        Label a subset of a DataFrame and concatenate the label to the index.

        Args:
            df (pd.DataFrame): The DataFrame with a multi-index where the first level is time.
            label (str): The label to assign to the rows within the specified time range.
            start (pd.Timestamp): The start of the time range to label.
            end (pd.Timestamp): The end of the time range to label.

        Returns:
            pd.DataFrame: The DataFrame with an additional index level for the label, 
                        containing only the rows within the specified time range.
        """
        period_data = df.loc[(df.index.get_level_values(0) >= start) & (df.index.get_level_values(0) <= end)]
        period_data['Label'] = label
        period_data = period_data.set_index('Label', append=True)
        
        return period_data

    def get_frames_in_periods(df_labels: pd.DataFrame, 
                            spectrograms: pd.DataFrame, 
                            features: pd.DataFrame, 
                            q_factors: pd.DataFrame, 
                            leak_detection: pd.DataFrame, 
                            rms_exists: bool, 
                            timezone: str = 'Europe/Paris', 
                            emergence: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Retrieve frames DataFrame associated with each label period.

        Args:
            df_labels (pd.DataFrame): DataFrame containing columns 'Start', 'End', and 'Label'.
            spectrograms (pd.DataFrame): DataFrame containing spectrogram data.
            features (pd.DataFrame): DataFrame containing the features.
            q_factors (pd.DataFrame): DataFrame containing quality factors.
            leak_detection (pd.DataFrame): DataFrame containing leak detection.
            rms_exists (bool): Boolean indicating if RMS is in `features`.
            timezone (str, optional): Timezone for processing. Default is 'Europe/Paris'.
            emergence (bool, optional): Boolean to apply background adjustment. Default is True.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                - labeled_spectrograms (pd.DataFrame): Spectrogram data labeled by periods.
                - labeled_features (pd.DataFrame): Features data labeled by periods.
                - labeled_q_factors (pd.DataFrame): Quality factors labeled by periods.
                - features (pd.DataFrame): Updated features DataFrame, adjusted for SNR if applicable.
        """

        # Group by labels to process each period separately
        labeled_groups = df_labels.groupby('Label')
        labeled_features = pd.DataFrame()
        labeled_spectrograms = pd.DataFrame()
        labeled_q_factors = pd.DataFrame()
        labeled_leak_detection = pd.DataFrame()

        timezone = pytz.timezone(timezone)
        
        covered_intervals = []
        snr = False

        # Process each labeled group
        for label, group in tqdm(labeled_groups, desc="Labelled data processing"):
            for _, row in group.iterrows():
                start = pd.Timestamp(row['Start'])
                end = pd.Timestamp(row['End'])
                
                covered_intervals.append((start, end))
                
                # Label and concatenate data for the current period
                labeled_features = pd.concat([labeled_features, label_and_concat(features, label, start, end)])
                labeled_spectrograms = pd.concat([labeled_spectrograms, label_and_concat(spectrograms, label, start, end)])
                labeled_q_factors = pd.concat([labeled_q_factors, label_and_concat(q_factors, label, start, end)])
                labeled_leak_detection = pd.concat([labeled_leak_detection, label_and_concat(leak_detection, label, start, end)])
            if 'background' in labeled_spectrograms.index.get_level_values('Label').unique():
                snr = True

        # Determine the full range of the data
        full_range = (pd.Timestamp(spectrograms.index.get_level_values(0).min()), pd.Timestamp(spectrograms.index.get_level_values(0).max()))

        # Find uncovered intervals
        uncovered_intervals = find_uncovered_intervals(covered_intervals, full_range)

        # Process uncovered intervals with 'other' label
        for start, end in tqdm(uncovered_intervals, desc="Unlabelled data processing"):
            labeled_features = pd.concat([labeled_features, label_and_concat(features, 'other', start, end)])
            labeled_spectrograms = pd.concat([labeled_spectrograms, label_and_concat(spectrograms, 'other', start, end)])
            labeled_q_factors = pd.concat([labeled_q_factors, label_and_concat(q_factors, 'other', start, end)])
            labeled_leak_detection = pd.concat([labeled_leak_detection, label_and_concat(leak_detection, 'other', start, end)])

        # If SNR calculation is needed
        if snr:
            rms_bkg = labeled_features.loc[labeled_features.index.get_level_values('Label') == 'background', 'rms'].dropna().values.mean()
            y = 20 * np.log10(features['rms'].dropna().values / rms_bkg)
            labeled_y = 20 * np.log10(labeled_features['rms'].dropna().values / rms_bkg)

            features['SNR'] = pd.DataFrame(data=y, index=features.index, columns=['SNR'])
            labeled_features['SNR'] = pd.DataFrame(data=labeled_y, index=labeled_features.index, columns=['SNR'])

            # Apply background adjustment if specified
            if emergence:
                spectrum_bkg = labeled_spectrograms.loc[labeled_spectrograms.index.get_level_values('Label') == 'background'].groupby('freq').mean()
                spectrograms -= spectrum_bkg
                labeled_spectrograms -= spectrum_bkg

        # Remove 'rms' if it does not exist in the features list
        if not rms_exists:
            labeled_features = labeled_features.drop(columns=["rms"])
            features = features.drop(columns=["rms"])

        return labeled_spectrograms, labeled_features, labeled_q_factors, labeled_leak_detection, features

    def plot_spectra(labeled_spectrograms: pd.DataFrame, 
                    label_color_map: Dict[str, str], 
                    freq_limits: Optional[Tuple[float, float]] = None) -> None:
        """
        Plot the dispersion of features across different frequency bands.

        Args:
            labeled_spectrograms (pd.DataFrame): DataFrame containing labeled spectrogram data.
            label_color_map (dict): Dictionary mapping labels to colors.
            freq_limits (tuple, optional): Tuple specifying frequency limits for the x-axis (e.g., (min_freq, max_freq)).
        """
        # Exclude specified labels from the plot
        excluded_labels = ['background', 'other']
        filtered_spectrograms = labeled_spectrograms[~labeled_spectrograms.index.get_level_values('Label').isin(excluded_labels)]

        # Group the filtered spectrograms by label
        grouped = filtered_spectrograms.groupby(level='Label')
        freq = filtered_spectrograms.index.get_level_values(level='freq').unique()
        list_of_dfs = [group for _, group in grouped]
    
        # Prepare color and label sequences for plotting
        color_sequence = [color for label, color in label_color_map.items() if label not in excluded_labels]
        label_sequence = [label for label in label_color_map.keys() if label not in excluded_labels]

        # Create a new figure and axis for plotting
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot the feature dispersion using the custom plotting function
        plot_feature_dispersion(
            list_of_dfs,
            "bandleq",
            logx=True,
            fig_ax=(fig, ax),
            colors=color_sequence,
            labels=label_sequence,
        )
        
        # Apply frequency limits if provided
        if freq_limits:
            plt.xlim(freq_limits)
        
        # Show the plot
        plt.show()

    def plot_features_with_boxplot(features_list: List[str], 
                                    spectrograms: pd.DataFrame, 
                                    features: pd.DataFrame, 
                                    label_color_map: Dict[str, str], 
                                    freq_limits: Optional[Tuple[float, float]], 
                                    start : datetime.datetime,
                                    end : datetime.datetime,
                                    df_labels: Optional[pd.DataFrame] = None, 
                                    labeled_features: Optional[pd.DataFrame] = None, 
                                    campaign_name: str = "Campaign", 
                                    tz: str = "Europe/Paris", 
                                    window_rolling: int = 30) -> None:
        """
        Plot features and their statistics, including a spectrogram and boxplots.

        Args:
            features_list (list of str): List of feature names to plot.
            spectrograms (pd.DataFrame): DataFrame containing spectrogram data.
            features (pd.DataFrame): DataFrame containing feature data.
            label_color_map (dict): Dictionary mapping labels to colors.
            freq_limits (tuple, optional): Tuple specifying frequency limits for the x-axis (e.g., (min_freq, max_freq)).
            df_labels (pd.DataFrame, optional): DataFrame containing label periods with columns 'Start', 'End', and 'Label'.
            labeled_features (pd.DataFrame, optional): DataFrame containing features labeled by periods.
            campaign_name (str, optional): Name of the campaign for the plot title. Default is "Campaign".
            tz (str, optional): Timezone for formatting dates. Default is "Europe/Paris".
            window_rolling (int, optional): Window size for rolling median. Default is 30.
        """
        n_features = len(features_list)
        first_date = features.index[0]
        date_fmt = first_date.strftime('%d-%m-%Y')
        hours_fmt = mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone(tz))
        nrows, ncols = (n_features + 1, 1) if labeled_features is None or labeled_features.empty else (n_features + 1, 2)
        if df_labels.empty :
            ncols = 1


        
        log_scale = 'log' in frequency_scale

        fig, ax = plt.subplots(nrows, ncols, figsize=(12, 3 * (n_features + 1)), tight_layout=True, sharex='col', sharey='row')

        if ncols == 1:
            ax = np.expand_dims(ax, axis=-1)

        label_list = labeled_features.index.get_level_values(level=1).unique() if labeled_features is not None else []


        
        for c in range(ncols):
            if c == 0:
                # Plot spectrogram
                plot_2d_feature(
                    spectrograms,
                    "bandleq",
                    fig_ax=[fig, ax[0, 0]],
                    x_unit="absolute",
                    y_unit="band",
                    add_color_bar=False,
                    language='english',
                    log_scale=log_scale
                )
                ax[0, 0].set_xlabel('')
                ax[0, 0].set_title('Spectrogram')
                ax[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
                ax[0, 0].xaxis.set_major_formatter(hours_fmt)

                last_plot = False
                for i, feat in enumerate(features_list):
                    ax[i + 1, 0].plot(features[feat], '.', markersize=3)
                    ax[i + 1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
                    ax[i + 1, 0].set_ylabel(feat)
                    
                    if not df_labels.empty :
                        if feat == features_list[-1]:
                            last_plot = True
                        plot_periods(df_labels, labeled_features, ax[i + 1, 0], last_plot, label_color_map)
                
                ax[i + 1, 0].set_xlabel('Time (hh:mm:ss)')
                ax[i + 1, 0].set_xlim(start, end)
                ax[i + 1, 0].sharex(ax[0, 0])
                ax[1, 0].set_title('Features')
                
                if ncols > 1:
                    ax[i + 1, 0].set_xticklabels(ax[i + 1, 0].get_xticklabels(), rotation=45, ha='right')
            
            else:
                for i, label in enumerate(label_list):
                    modified_plot_feature_dispersion(
                        labeled_spectrograms.loc[:, :, label],
                        labeled_spectrograms,
                        "bandleq",
                        fig_ax=[fig, ax[0, 1]],
                        logx=False,
                        colors='k',
                        language='english',
                        idx=i
                    )
                    ax[0, 1].set_ylim(freq_limits)
                    ax[0, 1].set_xlabel('')
                    ax[0, 1].set_ylabel('')
                
                for i, feat in enumerate(features_list):
                    selected_data = labeled_features.loc[:, feat].reset_index().pivot(index='time', columns='Label', values=feat)
                    color_list = [label_color_map.get(next((label for label in label_color_map if label.strip() in col), 'other'), 'other') for col in selected_data.columns]

                    sns.boxplot(data=selected_data, ax=ax[i + 1, 1], boxprops={'alpha': 0.3}, palette=color_list)
                    ax[i + 1, 1].xaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
                    ax[i + 1, 1].yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
                    ax[i + 1, 1].set_ylabel('')
                    ax[i + 1, 1].set_xticklabels(label_list, rotation=45, ha='right')
                ax[1, 1].set_title('Statistics')
                ax[i + 1, 1].set_xlabel('')

        plt.suptitle(f'{campaign_name} - Date: {date_fmt}', fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Laisse 5% en haut
        
        if 'Q-prod' in features_list:
            all_values = [value for sublist in features['Q-factor'] for value in sublist]
            vmin = np.min(all_values)
            vmax = np.max(all_values)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            times_list = []
            peaks_list = []
            alpha_list = []

            for time in features.index.unique():
                q_values = features.loc[time]['Q-factor']
                peak_values = features.loc[time]['peak freq.']
                
                if isinstance(peak_values, (np.float64, float)):
                    peak_values = [peak_values]
                    q_values = [q_values]
                
                times_list.extend([time] * len(peak_values))
                peaks_list.extend(peak_values)
                alpha_list.extend([norm(q) for q in q_values]) 

            ax[0, 0].scatter(times_list, peaks_list, color='w', edgecolor=None, linewidth=0.5, alpha=alpha_list, s=3, zorder=5)

            rolling_median1 = q_factors['Q-prod'].rolling(window=window_rolling, min_periods=1).median()
            rolling_median2 = q_factors['spectrum correlation'].rolling(window=window_rolling, min_periods=1).median()
            ax[1, 0].plot(rolling_median1, 'r')
            ax[1, 0].set_ylim(0, 40)
            ax[2, 0].plot(rolling_median2, 'r')
            ax[2, 0].set_ylim(0, 1)
        if 'leak' in features_list:
            rolling_median = features['leak'].rolling(window=window_rolling, min_periods=1).median()
            ax[1, 0].plot(rolling_median, 'r')
            ax[1, 0].set_ylim(0, 1)

    def save_dataframe(df: pd.DataFrame, 
                        file_path: str, 
                        features_inputs: pd.DataFrame,
                        force_new_save: False) -> None:
        """
        Save a DataFrame to a CSV file if the rows in `features_inputs` do not already exist in the file.
        
        Args:
            df (pd.DataFrame): DataFrame to be saved.
            file_path (str): Path to the CSV file.
            features_inputs (pd.DataFrame): DataFrame containing feature inputs to check for existing rows.

        Returns:
            None
        """
        
        existing_results = False
        stored_data = None

        if os.path.exists(file_path) and force_new_save == False:
            stored_data = pd.read_csv(file_path)

            # Check if any rows in `features_inputs` already exist in the file
            existing_rows = stored_data.iloc[:, :len(features_inputs.columns)].to_dict(orient='records')
            new_rows = features_inputs.to_dict(orient='records')[0]

            for check_row in existing_rows:
                n_match = 0
                for element in check_row:
                    if check_row[element] == new_rows[element]:
                        n_match += 1
                if n_match == len(new_rows) :
                    existing_results = True
                    break

        if not existing_results:
            if stored_data is not None:
                # Add `df` to existing data
                combined_data = pd.concat([stored_data, df], ignore_index=True)
            else:
                # Use `df` as the base if the file does not exist
                combined_data = df

            combined_data.to_csv(file_path, index=False)

            print(f"Data saved to: {file_path}")
        else:
            print(f"Some rows from `features_inputs` already exist in {file_path}, data not added.")

    def reorder_label_after_features(df):
        cols = list(df.columns)
        if 'Label' in cols:
            cols.remove('Label')
        new_order = features_cols + ['Label'] + [c for c in cols if c not in features_cols]
        return df[new_order]
    ###################
    ### Compilation ###
    ###################

    data_path = os.path.join('data',sensor,sensor_id,date)
    metadata_path = os.path.join(data_path,'metadata.db')


    STRINGS = {
        "french": {
            "frequency": "fréquence",
            "third": "bandes de tiers d'octaves",
            "octave": "bandes d'octaves",
            "relative": "temps (s)",
            "absolute": "temps",
            "amplitude_db": "amplitude (dB)",
            "dispersion": "dispersion des {} normalisées",
        },
        "english": {
            "frequency": "frequency",
            "third": "third octave bands",
            "octave": "octave bands",
            "relative": "time (s)",
            "absolute": "time",
            "amplitude_db": "amplitude (dB)",
            "dispersion": "dispersion of normalised {} ",
        },
    }


    features_inputs = pd.DataFrame([{
        'sensor': str(sensor),
        'sensor_id': str(sensor_id),
        'channel': int(channel),
        'campaign_name': campaign_name,
        'block_duration': block_duration,
        'band_freq': band_freq
    }])


    start = date_time_obj = datetime.datetime.strptime(date+' '+start_time, '%Y-%m-%d %H:%M:%S').astimezone(pytz.timezone(tz))
    end = date_time_obj = datetime.datetime.strptime(date+' '+end_time, '%Y-%m-%d %H:%M:%S').astimezone(pytz.timezone(tz))

    if sensor == 'magneto':
        metadata,db_url,start_times,file_groups,channel = get_recordings_from_time_range(metadata_path,start,end,tz,recording_type,channel,sensor)

        filenames = [os.path.join(data_path,meta.filename) for meta in metadata]
        files = glob.glob(os.path.join(data_path, "*.wav"))

        recordings = list(set(files).intersection(set(filenames)))
        recordings = sorted(recordings, key=sort_key)

    else: # zoom f4 : faire la synchronisation des pistes à la main en amont, à partir des enregistrements du magneto 


        # Récupérer tous les fichiers WAV correspondant au channel
        recordings = glob.glob(os.path.join(data_path, 'long_format', f"ch{channel}","*.wav"))

        # Extraire les datetime à partir du nom des fichiers
        data = {}
        for path in recordings:
            filename = os.path.basename(path)
            match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', filename)
            if match:
                dt_str = match.group(1)
                dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d_%H:%M:%S")
                data[filename] = dt

        # Créer un DataFrame avec les datetime
        
        start_times = pd.DataFrame.from_dict(data, orient='index', columns=['start_times'])
        
        start_times.index.name = 'filename'
        timezone = pytz.timezone(tz)  # même tz que dans start / end
        start_times['start_times'] = pd.to_datetime(start_times['start_times']).dt.tz_localize(timezone)
        recording_filenames = set(os.path.basename(r) for r in recordings)

        # Filtrer start_times selon la plage temporelle ET la présence dans recordings
        filtered_start_times = start_times[
            (start_times['start_times'] >= start) &
            (start_times.index.isin(recording_filenames))
        ].sort_values(by='start_times', ascending=True)

        first_index = filtered_start_times.index[0]
        recordings = [r for r in recordings if os.path.basename(r) in first_index]
        file_groups = {os.path.basename(path): [os.path.basename(path)] for path in recordings}

    # Vérification si aucun fichier 1.txt n'a été trouvé
    if not recordings:
        raise FileNotFoundError("No recordings. Please check the inputs")

    labels_list = []
    if format_long:
        labels_dir = os.path.join(data_path, 'long_format', 'labels')
    else:
        labels_dir = os.path.join(data_path, 'labels')

    # Variable pour détecter si au moins un fichier 1.txt a été trouvé
    found_label_file = False  

    for subdir in sorted(os.listdir(labels_dir)):
        subdir_path = os.path.join(labels_dir, subdir)
        label_file_path = os.path.join(subdir_path, audacity_label_name)  # audacity_label_name est supposé être '1.txt'
        
        if os.path.exists(label_file_path):
            found_label_file = True  # Au moins un fichier trouvé
            df_label = audacity_label_csv_formatting(label_file_path, start_times, term_replacement, label_removals, format_long, tz)
            labels_list.append(df_label)    

    if not found_label_file:
        logging.warning("Aucun fichier de label '1.txt' n'a été trouvé dans les sous-dossiers de 'labels'.")


    # Concaténation si des fichiers ont été trouvés
    if labels_list:
        df_labels = pd.concat(labels_list, ignore_index=True)
        print(f"Concaténation des labels terminée. Nombre total de labels: {len(df_labels)}")
    else:
        df_labels = pd.DataFrame()


    spectrogram = pd.DataFrame()
    features = pd.DataFrame()


    if apply_filter:
        metadata_df = apply_filter_and_save(recordings, cutoff_freqs, btype, order)
        recordings = [
            os.path.join(os.path.dirname(path), 'filtered_data',f'ch{channel}', os.path.basename(path))
            for path in recordings
        ]

    spectrograms,features,q_factors,leak_detection,rms_exists,freq_limits =  process_data(recordings,channel,start_times,file_groups,freq_limits,block_duration,band_freq,features_list,compute_leak_detection,bandleq_features)

    if df_labels.empty:
        labeled_spectrograms = spectrograms.copy()
        labeled_features = spectrograms.copy()
        labeled_q_factors = spectrograms.copy()
        labeled_leak_detection = spectrograms.copy()
    else:
        labeled_spectrograms,labeled_features,labeled_q_factors,labeled_leak_detection,features = get_frames_in_periods(df_labels,spectrograms,features,q_factors,leak_detection,rms_exists,tz,emergence)

    if save_data:
        result_folder = os.path.join('results', sensor, sensor_id, f'ch{channel}')
        print('Sauvegarde des données dans le répertoire results...')
        os.makedirs(result_folder, exist_ok=True)

        # Save features
        df_reset = labeled_features.reset_index(level=1).reset_index()
        inputs = pd.concat([features_inputs] * len(df_reset)).reset_index(drop=True)
        new_data = pd.concat([inputs, df_reset.reset_index(drop=True)], axis=1)
        save_dataframe(new_data, f'{result_folder}/dataframe_features.csv', features_inputs,force_new_save=False)

        if compute_leak_detection:
            df_reset = labeled_leak_detection.reset_index(level=1).reset_index()
            inputs = pd.concat([features_inputs] * len(df_reset)).reset_index(drop=True)
            new_data = pd.concat([inputs, df_reset.reset_index(drop=True)], axis=1)
            save_dataframe(new_data, f'{result_folder}/dataframe_leak_detection.csv', features_inputs,force_new_save=False)

        if df_labels.empty:
            grouped_spectrograms = labeled_spectrograms.groupby(['freq'])
            has_label = False
        else:
            grouped_spectrograms = labeled_spectrograms.groupby(['Label', 'freq'])
            has_label = True

        quartile_1 = grouped_spectrograms.quantile(0.25).unstack()
        quartile_3 = grouped_spectrograms.quantile(0.75).unstack()
        median_spectrum = grouped_spectrograms.median().unstack()

        # Récupérer l'index multi (Label et freq) si Label existe, sinon juste freq
        if has_label:
            labels = quartile_1.index.get_level_values('Label')
        else:
            labels = pd.Series([None] * len(quartile_1), name='Label')

        inputs_spectrum = pd.concat([features_inputs] * len(quartile_1)).reset_index(drop=True)

        # Ajouter la colonne Label dans les DataFrames
        quartile_1 = quartile_1.reset_index(drop=True)
        quartile_1 = pd.concat([inputs_spectrum.reset_index(drop=True), quartile_1], axis=1)
        quartile_1['Label'] = labels.values


        quartile_3 = quartile_3.reset_index(drop=True)
        quartile_3 = pd.concat([inputs_spectrum.reset_index(drop=True), quartile_3], axis=1)
        quartile_3['Label'] = labels.values


        median_spectrum = median_spectrum.reset_index(drop=True)
        median_spectrum = pd.concat([inputs_spectrum.reset_index(drop=True), quartile_3], axis=1)
        median_spectrum['Label'] = labels.values

        # Fonction pour remettre Label juste après les colonnes de features_inputs
        features_cols = list(features_inputs.columns)

        quartile_1 = reorder_label_after_features(quartile_1)
        quartile_3 = reorder_label_after_features(quartile_3)
        median_spectrum = reorder_label_after_features(median_spectrum)

        save_dataframe(quartile_1, f'{result_folder}/dataframe_1st-quartile-spectrum.csv', features_inputs,force_new_save=True)
        save_dataframe(quartile_3, f'{result_folder}/dataframe_3rd-quartile-spectrum.csv', features_inputs,force_new_save=True)
        save_dataframe(median_spectrum, f'{result_folder}/dataframe_median-spectrum.csv', features_inputs,force_new_save=True)


    features_list = features.columns
    if not df_labels.empty:
        labels = np.concatenate([labeled_features.index.get_level_values('Label').unique(),['other']])
        color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        label_color_map = {label: next(color_cycle) for label in labels}
    else:
        labels = None
        color_cycle = NotImplemented
        label_color_map = None

    if display_features:
        if not features.empty:
            plot_features_with_boxplot(features_list,spectrograms, features,label_color_map, freq_limits,start,end,df_labels, labeled_features, campaign_name,tz,window_rolling)
    if display_leak:
        if not leak_detection.empty:
            plot_features_with_boxplot(['leak'],spectrograms, leak_detection,label_color_map, freq_limits,start,end,df_labels, labeled_leak_detection, campaign_name,tz,window_rolling)
    if display_spectra:
        plot_spectra(labeled_spectrograms,label_color_map,freq_limits)


    if Q_factor:
        
        plot_features_with_boxplot(['Q-prod','spectrum correlation'],spectrograms, q_factors,label_color_map, freq_limits,start,end,df_labels, labeled_q_factors, campaign_name,tz,window_rolling)

        for label in labeled_q_factors.index.get_level_values('Label').unique():

            specific_time = labeled_spectrograms.index.get_level_values('time')[0]
            filtered_data = labeled_spectrograms.xs(label, level='Label')

            # Calculer le percentile 50 pour chaque fréquence
            df_middle_percentile = pd.DataFrame(
                filtered_data.groupby('freq')['bandleq'].quantile(0.5)
            )

            # Réindexer pour ajouter 'time' comme niveau 0
            df_middle_percentile.index.name = 'freq'
            df_middle_percentile['time'] = specific_time
            df_middle_percentile = df_middle_percentile.reset_index().set_index(['time', 'freq'])

            q_factors_median = Q_factors(df_middle_percentile,freq_limits,band_freq)

            x_peaks = q_factors_median.loc[specific_time]['peak freq.']
            y_peaks = q_factors_median.loc[specific_time]['peak mag.']
            q_peaks = q_factors_median.loc[specific_time]['Q-factor']

            # Tracer les résultats
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(df_middle_percentile.index.get_level_values('freq'), df_middle_percentile['bandleq'], label='Spectrogram Data')
            ax.plot(x_peaks, y_peaks, 'or', label='Top Peaks')
            ax.set_xlim(freq_limits)
            ax.set_ylim(-10,50)
            # Annoter chaque pic avec la valeur associée de 'Q-factor'
            for (x, y, q) in zip(x_peaks, y_peaks, q_peaks):
                # Annoter les points avec flèches          
                ax.annotate(
                    f'{round(q,2)}',  # Texte à afficher
                    (x, y),  # Position du texte
                    textcoords="offset points",  # Les coordonnées du texte sont un décalage par rapport au point
                
                    xytext=(0, 20),  # Décalage du texte par rapport au point
                    ha='center',  # Alignement horizontal
                    va='bottom',  # Alignement vertical
                    fontsize=12,
                    color='red',
                )

            ax.set_xlabel('Frequency')
            ax.set_ylabel('Amplitude')
            ax.set_title(label)
            ax.legend()
            ax.grid()

            plt.show()


 # %%
