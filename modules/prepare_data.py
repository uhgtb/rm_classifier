import numpy as np
import gc
import warnings
from scipy import signal

def find_bin(frequency, freq_bins):
    """
    Find the bin index of a given frequency in a list of frequency bins.
    Args:
        frequency (float): The frequency to find.
        freq_bins (array-like): The array of frequency bins.
    Returns:
        int: The index of the bin closest to the given frequency.
    """
    return np.argmin(np.abs(freq_bins - frequency))

def cut_beacon_frequency_spectra(spectra, beacon_frequencies, freq_bins=None, peak_width=3):
    """
    Cut the spectra around the beacon frequencies.
    Args:           
        spectra (numpy.ndarray): The spectral data
        beacon_frequencies (list or array): Frequencies to cut out
        freq_bins (numpy.ndarray, optional): Frequency bins used in the spectra
        peak_width (int or list, optional): Width of the peak(s) to cut. If a list, must be the same length as beacon_frequencies.
    Returns:
        numpy.ndarray: The spectra with the beacon frequencies cut out.
    """
    # Convert peak_width to a list if it's a single value

    peak_width = int(np.ceil(peak_width))
    if not isinstance(peak_width, (list, np.ndarray)):
        peak_width = [peak_width] * len(beacon_frequencies)
    
    # Ensure peak_width and beacon_frequencies have the same length
    if len(peak_width) != len(beacon_frequencies):
        raise ValueError("If peak_width is a list, it must have the same length as beacon_frequencies")
        
    for i, frequency in enumerate(beacon_frequencies):
        pw = peak_width[i]
        freq_bin = find_bin(frequency, freq_bins)+1
        avg_values = (spectra[:,freq_bin-pw-1] + spectra[:,freq_bin+pw+1])/2
        spectra[:,freq_bin-pw:freq_bin+pw] = avg_values[:, np.newaxis] * np.ones((1, 2*pw))
    gc.collect()
    return spectra

def cut_beacon_frequency_timeseries(signal, beacon_frequencies, freq_bins=None, peak_width=3):
    """
    Cut the time series around the beacon frequencies using the hard cuts in the fft spectra.
    Args:
        signal (numpy.ndarray): The time series data
        beacon_frequencies (list or array): Frequencies to cut out
        freq_bins (numpy.ndarray, optional): Frequency bins used in the FFT of the time series
        peak_width (int or list, optional): Width of the peak(s) to cut. If a list, must be the same length as beacon_frequencies.
    Returns:
        numpy.ndarray: The time series with the beacon frequencies cut out.
    """	
    # process in chunks to avoid memory issues
    signal = signal.copy()  # Avoid modifying the original data
    batch_size = 1000
    n_batches = (len(signal) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(signal))

        signal_fft = np.fft.rfft(signal[start_idx:end_idx])+1e-10
        # Convert peak_width to a list if it's a single value
        if not isinstance(peak_width, (list, np.ndarray)):
            peak_width = [peak_width] * len(beacon_frequencies)
        
        # Ensure peak_width and beacon_frequencies have the same length
        if len(peak_width) != len(beacon_frequencies):
            raise ValueError("If peak_width is a list, it must have the same length as beacon_frequencies")
        
        for i, frequency in enumerate(beacon_frequencies):
            pw = peak_width[i]
            freq_bin = find_bin(frequency, freq_bins)+1
            avg_values = (abs(signal_fft)[:,freq_bin-pw-1] + abs(signal_fft)[:,freq_bin+pw+1])/2
            signal_fft[:,freq_bin-pw:freq_bin+pw] *= (avg_values[:, np.newaxis] * np.ones((1, 2*pw))/abs(signal_fft)[:,freq_bin-pw:freq_bin+pw])
        signal[start_idx:end_idx] = np.fft.irfft(signal_fft, axis=1)
    return signal

def calculate_n_frequency_bins(input_data_type, len_input_data, suppress_dc=False): 
    """
    Calculate the number of frequency bins and time bins based on the input data type and length.
    Args:
        input_data_type (str): Type of the input data. Must be one of 'fft', 'time', 'fft_phase', or 'fft_time'.
        len_input_data (int): Length of the input data (number of columns).
        suppress_dc (bool): Whether the DC component is suppressed.
    Returns:
        tuple: (n_fft, n_time) where n_fft is the number of frequency bins and n_time is the number of time bins.
    """
    if input_data_type == "fft":
        n_fft, n_time = len_input_data, 2*(len_input_data + int(suppress_dc) - 1)
    elif input_data_type == "time":
        n_fft, n_time = len_input_data // 2 + 1 - int(suppress_dc), len_input_data
    elif input_data_type == "fft_phase":
        if len_input_data % 2 == 0:
            n_fft = len_input_data // 2
            n_time = 2*(n_fft - 1 + int(suppress_dc))
        else:
            raise ValueError("fft_phase data must have an even number of columns for conversion to FFT.")
    elif input_data_type == "fft_time":
        if (len_input_data + int(suppress_dc)) % 3 == 1:
            n_fft = (len_input_data + 2 ) // 3
        elif (len_input_data + int(suppress_dc)) % 3 == 2:
            n_fft = (len_input_data + 1) // 3
        else:
            raise ValueError("fft_time data must have a shape, where n_fft_bins = n_time_bins // 2 +1")
        n_time = len_input_data - n_fft
    else:
        raise ValueError(f"Invalid input_data_type: {input_data_type}. Must be one of 'fft', 'time', 'fft_phase', or 'fft_time'.")
    return n_fft, n_time

def denoiser(data,n_rolling_average=10, n_peak=3, n_fft=None):
    """
    Remove statistical noise from the data using a rolling average, without smoothing out peaks.
    Args:
        data (numpy.ndarray): The input data to be denoised.
        n_rolling_average (int): Number of bins for the rolling average.
        n_peak (int): Number of bins for the peak, which are excluded from the rolling average calculation.
        n_fft (int, optional): Number of frequency bins. If None, it is set to the number of columns in data.
    Returns:
        numpy.ndarray: The denoised data.
    """

    if n_fft is None:
        n_fft = data.shape[1]
    def cut_function(x, cut1, cut2):
        return np.where(x < cut1, 0, np.where(x > cut2, 1, (x - cut1) / (cut2 - cut1)))
    conv= np.ones(2*n_rolling_average+1)/(2*n_rolling_average-2*n_peak)
    conv[n_rolling_average-n_peak:-n_rolling_average+n_peak]=np.zeros(n_peak*2+1) # calculate the rolling average with n_pak bins excluded in the middle
    
    # process in chunks to avoid memory issues
    batch_size = 1000
    n_batches = (data.shape[0] + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, data.shape[0])
        
        # calculate rolling average for the current batch
        rolling_average = np.apply_along_axis(lambda x: np.convolve(x, conv, mode='same'), 1, data[start_idx:end_idx, :n_fft])
        
        # calculate alpha, which is the weight of the rolling average
        alpha = cut_function(abs(data[start_idx:end_idx, :n_fft]-rolling_average)/(abs(rolling_average)+1e-9), 1, 2.5)
        alpha[:,1:-1] = np.maximum(alpha[:,:-2], alpha[:,1:-1], alpha[:,2:])
        alpha = np.clip(alpha[:, (n_rolling_average+1):n_fft-(n_rolling_average+1)], 0, 1)  # Ensure alpha is between 0 and 1 and has the same shape as the data after cutting the edges
        # apply the rolling average to the data
        data[start_idx:end_idx, (n_rolling_average+1):n_fft-(n_rolling_average+1)] *= alpha #keep the edges unchanged
        data[start_idx:end_idx, (n_rolling_average+1):n_fft-(n_rolling_average+1)] += (1-alpha) * rolling_average[:,(n_rolling_average+1):n_fft-(n_rolling_average+1)]
    
    return data # return weighted denoised data

def welch_abs_fft(input_time_trace, params):
    """
    Calculate the absolute FFT of the input time trace using either standard FFT or Welch's method.
    Args:
        input_time_trace (numpy.ndarray): The input time trace data.
        params (dict): Dictionary containing parameters for the FFT calculation, including:
            - "spectrum_filter": Method to use for FFT calculation. Must be either None or "welch".
            - "sampling_frequency": Sampling frequency of the input time trace.
            - "welch_nperseg": Length of each segment for Welch's method.
            - "welch_window": Type of window to use for Welch's method.
            - "welch_average": Method to use for averaging in Welch's method.
            - "welch_noverlap": Number of overlapping points between segments for Welch's method.
    Returns:
        numpy.ndarray: The absolute FFT of the input time trace.
    """
    if params["spectrum_filter"] != "welch":
        return np.abs(np.fft.rfft(input_time_trace, axis=1))
    else:
        print("Applying Welch's method to calculate absolute FFT.")
        psd=signal.welch(input_time_trace, 
                         fs = params["sampling_frequency"], 
                         nperseg = params["welch_nperseg"], 
                         window = params["welch_window"], 
                         average = params["welch_average"], 
                         noverlap = params["welch_noverlap"], 
                         axis=1)[1]
        win = signal.get_window(params["welch_window"], params["welch_nperseg"])
        if win is not None:
            psd *= np.sum(win**2) * params["sampling_frequency"]/2 # Normalize the power spectral density
        print(np.sum(win**2), params["sampling_frequency"])
        psd = np.clip(psd, 0, None)  # Avoid log(0) issues
        return np.sqrt(psd)
    
def prepare_data(umap_classifier, data, verbose = True):
    """
    Prepare the data according to the specifications in the umap_classifier object.
    This includes cutting beacon frequencies, applying window functions, converting data types,
    suppressing DC components, applying denoising and log filters.
    Args:
        umap_classifier (_umap_classifier_): An umap_classifier object containing data preparation specifications.
        data (numpy.ndarray): The input data to be prepared.
        verbose (bool): Whether to print progress messages.
    Returns:
        numpy.ndarray: The prepared data.
    """
    input_data_type = umap_classifier.input_data_type
    target_data_type = umap_classifier.data_preparation.get("target_data_type", input_data_type)
    params = umap_classifier.data_preparation
    # Check for invalid input data types
    if input_data_type not in ["fft", "time", "fft_phase", "fft_time"]:
        raise ValueError(f"Invalid input_data_type: {input_data_type}. Must be one of 'fft', 'time', 'fft_phase', or 'fft_time'.")
    if target_data_type not in ["fft", "time", "fft_phase", "fft_time"]:
        raise ValueError(f"Invalid target_data_type: {target_data_type}. Must be one of 'fft', 'time', 'fft_phase', or 'fft_time'.")    
    if params["spectrum_filter"] not in [None, "welch", "denoiser"]:
        raise ValueError(f"Invalid spectrum_filter: {params['spectrum_filter']}. Must be None, 'welch', or 'denoiser'.")
    if params["windowing"] not in [None, "hamming", "hanning", "blackman", "blackmanharris", "bartlett", "kaiser"]:
        raise ValueError(f"Invalid windowing: {params['windowing']}. Must be None, 'hamming', 'hanning', 'blackman', 'blackmanharris', 'bartlett', or 'kaiser'.")
    if params["spectrum_filter"] == "welch" and input_data_type != "time":
        raise ValueError("Welch filter can only be applied to time traces as input data. Specify input_data_type as 'time'.")

    # Cut beacon frequencies if specified
    if params["cut_beacon_frequencies"]:
        n_fft, _ = calculate_n_frequency_bins(input_data_type, data.shape[1])
        if not params["beacon_frequencies"]:
            raise ValueError("Beacon frequencies must be specified if cut_beacon_frequencies is True.")
        if not params["beacon_width"]:
            params["beacon_width"] = 3  # Default width if not specified
            warnings.warn("Beacon width not specified. Using default value of 3.")
        if isinstance(params["frequency_bins"], type(None)):
            params["frequency_bins"] = np.linspace(0, params["sampling_frequency"]/2, n_fft)  # Default frequency bins if not specified
            warnings.warn(f"Frequency bins not specified. Using default frequency bins from 0 to {params['sampling_frequency']/2} MHz with {n_fft} bins.")
        if input_data_type == "fft":
            data = cut_beacon_frequency_spectra(data, params["beacon_frequencies"], freq_bins=params["frequency_bins"],peak_width=params["beacon_width"])
        elif input_data_type == "time":
            data = cut_beacon_frequency_timeseries(data, params["beacon_frequencies"], freq_bins=params["frequency_bins"], peak_width=params["beacon_width"])
        elif input_data_type == "fft_time":
            data_fft = cut_beacon_frequency_spectra(data[:, :n_fft], params["beacon_frequencies"], freq_bins=params["frequency_bins"], peak_width=params["beacon_width"])
            data_time = cut_beacon_frequency_timeseries(data[:, n_fft:], params["beacon_frequencies"], freq_bins=params["frequency_bins"], peak_width=params["beacon_width"])
            data = np.concatenate([data_fft, data_time], axis=1)
            del data_fft, data_time
            gc.collect()
        elif input_data_type == "fft_phase":
            data_fft = cut_beacon_frequency_spectra(data[:, :n_fft], params["beacon_frequencies"], freq_bins=params["frequency_bins"], peak_width=params["beacon_width"])
            data_phase = data[:, n_fft:]
            data = np.concatenate([data_fft, data_phase], axis=1)
            del data_fft, data_phase
            gc.collect()
    
    # apply window functions to the time traces
    if params["windowing"]:
        if input_data_type == "time":
            if verbose:
                print(f"Applying {params['windowing']} windowing to the time traces.")
            data = data - np.mean(data, axis=1, keepdims=True)  # Remove DC component
            if params["windowing"] == "hamming":
                window = np.hamming(data.shape[1])
            elif params["windowing"] == "hanning":
                window = np.hanning(data.shape[1])
            elif params["windowing"] == "blackman":
                window = np.blackman(data.shape[1])
            elif params["windowing"] == "blackmanharris":
                window = np.blackmanharris(data.shape[1])
            elif params["windowing"] == "bartlett":
                window = np.bartlett(data.shape[1])
            elif params["windowing"] == "kaiser":
                window = np.kaiser(data.shape[1], beta=5)
            else:
                raise ValueError(f"Invalid window type: {params['windowing']}. Must be one of 'hamming', 'hanning', 'blackman', 'blackmanharris', 'bartlett', or 'kaiser'.")
            
            data = data * window
            del window
        else:
            raise ValueError("Windowing can only be applied to time traces as input data. Specify input_data_type as 'time'.")
    
    # finally convert the data to the target data type
    if input_data_type == "fft":
        if not target_data_type == "fft":
            raise ValueError("FFT data can only be converted to 'fft' target data type. Specify target_data_type as 'fft'.")        
    elif input_data_type == "time":
        if target_data_type == "fft":
            data = welch_abs_fft(data, params)
        elif target_data_type == "fft_phase":
            phases = np.angle(np.fft.rfft(data, axis=1))
            abs_values = welch_abs_fft(data, params)
            data = np.concatenate([abs_values, phases], axis=1)
            del phases, abs_values
            gc.collect()
        elif target_data_type == "fft_time":
            fft_data = welch_abs_fft(data, params)
            data = np.concatenate([fft_data, data], axis=1)
            del fft_data
            gc.collect()
    elif input_data_type == "fft_time":
        n_fft, _ = calculate_n_frequency_bins(input_data_type, data.shape[1])
        if target_data_type == "fft":
            data = data[:, :n_fft]
        elif target_data_type == "time":
            data = data[:, n_fft:]
        elif target_data_type == "fft_phase":
            data = np.concatenate([data[:, :n_fft], np.angle(np.fft.rfft(data[:, n_fft:], axis=1))], axis=1)
    elif input_data_type == "fft_phase":
        n_fft, _ = calculate_n_frequency_bins(input_data_type, data.shape[1])
        if target_data_type == "fft":
            data = data[:, :n_fft]  # Keep only absolute values of the fft
        elif target_data_type == "time":
            data = np.fft.irfft(data[:, :n_fft] * np.exp(1j * data[:, n_fft:]), axis=1)
        elif target_data_type == "fft_time":
            data = np.concatenate([data[:, :n_fft], np.fft.irfft(data[:, :n_fft] * np.exp(1j * data[:, n_fft:]), axis=1)], axis=1)
    
    #claculate n_fft and n_time
    if params["spectrum_filter"] == "welch":
        n_fft = params["welch_nperseg"] // 2 + 1  # Adjust n_fft for Welch filter
        n_time = data.shape[1] - n_fft
    else:
        n_fft, n_time = calculate_n_frequency_bins(target_data_type, data.shape[1])

    #apply suppress_dc if specified
    if params["suppress_dc"]:
        if target_data_type == "time":
            data = data - np.mean(data, axis=1, keepdims=True)
        elif target_data_type == "fft":
            data = data[:, 1:]
            n_fft -= 1
        elif target_data_type =="fft_phase":
            data = np.concatenate([data[:, 1:n_fft], data[:, n_fft+1:]], axis=1)
            n_fft -= 1
        elif target_data_type == "fft_time":
            data[:, n_fft:] = data[:, n_fft:] - np.mean(data[:, n_fft:], axis=1, keepdims=True)
            data = data[:, 1:]
            n_fft -= 1
    # apply denoiser if specified
    if params["spectrum_filter"] == "denoiser":
        if target_data_type == "time":
            raise ValueError("Denoiser can only be applied to FFT data. Specify target_data_type as 'fft', 'fft_phase', or 'fft_time'.")
        else:
            data = denoiser(data, n_rolling_average=params["denoiser_n"], n_peak=params["denoiser_npeak"], n_fft=n_fft)

    # apply log filter if specified
    if params["log_filter"]:
        if target_data_type == "time":
            raise ValueError("Log filter can only be applied to FFT data. Specify target_data_type as 'fft', 'fft_phase', or 'fft_time'.")
        else:
            if input_data_type == "fft" and target_data_type == "fft":
                data = data.copy()  # Avoid modifying the original data
            data[:, :n_fft] = np.log(data[:, :n_fft] + 1e-10)  # Add small value to avoid log(0)
    return data

def normalize_data(umap_classifier, data, verbose = True):
    """
    Normalize the data using the mean and standard deviation from the training data.
    If the mean and std are not provided, they are calculated from the provided data.
    Args:
        umap_classifier (_umap_classifier_): An umap_classifier object containing normalization specifications.
        data (numpy.ndarray): The input data to be normalized.
        verbose (bool): Whether to print progress messages.
    Returns:
        tuple: (normalized_data, params) where normalized_data is the normalized data and params is the updated data preparation parameters.
    """
    params = umap_classifier.data_preparation
    if params["normalization"]:
        if "rd_train_mean" not in params:
            print("No rd_train_std provided. Calculating std from the provided prepared data.")
            params["rd_train_mean"] = np.mean(data, axis=0)
        if "rd_train_std" not in params:
            print("No rd_train_std provided. Calculating std from the provided prepared data.")
            params["rd_train_std"] = np.std(data, axis=0)
        else:
            print("Using previous rd_train_mean and rd_train_std for normalization.")

        data -= params["rd_train_mean"]
        data /= (params["rd_train_std"] + 1e-10)  # Add small value to avoid division by zero
    return data, params

def pooling_data(umap_classifier, data, verbose = True):
    """
    Apply pooling to the data according to the specifications in the umap_classifier object.
    Args:
        umap_classifier (_umap_classifier_): An umap_classifier object containing pooling specifications.
        data (numpy.ndarray): The input data to be pooled.
        verbose (bool): Whether to print progress messages.
    Returns:
        numpy.ndarray: The pooled data."""  
    params = umap_classifier.data_preparation
    if not params["max_pooling"] and not params["avg_pooling"]:
        return data
    if params["max_pooling"] and params["avg_pooling"]:
        raise ValueError("Cannot apply both avg_pooling and max_pooling at the same time. Choose one.")
    
    if params["avg_pooling"]:
        if verbose:
            print(f"Applying average pooling with pool size {params['avg_pooling']}.")
        if data.shape[1] % params["avg_pooling"] != 0:
            warnings.warn(f"Data length {data.shape[1]} is not a multiple of avg_pooling {params['avg_pooling']}. Truncating data to the nearest multiple.")
            max_idx = (data.shape[1] // params["avg_pooling"]) * params["avg_pooling"]
            data = data[:, :max_idx]  # Ensure the data length is a multiple of avg_pooling
        data = np.mean(data.reshape(data.shape[0], -1, params["avg_pooling"]), axis=2)
    
    if params["max_pooling"]:
        if verbose:
            print(f"Applying max pooling with pool size {params['max_pooling']}.")
        if data.shape[1] % params["max_pooling"] != 0:
            warnings.warn(f"Data length {data.shape[1]} is not a multiple of max_pooling {params['max_pooling']}. Truncating data to the nearest multiple.")
            max_idx = (data.shape[1] // params["max_pooling"]) * params["max_pooling"]
            data = data[:, :max_idx]  # Ensure the data length is a multiple of max_pooling
        data = np.max(data.reshape(data.shape[0], -1, params["max_pooling"]), axis=2)
    
    return data