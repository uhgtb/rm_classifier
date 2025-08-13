import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, List, Optional
import warnings

def generate_gaussian_spectrum(freq_bins: np.ndarray, 
                             n_gaussians: int = None, 
                             freq_range: Tuple[float, float] = None,
                             seed: int = None) -> np.ndarray:
    """
    Generate a spectral shape as a sum of Gaussians with random parameters.
    
    Parameters:
    -----------
    freq_bins : np.ndarray
        Frequency bins for the spectrum
    n_gaussians : int, optional
        Number of Gaussians to sum (random between 1-15 if None)
    freq_range : tuple, optional
        (min_freq, max_freq) for Gaussian centers. Uses freq_bins range if None
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray : Spectral amplitude values
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_gaussians is None:
        n_gaussians = np.random.randint(1, 16)  # 1 to 15 Gaussians
    
    if freq_range is None:
        freq_range = (freq_bins.min(), freq_bins.max())
    
    spectrum = np.zeros_like(freq_bins)
    
    for _ in range(n_gaussians):
        # Random Gaussian parameters
        mean = np.random.uniform(freq_range[0], freq_range[1])
        std = np.random.uniform(0.01, 0.3) * (freq_range[1] - freq_range[0])
        intensity = np.random.uniform(0.1, 2.0)*1e4
        
        # Add Gaussian to spectrum
        gaussian = intensity * np.exp(-0.5 * ((freq_bins - mean) / std) ** 2)
        spectrum += gaussian
    
    return spectrum

def create_clustered_spectra(base_spectrum: np.ndarray, 
                           n_samples: int, 
                           noise_level: float = 0.1,
                           seed: int = None) -> List[np.ndarray]:
    """
    Create n similar spectra based on a base spectrum with added noise.
    
    Parameters:
    -----------
    base_spectrum : np.ndarray
        Base spectral shape
    n_samples : int
        Number of samples to create
    noise_level : float
        Relative noise level (0-1)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    List[np.ndarray] : List of similar spectra
    """
    if seed is not None:
        np.random.seed(seed)
    
    clustered_spectra = []
    
    for _ in range(n_samples):
        # Add multiplicative and additive noise
        multiplicative_noise = 1 + noise_level * np.random.normal(0, 0.3, base_spectrum.shape)
        additive_noise = noise_level * np.random.normal(0, np.max(base_spectrum) * 0.1, base_spectrum.shape)
        
        noisy_spectrum = base_spectrum * multiplicative_noise + additive_noise
        # Ensure non-negative amplitudes
        noisy_spectrum = np.maximum(noisy_spectrum, 0.01 * np.max(noisy_spectrum))
        
        clustered_spectra.append(noisy_spectrum)
    
    return clustered_spectra

def generate_fourier_time_traces(n_samples: int,
                                n_time_points: int = 1024,
                                sampling_rate: float = 1.0,
                                max_gaussians: int = 15,
                                noise_level: float = 0.1,
                                time_domain_noise: float = 0.05,
                                freq_range: Optional[Tuple[float, float]] = None,
                                seed: int = None) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray]]:
    """
    Generate time traces and Fourier spectra with random phases and noise,
    where amplitudes come from a random clusters based on sums of Gaussians.
    
    Parameters:
    -----------
    n_samples : int
        Number of similar spectral clusters
    n_time_points : int, default=1024
        Length of time series
    sampling_rate : float, default=1.0
        Sampling rate for time series
    max_gaussians : int, default=15
        Maximum number of Gaussians in base spectrum
    noise_level : float, default=0.1
        Noise level for spectral clustering (0-1)
    time_domain_noise : float, default=0.05
        Additional noise in time domain (0-1)
    freq_range : tuple, optional
        (min_freq, max_freq) for analysis. Uses Nyquist range if None
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (frequencies, clustered_spectra, time_axis, time_traces)
        - frequencies: np.ndarray of frequency bins
        - clustered_spectra: List of spectral amplitudes for each cluster
        - time_axis: np.ndarray of time points
        - time_traces: List of time domain signals for each cluster
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create frequency axis
    freqs = (-1)*np.fft.fftfreq(n_time_points, 1/sampling_rate)
    positive_freqs = freqs[freqs >= 0]
    
    if freq_range is None:
        freq_range = (0, sampling_rate/2)  # DC to Nyquist
    
    # Select k frequency bins within the specified range
    freq_mask = (positive_freqs >= freq_range[0]) & (positive_freqs <= freq_range[1])
    available_freqs = positive_freqs[freq_mask]
    
    k_bins = len(available_freqs)
    
    # Select k bins uniformly or randomly
    if k_bins < len(available_freqs):
        selected_indices = np.linspace(0, len(available_freqs)-1, k_bins, dtype=int)
        selected_freqs = available_freqs[selected_indices]
    else:
        selected_freqs = available_freqs
    
    # Generate base spectrum using sum of Gaussians
    n_gaussians = np.random.randint(4, max_gaussians + 1)
    base_spectrum = generate_gaussian_spectrum(
        selected_freqs, 
        n_gaussians=n_gaussians,
        freq_range=freq_range,
        seed=seed
    )
    
    # Create clustered spectra
    clustered_spectra = create_clustered_spectra(
        base_spectrum, 
        n_samples, 
        noise_level=noise_level,
        seed=seed
    )
    
    # Generate time traces
    time_traces, phases = [], []
    time_axis = np.arange(n_time_points) / sampling_rate
    
    for i, spectrum in enumerate(clustered_spectra):
        
        # generate random phase for each frequency bin
        phase = np.random.uniform(-np.pi, np.pi, len(selected_freqs))
        
        # Inverse FFT to get time domain signal
        time_trace = np.fft.irfft(spectrum * np.exp(1j * phase), n=n_time_points)
        
        # Add time domain noise
        if time_domain_noise > 0:
            noise = time_domain_noise * np.std(time_trace) * np.random.normal(0, 1, len(time_trace))
            time_trace += noise
        
        time_traces.append(time_trace)
        phases.append(phase)
    
    return time_traces, clustered_spectra, phases, time_axis, selected_freqs 

def generate_random_rm_traces(n_samples: int,
                              n_clusters: int = 4,
                              outlier_fraction: float = 0.01,
                              n_time_points: int = 1024,
                              sampling_rate: float = 1.0,
                              max_gaussians: int = 15,
                              noise_level: float = 0.1,
                              time_domain_noise: float = 0.05,
                              freq_range: Optional[Tuple[float, float]] = None,
                              seed: int = None,
                              type: str = "all"):
    n_samples_per_cluster = int((n_samples*(1-outlier_fraction)) // n_clusters)
    n_outliers = int(n_samples - n_samples_per_cluster * n_clusters)
    all_clustered_spectra, all_time_traces, all_phases, all_cluster_idx = [], [], [], []
    for i in range(n_clusters):
        time_traces, clustered_spectra, phase, time_axis, selected_freqs = generate_fourier_time_traces(
            n_samples=n_samples_per_cluster,
            n_time_points=n_time_points,
            sampling_rate=sampling_rate,
            max_gaussians=max_gaussians,
            noise_level=noise_level,
            time_domain_noise=time_domain_noise,
            freq_range=freq_range,
            seed=seed
        )
        all_clustered_spectra.extend(clustered_spectra)
        all_time_traces.extend(time_traces)
        all_cluster_idx.extend([i] * n_samples_per_cluster)
        all_phases.extend(phase)
    
    # Generate outliers
    for i in range(n_outliers):
        time_traces, clustered_spectra, phase, time_axis, selected_freqs  = generate_fourier_time_traces(
            n_samples=1,
            n_time_points=n_time_points,
            sampling_rate=sampling_rate,
            max_gaussians=max_gaussians,
            noise_level=noise_level,
            time_domain_noise=time_domain_noise,
            freq_range=freq_range,
            seed=seed + i if seed is not None else None
        )
        all_clustered_spectra.extend(clustered_spectra)
        all_time_traces.extend(time_traces)
        all_cluster_idx.extend([-1])  # Mark as outlier
        all_phases.extend(phase)
    
    # Convert to numpy arrays
    all_clustered_spectra = np.array(all_clustered_spectra)
    all_time_traces = np.array(all_time_traces)
    all_cluster_idx = np.array(all_cluster_idx)
    all_phases = np.array(all_phases)
    if type == "all":
        return all_time_traces, all_clustered_spectra, all_phases, all_cluster_idx, time_axis, selected_freqs
    elif type == "fft":
        return all_clustered_spectra, all_cluster_idx, selected_freqs
    elif type == "time":
        return all_time_traces, all_cluster_idx, time_axis
    elif type == "fft_phase":
        return all_clustered_spectra, all_phases, all_cluster_idx, selected_freqs
    elif type == "fft_time":
        return all_time_traces, all_clustered_spectra, all_phases, all_cluster_idx, time_axis, selected_freqs
    else:
        raise ValueError(f"Unknown type: {type}. Must be one of 'all', 'fft', 'time', 'fft_phase', or 'fft_time'.")
    
    