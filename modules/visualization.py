import numpy as np
import matplotlib.pyplot as plt
import prepare_data

def create_axes_configurations():
    """
    Predefined configurations for different plotting scenarios
    """
    return {
        'time': {
            'x_label': "t [ns]",
            'y_label': r"ADC",
            'x_func': lambda n_time, n_fft, sampling_frequency, suppress_dc: np.linspace(0, (n_time-1)/sampling_frequency, n_time),
            'y_scale': "linear"
        },
        'fft': {
            'x_label': "f [MHz]",
            'y_label': r"$|X(f)|$",
            'x_func': lambda n_time, n_fft, sampling_frequency, suppress_dc: np.linspace((suppress_dc*sampling_frequency)/(2*n_fft), sampling_frequency/2, n_fft),
            'y_scale': "log"
        },
        'phase': {
            'x_label': "f [MHz]",
            'y_label': r"$\phi$",
            'x_func': lambda n_time, n_fft, sampling_frequency, suppress_dc: np.linspace((suppress_dc*sampling_frequency)/(2*n_fft), sampling_frequency/2, n_fft),
            'y_scale': "linear"
        }
    }

def default_plot_types(input_data_type, plot_types):
    # default plot types
    if plot_types=="default":
        if input_data_type=="time":
            plot_types=["time"]
        elif input_data_type=="fft":
            plot_types=["fft"]
        elif input_data_type=="fft_time":
            plot_types=["fft","time"]
        elif input_data_type=="fft_phase":
            plot_types=["fft","phase"]
    return plot_types

def plot_idx(plot_type, data_type, n_fft, n_time, suppress_dc):
    """
    Determine the start and end indices for plotting based on the plot type.
    """
    if plot_type == "fft":
        if data_type in ["fft", "fft_phase", "fft_time"]:
            return 0, n_fft
        else:
            raise ValueError(f"FFT plot type is not applicable for data type: {data_type}")
    elif plot_type == "phase":
        if data_type == "fft_phase":
            return n_fft, n_fft * 2
        else:
            raise ValueError(f"Phase plot type is only applicable for 'fft_phase' data type, got: {data_type}")
    elif plot_type == "time":
        if data_type in ["time", "fft_time"]:
            return n_fft, n_fft + n_time
        else:
            raise ValueError(f"Time plot type is not applicable for data type: {data_type}")
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def plot_spectra(umap_classifier, k,
                 plot_types="default", 
                 modifier_function_name="",
                 sigma=1, 
                 q=None, 
                 save_fig=None):
    
    data_type = umap_classifier.data_preparation.get("target_data_type", "time")
    sampling_frequency = umap_classifier.data_preparation.get("sampling_frequency", 180)
    suppress_dc = umap_classifier.data_preparation.get("suppress_dc", False)
    n_fft, n_time = prepare_data.calculate_n_frequency_bins(data_type, umap_classifier.get_cluster_trace(k, trace_statistic="median", modifier_function_name=modifier_function_name).shape[0], suppress_dc=suppress_dc)

    plot_types=default_plot_types(data_type,plot_types)
    # Get axes configurations
    axes_configs = create_axes_configurations()
    
    # Prepare figure based on number of plot types
    fig, axes = plt.subplots(1, len(plot_types), figsize=(6*len(plot_types), 4))
    
    # Ensure axes is always a list
    if len(plot_types) == 1:
        axes = [axes]
    

    # Compute statistical metrics   
    if q:
        qup_array =  umap_classifier.get_cluster_trace(k, trace_statistic=f"quantile", modifier_function_name=modifier_function_name, q=q)
        qdown_array = umap_classifier.get_cluster_trace(k, trace_statistic=f"quantile", modifier_function_name=modifier_function_name, q=1-q)
        median_array = umap_classifier.get_cluster_trace(k, trace_statistic="median", modifier_function_name=modifier_function_name)
    if sigma:
        mean_array = umap_classifier.get_cluster_trace(k, trace_statistic="mean", modifier_function_name=modifier_function_name)
        std_array = umap_classifier.get_cluster_trace(k, trace_statistic="std", modifier_function_name=modifier_function_name)
        std_up_array = mean_array + sigma*std_array
        std_down_array = mean_array - sigma*std_array
    
    # Plot each specified type
    for i, plot_type in enumerate(plot_types):
        config = axes_configs[plot_type]
        start_idx, end_idx = plot_idx(plot_type, data_type, n_fft, n_time, suppress_dc)
        
        # Get x-axis values based on the configuration
        x = config['x_func'](n_time, n_fft, sampling_frequency, suppress_dc)
        
        # Plotting
        ax = axes[i]
        ax.set_xlabel(config['x_label'], fontsize=14)
        ax.set_ylabel(config['y_label'], fontsize=14)
        ax.set_yscale(config['y_scale'])
        if q:    
            ax.fill_between(x, qup_array[start_idx:end_idx], qdown_array[start_idx:end_idx], color='blue', alpha=0.3, label=f'{q}-quantiles')
            ax.plot(x, median_array[start_idx:end_idx], 'r-', label='median')
        if sigma:
            ax.plot(x, mean_array[start_idx:end_idx], 'k-', label='mean')  
            ax.plot(x, std_up_array[start_idx:end_idx], 'b--', label=f'{sigma} stds')
            ax.plot(x, std_down_array[start_idx:end_idx], 'b--')


    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig)
        print(f"Saved figure to {save_fig}")
    else:
        plt.show()
    return True