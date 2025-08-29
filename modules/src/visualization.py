import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prepare_data
import warnings
from matplotlib.lines import Line2D
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, LinearColorMapper
from bokeh.plotting import figure, show, output_notebook, output_file, save
from bokeh.palettes import Viridis256


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
    """Defines default plot types based on input data type.

    Args:
        input_data_type (list of str): list of input data types
        plot_types (list of str): list of plot types or string "default"

    Returns:
        list of str: list of plot types
    """
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
    """Plots the spectra of a given cluster with various statistical metrics.

    Args:
        umap_classifier (_umap_classifier_): The UMAP classifier object containing data and methods.
        k (int, str): cluster index
        plot_types (str, optional): . Defaults to "default".
        modifier_function_name (str, optional): Name of the modifier function to apply to the traces. Defaults to "".
        sigma (int, float, optional): number of standard deviations to display. Defaults to 1.
        q (float, optional): quantile ranges which are displayed. If None, no quantiles are shown. Defaults to None.
        save_fig (str, optional): Location of the figure if it shall be saved. Defaults to None.

    Returns:
        fig, axes: Matplotlib figure and axes objects if save_fig is None, otherwise True after saving the figure.
    """
    
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
        ax.tick_params(labelsize=12)
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
        plt.show()
        return True
    else:
        return fig, axes
    

def plot_cluster_samples(umap_classifier, data, clusters, cluster_idx,         
                 n_samples=10, # number of samples
                 alpha=0.05,
                 plot_types="default",
                 save_fig=None):
    """Plots random samples from a specified cluster.
    Args:
        umap_classifier (_umap_classifier_): The UMAP classifier object containing data and methods.
        data (np.ndarray): The data array containing all samples.
        clusters (np.ndarray): Array of cluster assignments for each sample.
        cluster_idx (int, str): The index of the cluster to plot samples from.
        n_samples (int, optional): Number of random samples to plot. Defaults to 10.
        alpha (float, optional): Transparency level for the sample plots. Defaults to 0.05.
        plot_types (str, optional): Types of plots to generate. Defaults to "default".
        save_fig (str, optional): Location to save the figure. If None, the figure is returned. Defaults to None.
    Returns:
        fig, axes: Matplotlib figure and axes objects if save_fig is None, otherwise True after saving the figure.
    """

    
    data_type = umap_classifier.data_preparation.get("target_data_type", "time")
    sampling_frequency = umap_classifier.data_preparation.get("sampling_frequency", 180)
    suppress_dc = umap_classifier.data_preparation.get("suppress_dc", False)
    n_fft, n_time = prepare_data.calculate_n_frequency_bins(data_type, data.shape[1], suppress_dc=suppress_dc)

    plot_types=default_plot_types(data_type,plot_types)

    # Get axes configurations
    axes_configs = create_axes_configurations()
    
    # Prepare figure based on number of plot types
    fig, axes = plt.subplots(1, len(plot_types), figsize=(6*len(plot_types), 4))
    
    # Ensure axes is always a list
    if len(plot_types) == 1:
        axes = [axes]
    
    mask = clusters == cluster_idx
    if sum(mask) < n_samples:
        n_samples = sum(mask)

    samples=np.random.choice(data[mask].shape[0], size=n_samples, replace=False)
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
        ax.tick_params(labelsize=12)
        for j in samples:
            ax.plot(x, data[mask][j][start_idx:end_idx],'k-',alpha=alpha)
        plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
        print(f"Saved figure to {save_fig}")
        plt.show()
    else:
        return fig, axes
    
def plot_index(umap_classifier,data,index, 
                 plot_types = 'default',
                 save_fig=None):
    """Plots the spectra of specified indices.
    Args:
        umap_classifier (_umap_classifier_): The UMAP classifier object containing data and methods.
        data (np.ndarray): The data array containing all samples.
        index (list of int): List of indices to plot.
        plot_types (str, optional): Types of plots to generate. Defaults to "default".
        save_fig (str, optional): Location to save the figure. If None, the figure is returned. Defaults to None.
    Returns:
        fig, axes: Matplotlib figure and axes objects if save_fig is None, otherwise True after saving the figure.
    """
    
    data_type = umap_classifier.data_preparation.get("target_data_type", "time")
    sampling_frequency = umap_classifier.data_preparation.get("sampling_frequency", 180)
    suppress_dc = umap_classifier.data_preparation.get("suppress_dc", False)
    n_fft, n_time = prepare_data.calculate_n_frequency_bins(data_type, data.shape[1], suppress_dc=suppress_dc)

    plot_types=default_plot_types(data_type,plot_types)
    alpha=1/len(index)**0.3
    # Get axes configurations
    axes_configs = create_axes_configurations()
    
    # Prepare figure based on number of plot types
    fig, axes = plt.subplots(1, len(plot_types), figsize=(6*len(plot_types), 5))
    
    # Ensure axes is always a list
    if len(plot_types) == 1:
        axes = [axes]
        
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
        ax.tick_params(labelsize=12)
        for j in index:
            ax.plot(x, data[j][start_idx:end_idx],alpha=alpha, label=j)
    plt.tight_layout()
    plt.legend()
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
        print(f"Saved figure to {save_fig}")
        plt.show()
        return fig, axes
    else:
        return fig, axes
    
def plot_embdding(embedding, labels=None, label_name='clusters', label_type='categorical', save_fig=None, alpha=0.01):
        """Plots a 2D embedding with optional coloring based on labels. 
        Args:
        embedding (np.ndarray): 2D array of shape (n_samples, 2) representing the embedding coordinates.
        labels (np.ndarray or None): Array of shape (n_samples,) containing labels for coloring. If None, no coloring is applied.
        label_name (str): Name of the label for the legend or colorbar.
        label_type (str): Type of labels, either 'categorical' or 'continuous' for continuous values.
        save_fig (str or None): If provided, the path to save the figure. If None, the figure is returned.
        alpha (float): Transparency level for the points in the scatter plot.
        Returns:
        fig, ax: Matplotlib figure and axes objects if save_fig is None, otherwise True after saving the figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        if not isinstance(labels, type(None)):
            if label_type == 'categorical':
                unique_labels = np.unique(labels)
                if len(unique_labels)>=100:
                    label_type = 'continuous'
                    warnings.warn("More than 100 unique labels detected. Using continuous color mapping.")
            if label_type == 'categorical':                    
                cmap = plt.cm.tab20
                colors = cmap(np.linspace(0, 1, len(unique_labels)))
                legend_handles = []
                for j,col in zip(unique_labels,colors):
                    if j == -1:
                        continue
                    marker='o'
                    mask = (labels == j)

                    xy = embedding[mask]
                    line, = plt.plot(
                        xy[:, 0],
                        xy[:, 1],
                        marker,
                        markerfacecolor=tuple(col),
                        markeredgecolor=tuple(col),
                        markersize=3,
                        alpha=alpha,
                        label=j
                    )
                    legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markeredgecolor=col, markersize=6))
                noise_mask = labels == -1
                xy = embedding[noise_mask]
                plt.plot(
                        xy[:, 0],
                        xy[:, 1],
                        "xk",
                        markersize=5,
                        label = "not classified"
                    )
                _, plot_labels = plt.gca().get_legend_handles_labels()
                if len(unique_labels)<22:
                    plt.legend(handles=legend_handles + [Line2D([0], [0], marker="x", color='w',markerfacecolor="k", markeredgecolor="k")], title=label_name, labels=plot_labels, loc='center left', bbox_to_anchor=(1, 0.5), markerscale=2, frameon=False)

                del legend_handles,plot_labels
            elif label_type == 'continuous':
                cmap = plt.cm.viridis
                norm = plt.Normalize(vmin=np.min(labels), vmax=np.max(labels))
                colors = cmap(norm(labels))
                plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=3, alpha=alpha)
                plt.colorbar(label=label_name)
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], s=3, alpha=alpha)
        plt.tick_params(labelsize=12)
        plt.xlabel("latent dimension 1", fontsize=14)
        plt.ylabel("latent dimension 2", fontsize=14)
        if save_fig:
            plt.savefig(save_fig, dpi =300, bbox_inches='tight')
            print(f"Saved figure to {save_fig}")
            plt.show()
        else:
            return fig, ax
    
def plot_overview_bokeh(embedding, color_key=None, title="Bokeh PLot of Embedded Data", save_fig=None, **hover_kwargs):
    """Creates an interactive Bokeh plot of the 2D embedding with optional coloring based on additional properties.
    Args:
        embedding (np.ndarray): 2D array of shape (n_samples, 2) representing the embedding coordinates.
        color_key (str or None): Attribute name in hover_kwargs to use for coloring the points. If None, no coloring is applied.
        title (str): Title of the plot.
        save_fig (str or None): If provided, the path to save the plot as an HTML file. If None, the plot is shown in a notebook.
        **hover_kwargs: Additional properties to include in the hover tooltips. Each property should be a list or numpy array of length n_samples.
    Returns:
        None: Displays the plot in a notebook or saves it as an HTML file.
    """
    output_notebook()

    plot_df = pd.DataFrame(embedding, columns=('x', 'y'))
    for property, value in hover_kwargs.items():
        if not isinstance(value, (list, np.ndarray)):
            raise ValueError(f"Property '{property}' must be a list or numpy array.")
        if len(value) != plot_df.shape[0]:
            raise ValueError(f"Property '{property}' must have the same number of elements as the embedding data ({plot_df.shape[0]}).")
        plot_df[property] = value
        
    if color_key:    
        if color_key not in plot_df.columns:
            raise ValueError(f"Color key '{color_key}' not found in the DataFrame columns. Available columns: {plot_df.columns.tolist()}")
        if np.unique(plot_df[color_key]).size >= 100:
            # Create a color mapper from blue (low values) to red (high values)
            color_mapping = LinearColorMapper(palette=Viridis256, 
                                            low=np.quantile(plot_df[color_key],0.95), 
                                            high=np.quantile(plot_df[color_key],0.05))
            color = dict(field=color_key, transform=color_mapping)
        else:
            plot_df[color_key] = plot_df[color_key].astype(str)
            cols=np.unique(plot_df[color_key])
            indices = np.linspace(0, 255, len(cols), dtype=int)
            color_mapping = CategoricalColorMapper(factors=cols, palette=[Viridis256[i] for i in indices])
            color = dict(field=color_key, transform=color_mapping)
    datasource = ColumnDataSource(plot_df)
    plot_figure = figure(
        title=title,
        width=600,
        height=600,
        tools=('pan, wheel_zoom, reset'),
        x_axis_label='latent dimension 1',
        y_axis_label='latent dimension 1'
    )
    plot_figure.axis.axis_label_text_font_size = "14pt"
    tooltip_html = "<div>"
    for col in plot_df.columns:
        if col not in ['x', 'y']:
            tooltip_html += f"""
                <div>
                    <span style='font-size: 14px; color: #224499'>{col}:</span>
                    <span style='font-size: 14px'>@{col}</span>
                </div>"""
    tooltip_html += "</div>"
    plot_figure.add_tools(HoverTool(tooltips=tooltip_html, mode='mouse', point_policy='follow_mouse'))
    if color_key:
        plot_figure.scatter('x','y',
                    source=datasource,
                    line_alpha=0.6,
                    #color=factor_cmap('x', palette=palette, factors=str_categories)
                    color=color,
                    fill_alpha=0.6,
                    size=10
                )
    else:
        plot_figure.scatter('x', 'y',
                    source=datasource,
                    line_alpha=0.6,
                    fill_alpha=0.6,
                    size=10
                )
    
    if save_fig==None:
        show(plot_figure)
    else:
        output_file(save_fig)
        save(plot_figure)
        #show(plot_figure)
        print(f"Saved bokeh plot at {save_fig}.")
