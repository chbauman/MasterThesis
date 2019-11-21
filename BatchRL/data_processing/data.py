import os
from typing import Dict, Optional, Sequence, List

import numpy as np

from data_processing.dataset import SeriesConstraint, no_inds, Dataset, ModelDataView
from data_processing.preprocess import clean_data, remove_out_interval, clip_to_interval, interpolate_time_series, \
    fill_holes_linear_interpolate, remove_outliers, gaussian_filter_ignoring_nans, standardize
from rest.client import DataStruct
from util.numerics import align_ts, copy_arr_list, solve_ls, \
    find_rows_with_nans, extract_streak, nan_array_equal
from util.util import clean_desc, b_cast, create_dir, add_dt_and_t_init
from util.visualize import plot_time_series, plot_all, plot_single, preprocess_plot_path, \
    plot_multiple_time_series, plot_dataset, stack_compare_plot

#######################################################################################################
# NEST Data

# UMAR Room Data
Room272Data = DataStruct(id_list=[42150280,
                                  42150288,
                                  42150289,
                                  42150290,
                                  42150291,
                                  42150292,
                                  42150293,
                                  42150294,
                                  42150295,
                                  42150483,
                                  42150484,
                                  42150284,
                                  42150270],
                         name="UMAR_Room272",
                         start_date='2017-01-01',
                         end_date='2019-12-31')

Room274Data = DataStruct(id_list=[42150281,
                                  42150312,
                                  42150313,
                                  42150314,
                                  42150315,
                                  42150316,
                                  42150317,
                                  42150318,
                                  42150319,
                                  42150491,
                                  42150492,
                                  42150287,
                                  42150274],
                         name="UMAR_Room274",
                         start_date='2017-01-01',
                         end_date='2019-12-31')

# DFAB Data
Room4BlueData = DataStruct(id_list=[421110054,  # Temp
                                    421110023,  # Valves
                                    421110024,
                                    421110029,
                                    421110209  # Blinds
                                    ],
                           name="DFAB_Room41",
                           start_date='2017-01-01',
                           end_date='2019-12-31')

Room5BlueData = DataStruct(id_list=[421110072,  # Temp
                                    421110038,  # Valves
                                    421110043,
                                    421110044,
                                    421110219  # Blinds
                                    ],
                           name="DFAB_Room51",
                           start_date='2017-01-01',
                           end_date='2019-12-31')

Room4RedData = DataStruct(id_list=[421110066,  # Temp
                                   421110026,  # Valves
                                   421110027,
                                   421110028,
                                   ],
                          name="DFAB_Room43",
                          start_date='2017-01-01',
                          end_date='2019-12-31')

Room5RedData = DataStruct(id_list=[421110084,  # Temp
                                   421110039,  # Valves
                                   421110040,
                                   421110041,
                                   ],
                          name="DFAB_Room53",
                          start_date='2017-01-01',
                          end_date='2019-12-31')

DFAB_AddData = DataStruct(id_list=[421100168,  # Inflow Temp
                                   421100170,  # Outflow Temp
                                   421100174,  # Tot volume flow
                                   421100163,  # Pump running
                                   421110169,  # Speed of other pump
                                   421100070,  # Volume flow through other part
                                   ],
                          name="DFAB_Extra",
                          start_date='2017-01-01',
                          end_date='2019-12-31')

DFAB_AllValves = DataStruct(id_list=[421110008,  # First Floor
                                     421110009,
                                     421110010,
                                     421110011,
                                     421110012,
                                     421110013,
                                     421110014,
                                     421110023,  # Second Floor,
                                     421110024,
                                     421110025,
                                     421110026,
                                     421110027,
                                     421110028,
                                     421110029,
                                     421110038,  # Third Floor
                                     421110039,
                                     421110040,
                                     421110041,
                                     421110042,
                                     421110043,
                                     421110044,
                                     ],
                            name="DFAB_Valves",
                            start_date='2017-01-01',
                            end_date='2019-12-31')

rooms = [Room4BlueData, Room5BlueData, Room4RedData, Room5RedData]

# Weather Data
WeatherData = DataStruct(id_list=[3200000,
                                  3200002,
                                  3200008],
                         name="Weather",
                         start_date='2019-01-01',
                         end_date='2019-12-31')

# Battery Data
BatteryData = DataStruct(id_list=[40200000,
                                  40200001,
                                  40200002,
                                  40200003,
                                  40200004,
                                  40200005,
                                  40200006,
                                  40200007,
                                  40200008,
                                  40200009,
                                  40200010,
                                  40200011,
                                  40200012,
                                  40200013,
                                  40200014,
                                  40200015,
                                  40200016,
                                  40200017,
                                  40200018,
                                  40200019,
                                  40200087,
                                  40200088,
                                  40200089,
                                  40200090,
                                  40200098,
                                  40200099,
                                  40200102,
                                  40200103,
                                  40200104,
                                  40200105,
                                  40200106,
                                  40200107,
                                  40200108],
                         name="Battery",
                         start_date='2018-01-01',
                         end_date='2019-12-31')


#######################################################################################################
# Time Series Processing

def extract_date_interval(dates, values, d1, d2):
    """
    Extracts all data that lies within the dates d1 and d2.
    Returns None if there are no such points.
    """
    if d1 >= d2:
        raise ValueError("Invalid date interval passed.")
    mask = np.logical_and(dates < d2, dates > d1)
    if np.sum(mask) == 0:
        print("No values in provided date interval.")
        return None
    dates_new = dates[mask]
    values_new = values[mask]
    dt_init_new = dates_new[0]
    return dates_new, values_new, dt_init_new


def analyze_data(dat: Sequence) -> None:
    """
    Analyzes the provided raw data series and prints
    some information to the console.

    Args:
        dat: The raw data series to analyze.

    Returns:
        None
    """
    values, dates = dat
    n_data_p = len(values)
    print("Total: {} data points.".format(n_data_p))
    print("Data ranges from {} to {}".format(dates[0], dates[-1]))

    t_diffs = dates[1:] - dates[:-1]
    max_t_diff = np.max(t_diffs)
    mean_t_diff = np.mean(t_diffs)
    print("Largest gap:", np.timedelta64(max_t_diff, 'D'), "or", np.timedelta64(max_t_diff, 'h'))
    print("Mean gap:", np.timedelta64(mean_t_diff, 'm'), "or", np.timedelta64(mean_t_diff, 's'))

    print("Positive differences:", np.all(t_diffs > np.timedelta64(0, 'ns')))


def add_col(full_dat_array, data, dt_init, dt_init_new, col_ind, dt_mins=15):
    """
    Add time series as column to data array at the right index.
    If the second time series exceeds the datetime range of the
    first one it is cut to fit the first one. If it is too short
    the missing values are filled with NaNs.
    """

    n_data = full_dat_array.shape[0]
    n_data_new = data.shape[0]

    # Compute indices
    interval = np.timedelta64(dt_mins, 'm')
    offset_before = int(np.round((dt_init_new - dt_init) / interval))
    offset_after = n_data_new - n_data + offset_before
    dat_inds = [np.maximum(0, offset_before), n_data + np.minimum(0, offset_after)]
    new_inds = [np.maximum(0, -offset_before), n_data_new + np.minimum(0, -offset_after)]

    # Add to column
    full_dat_array[dat_inds[0]:dat_inds[1], col_ind] = data[new_inds[0]:new_inds[1]]
    return


def add_time(all_data, dt_init1, col_ind=0, dt_mins=15):
    """
    Adds the time as indices to the data,
    periodic with period one day.
    """

    n_data = all_data.shape[0]
    interval = np.timedelta64(dt_mins, 'm')
    n_ts_per_day = 24 * 60 / dt_mins
    t_temp_round = np.datetime64(dt_init1, 'D')
    start_t = (dt_init1 - t_temp_round) / interval
    for k in range(n_data):
        all_data[k, col_ind] = (start_t + k) % n_ts_per_day
    return


def pipeline_preps(orig_dat,
                   dt_mins,
                   all_data=None,
                   *,
                   dt_init=None,
                   row_ind=None,
                   clean_args=None,
                   clip_to_int_args=None,
                   remove_out_int_args=None,
                   rem_out_args=None,
                   hole_fill_args=None,
                   n_tot_cols=None,
                   gauss_sigma=None,
                   lin_ip=False):
    """
    Applies all the specified pre-processing to the
    given data.
    """
    modified_data = orig_dat

    # Clean Data
    if clean_args is not None:
        for k in clean_args:
            modified_data = clean_data(orig_dat, *k)

            # Clip to interval
    if remove_out_int_args is not None:
        remove_out_interval(modified_data, remove_out_int_args)

    # Clip to interval
    if clip_to_int_args is not None:
        clip_to_interval(modified_data, clip_to_int_args)

    # Interpolate / Subsample
    [modified_data, dt_init_new] = interpolate_time_series(modified_data, dt_mins, lin_ip=lin_ip)

    # Remove Outliers
    if rem_out_args is not None:
        remove_outliers(modified_data, *rem_out_args)

    # Fill holes
    if hole_fill_args is not None:
        fill_holes_linear_interpolate(modified_data, hole_fill_args)

    # Gaussian Filtering
    if gauss_sigma is not None:
        modified_data = gaussian_filter_ignoring_nans(modified_data, gauss_sigma)

    if all_data is not None:
        if dt_init is None or row_ind is None:
            raise ValueError("Need to provide the initial time of the first series and the column index!")

        # Add to rest of data
        add_col(all_data, modified_data, dt_init, dt_init_new, row_ind, dt_mins)
    else:
        if n_tot_cols is None:
            print("Need to know the total number of columns!")
            raise ValueError("Need to know the total number of columns!")

        # Initialize np array for compact storage
        n_data = modified_data.shape[0]
        all_data = np.empty((n_data, n_tot_cols), dtype=np.float32)
        all_data.fill(np.nan)
        all_data[:, 0] = modified_data

    return all_data, dt_init_new


def add_and_save_plot_series(data, m, curr_all_dat, ind: int, dt_mins,
                             dt_init, plot_name: str, base_plot_dir: str,
                             title: str = "",
                             pipeline_kwargs: Dict = None,
                             n_cols=None,
                             col_ind=None):
    """Adds the series with index `ind` to `curr_all_dat`
    and plots the series before and after processing
    with the pipeline.

    Args:
        data: The raw data.
        m: Metadata dictionary.
        curr_all_dat: The data array with the current processed data.
        dt_mins: Number of minutes in a timestep.
        dt_init: The initial time.
        plot_name: Name of the plot.
        base_plot_dir: Plotting directory.
        title: Title for plot.
        pipeline_kwargs: Arguments for `pipeline_preps`.
        ind: Index of series in raw data.
        n_cols: Total number of columns in final data.
        col_ind: Column index of series in processed data.
    """
    # Use defaults args if not specified.
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    dt_init_new = np.copy(dt_init)
    if col_ind is None:
        col_ind = ind
    elif curr_all_dat is None and col_ind != 0:
        raise NotImplementedError("col_ind cannot be chosen if curr_all_dat is None!")

    # Process data
    if curr_all_dat is None:
        col_ind = 0
        if n_cols is None:
            raise ValueError("Need to specify n_cols if data is None.")
        all_dat, dt_init_new = pipeline_preps(copy_arr_list(data[ind]),
                                              dt_mins,
                                              n_tot_cols=n_cols,
                                              **pipeline_kwargs)
        add_dt_and_t_init(m, dt_mins, dt_init_new)
    else:
        if dt_init is None:
            raise ValueError("dt_init cannot be None!")
        all_dat, _ = pipeline_preps(copy_arr_list(data[ind]),
                                    dt_mins,
                                    all_data=curr_all_dat,
                                    dt_init=dt_init,
                                    row_ind=col_ind,
                                    **pipeline_kwargs)

    if np.isnan(np.nanmax(all_dat[:, col_ind])):
        raise ValueError("Something went very fucking wrong!!")

    # Plot before data
    plot_file_name = base_plot_dir + "_" + plot_name + "_Raw"
    plot_time_series(data[ind][1], data[ind][0], m=m[ind], show=False, save_name=plot_file_name)

    # Plot after data
    plot_file_name2 = base_plot_dir + "_" + plot_name
    plot_single(np.copy(all_dat[:, col_ind]),
                m[ind],
                use_time=True,
                show=False,
                title_and_ylab=[title, m[ind]['unit']],
                save_name=plot_file_name2)

    return all_dat, dt_init_new


def save_ds_from_raw(all_data: np.ndarray, m_out: List[Dict], name: str,
                     c_inds: np.ndarray = None,
                     p_inds: np.ndarray = None,
                     standardize_data: bool = False,
                     custom_descs=None):
    """Creates a dataset from the raw input data.

    Args:
        all_data: 2D numpy data array
        m_out: List with metadata dictionaries.
        name: Name of the dataset.
        c_inds: Control indices.
        p_inds: Prediction indices.
        standardize_data: Whether to standardize the data.
        custom_descs: Optional custom description.

    Returns:
        Dataset
    """
    if c_inds is None:
        c_inds = no_inds
    if p_inds is None:
        p_inds = no_inds
    if standardize_data:
        all_data, m_out = standardize(all_data, m_out)
    dataset = Dataset.fromRaw(all_data, m_out, name, c_inds=c_inds, p_inds=p_inds)
    if custom_descs is not None:
        dataset.descriptions = custom_descs
    dataset.save()
    return dataset


def get_from_data_struct(dat_struct: DataStruct,
                         base_plot_dir: str,
                         dt_mins: int,
                         new_name: Optional[str],
                         ind_list: List[int],
                         prep_arg_list: List[Dict],
                         desc_list: np.ndarray = None,
                         c_inds: np.ndarray = None,
                         p_inds: np.ndarray = None,
                         standardize_data: bool = False) -> 'Dataset':
    """Extracts the specified series and applies
    pre-processing steps to all of them and puts them into a
    Dataset.
    """

    # Get name
    name = dat_struct.name
    if new_name is None:
        new_name = name

    # Try loading data
    try:
        loaded = Dataset.loadDataset(new_name)
        if desc_list is not None:
            loaded.descriptions = desc_list
        return loaded
    except FileNotFoundError:
        print("Did not find Dataset:", new_name)
        data, m = dat_struct.get_data()
        n_cols = len(data)

        # Check arguments
        n_inds = len(ind_list)
        n_preps = len(prep_arg_list)
        if n_inds > n_cols or n_preps > n_inds:
            raise ValueError("Too long lists!")
        if desc_list is not None:
            if len(desc_list) != n_inds:
                raise ValueError("List of description does not have correct length!!")

        all_data = None
        dt_init = None
        m_out = []

        # Sets
        for ct, i in enumerate(ind_list):
            n_cs = n_inds if ct == 0 else None
            title = clean_desc(m[i]['description'])
            title = title if desc_list is None else desc_list[ct]
            added_cols = m[i]['additionalColumns']
            plot_name = ""
            plot_file_name = os.path.join(base_plot_dir, added_cols['AKS Code'])
            all_data, dt_init = add_and_save_plot_series(data, m, all_data, i, dt_mins, dt_init, plot_name,
                                                         plot_file_name, title,
                                                         n_cols=n_cs,
                                                         pipeline_kwargs=prep_arg_list[ct],
                                                         col_ind=ct)
            m_out += [m[i]]

        # Save
        return save_ds_from_raw(all_data, m_out, new_name, c_inds, p_inds, standardize_data,
                                custom_descs=desc_list)


def convert_data_struct(dat_struct: DataStruct, base_plot_dir: str, dt_mins: int, pl_kwargs,
                        c_inds: np.ndarray = None,
                        p_inds: np.ndarray = None,
                        standardize_data=False) -> 'Dataset':
    """Converts a DataStruct to a Dataset.
    Using the same pre-processing steps for each series
    in the DataStruct.

    :param dat_struct: DataStruct to convert to Dataset.
    :param base_plot_dir: Where to save the preprocessing plots.
    :param dt_mins: Number of minutes in a timestep.
    :param pl_kwargs: Preprocessing pipeline kwargs.
    :param c_inds: Control indices.
    :param p_inds: Prediction indices.
    :param standardize_data: Whether to standardize the series in the dataset.
    :return: Converted Dataset.
    """

    # Get name
    name = dat_struct.name

    # Try loading data
    try:
        loaded = Dataset.loadDataset(name)
        return loaded
    except FileNotFoundError:
        data, m = dat_struct.get_data()
        n_cols = len(data)
        pl_kwargs = b_cast(pl_kwargs, n_cols)
        all_data = None
        dt_init = None

        # Sets
        for i in range(n_cols):
            n_cs = n_cols if i == 0 else None
            title = clean_desc(m[i]['description'])
            added_cols = m[i]['additionalColumns']
            plot_name = added_cols['AKS Code']
            plot_file_name = os.path.join(base_plot_dir, name)
            all_data, dt_init = add_and_save_plot_series(data, m, all_data, i, dt_mins, dt_init, plot_name,
                                                         plot_file_name, title,
                                                         n_cols=n_cs,
                                                         pipeline_kwargs=pl_kwargs[i])

        # Save
        return save_ds_from_raw(all_data, m, name, c_inds, p_inds, standardize_data)


#######################################################################################################
# Full Data Retrieval and Pre-processing

def get_battery_data() -> 'Dataset':
    """Loads the battery dataset if existing.

    Else it is created from the raw data and a few plots are make.
    Then the dataset is returned.

    Returns:
        Battery dataset.
    """
    # Constants
    dt_mins = 15
    name = "Battery"
    bat_plot_path = os.path.join(preprocess_plot_path, name)
    create_dir(bat_plot_path)
    inds = [19, 17]
    n_feats = len(inds)

    # Define arguments
    p_kwargs_soc = {'clean_args': [([0.0], 24 * 60, [])],
                    'rem_out_args': (100, [0.0, 100.0]),
                    'lin_ip': True}
    p_kwargs_ap = {'clean_args': [([], 6 * 60, [])]}
    kws = [p_kwargs_soc, p_kwargs_ap]
    c_inds = np.array([1], dtype=np.int32)
    custom_descs = np.array(["State of Charge [%]", "Active Power [kW]"])

    # Get the data
    ds = get_from_data_struct(BatteryData, bat_plot_path, dt_mins, name, inds, kws,
                              c_inds=c_inds,
                              standardize_data=True,
                              desc_list=custom_descs)

    # Plot files
    plot_name_roi = os.path.join(bat_plot_path, "Strange")
    plot_name_after = os.path.join(bat_plot_path, "Processed")

    # Plot all data
    y_lab = '% / kW'
    plot_dataset(ds, False, ['Processed Battery Data', y_lab], plot_name_after)

    # Get data
    dat, m = BatteryData.get_data()
    x = [dat[i][1] for i in inds]
    y = [dat[i][0] for i in inds]
    m_used = [m[i] for i in inds]

    # Extract and plot ROI of data where it behaves strangely
    d1 = np.datetime64('2019-05-24T12:00')
    d2 = np.datetime64('2019-05-25T12:00')
    x_ext, y_ext = [[extract_date_interval(x[i], y[i], d1, d2)[k] for i in range(n_feats)] for k in range(2)]
    plot_multiple_time_series(x_ext, y_ext, m_used,
                              show=False,
                              title_and_ylab=["Strange Battery Behavior", y_lab],
                              save_name=plot_name_roi)
    return ds


def get_weather_data() -> 'Dataset':
    """Load and interpolate the weather data.
    """
    # Constants
    dt_mins = 15
    filter_sigma = 2.0
    fill_by_ip_max = 2
    name = "Weather"
    name += "" if filter_sigma is None else str(filter_sigma)
    inds = [0, 2]

    # Specify kwargs for pipeline
    p_kwargs_temp = {'clean_args': [([], 30, [])],
                     'hole_fill_args': fill_by_ip_max,
                     'gauss_sigma': filter_sigma}
    p_kwargs_irr = {'clean_args': [([], 60, [1300.0, 0.0]), ([], 60 * 20)],
                    'hole_fill_args': fill_by_ip_max,
                    'gauss_sigma': filter_sigma}
    kws = [p_kwargs_temp, p_kwargs_irr]

    # Plot files
    prep_plot_dir = os.path.join(preprocess_plot_path, name)
    create_dir(prep_plot_dir)

    # Get the data
    custom_descs = np.array(["Outside Temperature [°C]", "Irradiance [W/m^2]"])
    ds = get_from_data_struct(WeatherData, prep_plot_dir, dt_mins, name, inds, kws,
                              desc_list=custom_descs,
                              standardize_data=True)
    return ds


def get_UMAR_heating_data() -> List['Dataset']:
    """Load and interpolate all the necessary data.

    Returns:
        The dataset with the UMAR data.
    """

    dat_structs = [Room272Data, Room274Data]
    dt_mins = 15
    fill_by_ip_max = 2
    filter_sigma = 2.0

    name = "UMAR"
    umar_rooms_plot_path = os.path.join(preprocess_plot_path, name)
    create_dir(umar_rooms_plot_path)
    all_ds = []

    for dat_struct in dat_structs:
        p_kwargs_room_temp = {'clean_args': [([0.0], 10 * 60)],
                              'rem_out_args': (4.5, [10, 100]),
                              'hole_fill_args': fill_by_ip_max,
                              'gauss_sigma': filter_sigma}
        p_kwargs_win = {'hole_fill_args': fill_by_ip_max,
                        'gauss_sigma': filter_sigma}
        p_kwargs_valve = {'hole_fill_args': 3,
                          'gauss_sigma': filter_sigma}
        kws = [p_kwargs_room_temp, p_kwargs_win, p_kwargs_valve]
        inds = [1, 11, 12]
        all_ds += [get_from_data_struct(dat_struct, umar_rooms_plot_path, dt_mins, dat_struct.name, inds, kws)]

    return all_ds


def get_DFAB_heating_data() -> List['Dataset']:
    """Loads or creates all data from DFAB then returns
    a list of all datasets. 4 for the rooms, one for the heating water
    and one with all the valves.

    Returns:
        List of all required datasets.
    """
    data_list = []
    dt_mins = 15
    dfab_rooms_plot_path = os.path.join(preprocess_plot_path, "DFAB")
    create_dir(dfab_rooms_plot_path)

    # Single Rooms
    for e in rooms:
        data, m = e.get_data()
        n_cols = len(data)

        # Single Room Heating Data  
        temp_kwargs = {'clean_args': [([0.0], 24 * 60, [])], 'gauss_sigma': 5.0, 'rem_out_args': (1.5, None)}
        valve_kwargs = {'clean_args': [([], 30 * 24 * 60, [])]}
        blinds_kwargs = {'clip_to_int_args': [0.0, 100.0], 'clean_args': [([], 7 * 24 * 60, [])]}
        prep_kwargs = [temp_kwargs, valve_kwargs, valve_kwargs, valve_kwargs]
        if n_cols == 5:
            prep_kwargs += [blinds_kwargs]
        data_list += [convert_data_struct(e, dfab_rooms_plot_path, dt_mins, prep_kwargs)]

    # General Heating Data
    temp_kwargs = {'remove_out_int_args': [10, 50], 'gauss_sigma': 5.0}
    prep_kwargs = [temp_kwargs, temp_kwargs, {}, {}, {}, {}]
    data_list += [convert_data_struct(DFAB_AddData, dfab_rooms_plot_path, dt_mins, prep_kwargs)]

    # All Valves Together
    prep_kwargs = {'clean_args': [([], 30 * 24 * 60, [])]}
    data_list += [convert_data_struct(DFAB_AllValves, dfab_rooms_plot_path, dt_mins, prep_kwargs)]
    return data_list


def compute_DFAB_energy_usage(show_plots=True):
    """
    Computes the energy usage for every room at DFAB
    using the valves data and the inlet and outlet water
    temperature difference.
    """

    # Load data from Dataset
    w_name = "DFAB_Extra"
    w_dataset = Dataset.loadDataset(w_name)
    w_dat = w_dataset.get_unscaled_data()
    t_init_w = w_dataset.t_init
    dt = w_dataset.dt

    v_name = "DFAB_Valves"
    dfab_rooms_plot_path = os.path.join(preprocess_plot_path, "DFAB")
    v_dataset = Dataset.loadDataset(v_name)
    v_dat = v_dataset.get_unscaled_data()
    t_init_v = v_dataset.t_init

    # Align data
    aligned_data, t_init_new = align_ts(v_dat, w_dat, t_init_v, t_init_w, dt)
    aligned_len = aligned_data.shape[0]
    w_dat = aligned_data[:, 21:]
    v_dat = aligned_data[:, :21]

    # Find nans
    not_nans = np.logical_not(find_rows_with_nans(aligned_data))
    aligned_not_nan = np.copy(aligned_data[not_nans])
    n_not_nans = aligned_not_nan.shape[0]
    w_dat_not_nan = aligned_not_nan[:, 21:]
    v_dat_not_nan = aligned_not_nan[:, :21]
    thresh = 0.05
    usable = np.logical_and(w_dat_not_nan[:, 3] > 1 - thresh,
                            w_dat_not_nan[:, 4] < thresh)
    usable = np.logical_and(usable, w_dat_not_nan[:, 5] > 0.1)
    usable = np.logical_and(usable, w_dat_not_nan[:, 5] < 0.2)

    usable = np.logical_and(usable, w_dat_not_nan[:, 2] < 0.9)
    usable = np.logical_and(usable, w_dat_not_nan[:, 2] > 0.6)
    n_usable = np.sum(usable)
    first_n_del = n_usable // 3

    # Room info
    room_dict = {0: "31", 1: "41", 2: "42", 3: "43", 4: "51", 5: "52", 6: "53"}
    valve_room_allocation = np.array(["31", "31", "31", "31", "31", "31", "31",  # 3rd Floor
                                      "41", "41", "42", "43", "43", "43", "41",  # 4th Floor
                                      "51", "51", "53", "53", "53", "52", "51",  # 5th Floor
                                      ])
    n_rooms = len(room_dict)
    n_valves = len(valve_room_allocation)

    # Loop over rooms and compute flow per room
    a_mat = np.empty((n_not_nans, n_rooms), dtype=np.float32)
    for i, room_nr in room_dict.items():
        room_valves = v_dat_not_nan[:, valve_room_allocation == room_nr]
        a_mat[:, i] = np.mean(room_valves, axis=1)
    b = w_dat_not_nan[:, 2]
    x = solve_ls(a_mat[usable][first_n_del:], b[usable][first_n_del:], offset=True)
    print("Flow", x)

    # Loop over rooms and compute flow per room
    a_mat = np.empty((n_not_nans, n_valves), dtype=np.float32)
    for i in range(n_valves):
        a_mat[:, i] = v_dat_not_nan[:, i]
    b = w_dat_not_nan[:, 2]
    x, fitted = solve_ls(a_mat[usable][first_n_del:], b[usable][first_n_del:], offset=False, ret_fit=True)
    print("Flow per valve", x)

    x = solve_ls(a_mat[usable][first_n_del:], b[usable][first_n_del:], non_neg=True)
    print("Non Negative Flow per valve", x)
    x = solve_ls(a_mat[usable][first_n_del:], b[usable][first_n_del:], non_neg=True, offset=True)
    print("Non Negative Flow per valve with offset", x)

    # stack_compare_plot(A[usable][first_n_del:], [21 * b[usable][first_n_del:], 21 * fitted], title="Valve model")
    stack_compare_plot(a_mat[usable][first_n_del:], [21 * 0.0286 * np.ones(b[usable][first_n_del:].shape), 21 * fitted],
                       title="Valve model")
    # PO
    x[:] = 0.0286
    if show_plots:
        tot_room_valves_plot_path = os.path.join(dfab_rooms_plot_path, "DFAB_All_Valves")
        m_all = {'description': 'All valves summed',
                 'unit': 'TBD',
                 'dt': dt,
                 't_init': t_init_new}
        plot_single(np.sum(v_dat, axis=1),
                    m_all,
                    use_time=True,
                    show=False,
                    title_and_ylab=['Sum All Valves', '0/1'],
                    save_name=tot_room_valves_plot_path)

    flow_rates_f3 = np.array([134, 123, 129, 94, 145, 129, 81], dtype=np.float32)
    print(np.sum(flow_rates_f3), "Flow rates sum, 3. OG")
    del_temp = 13
    powers_f45 = np.array([137, 80, 130, 118, 131, 136, 207,
                           200, 192, 147, 209, 190, 258, 258], dtype=np.float32)
    c_p = 4.186
    d_w = 997
    h_to_s = 3600

    flow_rates_f45 = h_to_s / (c_p * d_w * del_temp) * powers_f45
    print(flow_rates_f45)

    tot_n_val_open = np.sum(v_dat, axis=1)
    d_temp = w_dat[:, 0] - w_dat[:, 1]

    # Prepare output
    out_dat = np.empty((aligned_len, n_rooms), dtype=np.float32)
    m_list = []
    m_room = {'description': 'Energy consumption room',
              'unit': 'TBD',
              'dt': dt,
              't_init': t_init_new}
    if show_plots:
        w_plot_path = os.path.join(dfab_rooms_plot_path, w_name + "_WaterTemps")
        dw_plot_path = os.path.join(dfab_rooms_plot_path, w_name + "_WaterTempDiff")
        plot_dataset(w_dataset, show=False,
                     title_and_ylab=["Water Temps", "Temperature"],
                     save_name=w_plot_path)
        plot_single(d_temp, m_room, use_time=True, show=False,
                    title_and_ylab=["Temperature Difference", "DT"],
                    scale_back=False,
                    save_name=dw_plot_path)

    # Loop over rooms and compute energy
    for i, room_nr in room_dict.items():
        room_valves = v_dat[:, valve_room_allocation == room_nr]
        room_sum_valves = np.sum(room_valves, axis=1)

        # Divide ignoring division by zero
        room_energy = d_temp * np.divide(room_sum_valves,
                                         tot_n_val_open,
                                         out=np.zeros_like(room_sum_valves),
                                         where=tot_n_val_open != 0)

        m_room['description'] = 'Room ' + room_nr
        if show_plots:
            tot_room_valves_plot_path = os.path.join(dfab_rooms_plot_path,
                                                     "DFAB_" + m_room['description'] + "_Tot_Valves")
            room_energy_plot_path = os.path.join(dfab_rooms_plot_path, "DFAB_" + m_room['description'] + "_Tot_Energy")
            plot_single(room_energy,
                        m_room,
                        use_time=True,
                        show=False,
                        title_and_ylab=['Energy Consumption', 'Energy'],
                        save_name=room_energy_plot_path)
            plot_single(room_sum_valves,
                        m_room,
                        use_time=True,
                        show=False,
                        title_and_ylab=['Sum All Valves', '0/1'],
                        save_name=tot_room_valves_plot_path)

        # Add data to output
        out_dat[:, i] = room_energy
        m_list += [m_room.copy()]

    # Save dataset
    ds_out = Dataset.fromRaw(out_dat,
                             m_list,
                             "DFAB_Room_Energy_Consumption")
    ds_out.save()


def analyze_room_energy_consumption():
    """
    Compares the energy consumption of the different rooms
    summed over whole days.
    """

    ds = Dataset.loadDataset("DFAB_Room_Energy_Consumption")
    relevant_rooms = np.array([False, True, False, True, True, False, True], dtype=np.bool)
    dat = ds.data[:, relevant_rooms]
    n = dat.shape[0]
    d = dat.shape[1]

    # Set rows with nans to nan
    row_with_nans = find_rows_with_nans(dat)
    dat[row_with_nans, :] = np.nan

    # Sum Energy consumption over days
    n_ts = 4 * 24  # 1 Day
    n_mins = ds.dt * n_ts
    offset = n % n_ts
    dat = dat[offset:, :]
    dat = np.sum(dat.reshape((-1, n_ts, d)), axis=1)

    # Plot
    m = [{'description': ds.descriptions[relevant_rooms][k], 'dt': n_mins} for k in range(d)]
    f_name = os.path.join(preprocess_plot_path, "DFAB")
    f_name = os.path.join(f_name, "energy_comparison")
    plot_all(dat, m, use_time=False, show=False, title_and_ylab=["Room Energy Consumption", "Energy over one day"],
             save_name=f_name)


#######################################################################################################
# Dataset generation


def generate_room_datasets() -> List[Dataset]:
    """Gather the right data and put it all together.

    Returns:
        List of room datasets of DFAB.
    """

    # Get weather
    w_dataset = get_weather_data()

    # Get room data
    dfab_dataset_list = get_DFAB_heating_data()
    n_rooms = len(rooms)
    dfab_room_dataset_list = [dfab_dataset_list[i] for i in range(n_rooms)]

    # Heating water temperature
    dfab_heat_water_temp_ds = dfab_dataset_list[n_rooms]
    heat_water_ds = dfab_heat_water_temp_ds[0:2]
    inlet_water_and_weather = w_dataset + heat_water_ds

    out_ds_list = []

    # Single room datasets
    for ct, room_ds in enumerate(dfab_room_dataset_list):

        # Get name
        room_nr_str = room_ds.name[-2:]
        new_name = "Model_Room" + room_nr_str
        print("Processing", new_name)

        # Try loading from disk
        try:
            curr_out_ds = Dataset.loadDataset(new_name)
            out_ds_list += [curr_out_ds]
            continue
        except FileNotFoundError:
            pass

        # Extract datasets
        valves_ds = room_ds[1:4]
        room_temp_ds = room_ds[0]

        # Compute average valve data and put into dataset
        valves_avg = np.mean(valves_ds.data, axis=1)
        valves_avg_ds = Dataset(valves_avg,
                                valves_ds.dt,
                                valves_ds.t_init,
                                np.empty((1, 2), dtype=np.float32),
                                np.array([False]),
                                np.array(["Averaged Valve Open Time [100%]"]))

        # Put all together
        full_ds = (inlet_water_and_weather + valves_avg_ds) + room_temp_ds
        full_ds.c_inds = np.array([4], dtype=np.int32)

        # Set descriptions
        full_ds.descriptions[-1] = "Room Temperature [°C]"
        full_ds.descriptions[2] = "Heating Water Temperature (In) [°C]"
        full_ds.descriptions[3] = "Heating Water Temperature (Out) [°C]"

        # Add blinds
        if len(room_ds) == 5:
            blinds_ds = room_ds[4]
            full_ds = full_ds + blinds_ds

        # Save
        full_ds.name = new_name
        full_ds.save()
        out_ds_list += [full_ds]

    # Return all
    return out_ds_list


def generate_sin_cos_time_ds(other: Dataset) -> Dataset:
    """
    Generates a time dataset from the last two
    time series of another dataset.

    :param other: The other dataset.
    :return: A new dataset containing only the time series.
    """
    # Try loading
    name = other.name + "_SinCosTime"
    try:
        return Dataset.loadDataset(name)
    except FileNotFoundError:
        pass

    # Construct Time dataset
    n_feat = other.d
    ds_sin_cos_time: Dataset = Dataset.copy(other[n_feat - 2: n_feat])
    ds_sin_cos_time.name = name
    ds_sin_cos_time.p_inds = np.array([0], dtype=np.int32)
    ds_sin_cos_time.c_inds = no_inds
    ds_sin_cos_time.save()
    return ds_sin_cos_time


def choose_dataset(base_ds_name: str = "Model_Room43",
                   seq_len: int = 20,
                   add_battery: bool = False) -> Dataset:
    """Let's you choose a dataset.

    Reads a room dataset, if it is not found, it is generated.
    Then the sequence length is set, the time variable is added and
    it is standardized and split into parts for training, validation
    and testing. Finally it is returned with the corresponding constraints.

    Args:
        base_ds_name: The name of the base dataset, must be of the form "Model_Room<nr>",
            with nr = 43 or 53.
        seq_len: The sequence length to use for the RNN training.
        add_battery: Whether to add the battery dataset.

    Returns:
        The prepared dataset.
    """
    # Check `base_ds_name`.
    if base_ds_name[:10] != "Model_Room" or base_ds_name[-2:] not in ["43", "53"]:
        raise ValueError(f"Dataset: {base_ds_name} does not exist!")

    # Load dataset, generate if not found.
    try:
        ds = Dataset.loadDataset(base_ds_name)
    except FileNotFoundError:
        get_DFAB_heating_data()
        generate_room_datasets()
        ds = Dataset.loadDataset(base_ds_name)

    # Set sequence length
    ds.seq_len = seq_len
    ds.name = base_ds_name[-6:] + "_" + str(ds.seq_len)

    # Add time variables and optionally the battery data
    ds = ds.add_time()
    if add_battery:
        bat_ds = get_battery_data()
        assert bat_ds.dt == ds.dt, "Incompatible timestep!"
        ds = ds + bat_ds

    # Standardize and prepare different parts of dataset.
    ds.standardize()
    ds.split_data()

    # Return
    return ds


def get_constraints(ds: Dataset = None, include_bat: bool = False) -> List[SeriesConstraint]:
    """Defines the constraints for a full dataset.

    Args:
        ds: Dataset
        include_bat: Whether the dataset includes the battery data.

    Returns:
        List of constraints
    """
    # Constraints for room dataset.
    rnn_consts = [
        SeriesConstraint('interval', [-15.0, 40.0]),
        SeriesConstraint('interval', [0.0, 1300.0]),
        SeriesConstraint('interval', [-10.0, 100.0]),
        SeriesConstraint('interval', [-10.0, 100.0]),
        SeriesConstraint('interval', [0.0, 1.0]),
        SeriesConstraint('interval', [0.0, 40.0]),
        SeriesConstraint('exact'),
        SeriesConstraint('exact'),
    ]
    # Constraints for battery dataset.
    if include_bat:
        bat_consts = [
            SeriesConstraint('interval', [20.0, 80.0]),
            SeriesConstraint('interval', [-100.0, 100.0]),
        ]
        rnn_consts += bat_consts

    # Transform
    if ds is not None:
        assert ds.d == len(rnn_consts), "Incorrect number of columns in dataset."
        ds.transform_c_list(rnn_consts)

    return rnn_consts


#######################################################################################################
# Testing

# Test DataStruct
TestData = DataStruct(id_list=[421100171, 421100172],
                      name="Test",
                      start_date='2019-08-08',
                      end_date='2019-08-09')


# Synthetic class imitating a DataStruct
class TestDataSynthetic:
    """
    Synthetic and short dataset to be used for debugging.
    Imitates a DataStruct.
    """

    @staticmethod
    def get_data():
        # First Time series
        dict1 = {'description': "Synthetic Data Series 1: Base Series", 'unit': "Test Unit 1"}
        val_1 = np.array([1.0, 2.3, 2.3, 1.2, 2.3, 0.8])
        dat_1 = np.array([
            np.datetime64('2005-02-25T03:31'),
            np.datetime64('2005-02-25T03:39'),
            np.datetime64('2005-02-25T03:48'),
            np.datetime64('2005-02-25T04:20'),
            np.datetime64('2005-02-25T04:25'),
            np.datetime64('2005-02-25T04:30'),
        ], dtype='datetime64')

        # Second Time series
        dict2 = {'description': "Synthetic Data Series 2: Delayed and longer", 'unit': "Test Unit 2"}
        val_2 = np.array([1.0, 1.4, 2.1, 1.5, 3.3, 1.8, 2.5])
        dat_2 = np.array([
            np.datetime64('2005-02-25T03:51'),
            np.datetime64('2005-02-25T03:59'),
            np.datetime64('2005-02-25T04:17'),
            np.datetime64('2005-02-25T04:21'),
            np.datetime64('2005-02-25T04:34'),
            np.datetime64('2005-02-25T04:55'),
            np.datetime64('2005-02-25T05:01'),
        ], dtype='datetime64')

        # Third Time series
        dict3 = {'description': "Synthetic Data Series 3: Repeating Values", 'unit': "Test Unit 3"}
        val_3 = np.array([0.0, 1.4, 1.4, 0.0, 3.3, 3.3, 3.3, 3.3, 0.0, 3.3, 2.5])
        dat_3 = np.array([
            np.datetime64('2005-02-25T03:45'),
            np.datetime64('2005-02-25T03:53'),
            np.datetime64('2005-02-25T03:59'),
            np.datetime64('2005-02-25T04:21'),
            np.datetime64('2005-02-25T04:23'),
            np.datetime64('2005-02-25T04:25'),
            np.datetime64('2005-02-25T04:34'),
            np.datetime64('2005-02-25T04:37'),
            np.datetime64('2005-02-25T04:45'),
            np.datetime64('2005-02-25T04:55'),
            np.datetime64('2005-02-25T05:01'),
        ], dtype='datetime64')
        return [(val_1, dat_1), (val_2, dat_2), (val_3, dat_3)], [dict1, dict2, dict3]


TestData2 = TestDataSynthetic()


# Tests
def test_rest_client() -> None:
    """
    Tests the REST client by requesting test data,
    saving it locally, reading it locally and deleting
    it again.

    Returns: None
    """

    # Load using REST api and locally
    t_dat = TestData
    t_dat.get_data()
    t_dat.get_data()

    # Remove data again
    fol = TestData.get_data_folder()
    for f in os.listdir(fol):
        file_path = os.path.join(fol, f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def get_data_test():
    name = "Test"

    # Test Sequence Cutting
    seq1 = np.array([1, 2, np.nan, 1, 2, np.nan, 3, 2, 1, 3, 4, np.nan, np.nan, 3, 2, 3, 1, 3, np.nan, 1, 2])
    seq2 = np.array([3, 4, np.nan, np.nan, 2, np.nan, 3, 2, 1, 3, 4, 7, np.nan, 3, 2, 3, 1, np.nan, np.nan, 3, 4])
    n = seq1.shape[0]
    all_dat = np.empty((n, 2), dtype=np.float32)
    all_dat[:, 0] = seq1
    all_dat[:, 1] = seq2
    print(all_dat)

    # Test
    streak = extract_streak(all_dat, 2, 2)
    print(streak)

    # Test Standardizing
    m = [{}, {}]
    all_dat, m = standardize(all_dat, m)
    print(all_dat)
    print(m)

    # Test hole filling by interpolation
    test_ts = np.array([np.nan, np.nan, 1, 2, 3.5, np.nan, 4.5, np.nan])
    test_ts2 = np.array([1, 2, 3.5, np.nan, np.nan, 5.0, 5.0, np.nan, 7.0])
    fill_holes_linear_interpolate(test_ts, 1)
    fill_holes_linear_interpolate(test_ts2, 2)
    print(test_ts)
    print(test_ts2)

    # Test Outlier Removal
    test_ts3 = np.array(
        [1, 2, 3.5, np.nan, np.nan, 5.0, 5.0, 17.0, 5.0, 2.0, -1.0, np.nan, 7.0, 7.0, 17.0, np.nan, 20.0, 5.0, 6.0])
    print(test_ts3)
    remove_outliers(test_ts3, 5.0, [0.0, 100.0])
    print(test_ts3)

    # Show plots
    dt_mins = 15
    dat, m = TestData2.get_data()

    # Clean data
    clean_data(dat[2], [0.0], 4, [3.3])

    # Interpolate
    [data1, dt_init1] = interpolate_time_series(dat[0], dt_mins)
    n_data = data1.shape[0]
    print(n_data, "total data points.")

    # Initialize np array for compact storage
    all_data = np.empty((n_data, 3), dtype=np.float32)
    all_data.fill(np.nan)

    # Add data
    add_time(all_data, dt_init1, 0, dt_mins)
    all_data[:, 1] = data1
    [data2, dt_init2] = interpolate_time_series(dat[1], dt_mins)

    print(data2)
    data2 = gaussian_filter_ignoring_nans(data2)
    print(data2)

    add_col(all_data, data2, dt_init1, dt_init2, 2, dt_mins)
    return all_data, m, name


def test_align() -> None:
    """
    Tests the alignment of two time series.
    """

    # Test data
    t_i1 = '2019-01-01 00:00:00'
    t_i2 = '2019-01-01 00:30:00'
    dt = 15
    ts_1 = np.array([1, 2, 2, 2, 3, 3], dtype=np.float32)
    ts_2 = np.array([2, 3, 3], dtype=np.float32)

    # Do tests
    test1 = align_ts(ts_1, ts_2, t_i1, t_i2, dt)
    print('Test 1:', test1)
    test2 = align_ts(ts_2, ts_1, t_i1, t_i2, dt)
    print('Test 2:', test2)
    test3 = align_ts(ts_1, ts_1, t_i1, t_i2, dt)
    print('Test 3:', test3)
    test4 = align_ts(ts_1, ts_1, t_i2, t_i1, dt)
    print('Test 4:', test4)
    test5 = align_ts(ts_2, ts_1, t_i2, t_i1, dt)
    print('Test 5:', test5)
    return


def get_test_ds(dat: np.ndarray, c_inds: np.ndarray,
                name: str = "SyntheticTest",
                dt: int = 60 * 12,
                t_init: str = '2019-01-01 12:00:00'):
    """Constructs a test dataset.

    Args:
        dat: The data array.
        c_inds: The control indices.
        name: The name of the dataset.
        dt: Timestep in minutes.
        t_init: Initial time.

    Returns:
        New dataset with dummy descriptions and unscaled data.
    """
    n_series = dat.shape[1]
    descs = np.array([f"Series {i} [unit{i}]" for i in range(n_series)])
    is_sc = np.array([False for _ in range(n_series)])
    sc = np.empty((n_series, 2), dtype=np.float32)
    ds = Dataset(dat, dt, t_init, sc, is_sc, descs, c_inds, name=name)
    return ds


def test_dataset_artificially() -> None:
    """Constructs a small synthetic dataset and makes tests.

    Tests the plotting and the index conversion of the dataset.

    Returns:
        None

    Raises:
        AssertionError: If a test is not passed.
    """

    dat = np.array([0, 2, 3, 7, 8,
                    1, 3, 4, 8, 9,
                    1, 4, 5, 7, 8,
                    2, 5, 6, 7, 9], dtype=np.float32).reshape((4, -1))
    c_inds = np.array([1, 3])
    ds = get_test_ds(dat, c_inds)
    ds.save()
    plot_dataset(ds, False, ["Test", "Fuck"])

    # Specify index tests
    test_list = [
        (np.array([2, 4], dtype=np.int32), np.array([1, 2], dtype=np.int32), ds.to_prepared),
        (np.array([2, 3], dtype=np.int32), np.array([1, 4], dtype=np.int32), ds.to_prepared),
        (np.array([0, 1, 2, 3, 4], dtype=np.int32), np.array([0, 3, 1, 4, 2], dtype=np.int32), ds.to_prepared),
        (np.array([0, 1, 2], dtype=np.int32), np.array([0, 2, 4], dtype=np.int32), ds.from_prepared),
        (np.array([2, 3, 4], dtype=np.int32), np.array([4, 1, 3], dtype=np.int32), ds.from_prepared),
    ]

    # Run index tests
    for t in test_list:
        inp, sol, fun = t
        out = fun(inp)
        if not np.array_equal(sol, out):
            print("Test failed :(")
            raise AssertionError("Function: {} with input: {} not giving: {} but: {}!!!".format(fun, inp, sol, out))

    # Test the constraints and the standardization
    c_list = [
        SeriesConstraint('interval', np.array([0.0, 1.0])),
        SeriesConstraint(),
        SeriesConstraint(),
        SeriesConstraint(),
        SeriesConstraint(),
    ]

    ds.standardize()
    if not np.allclose(ds.scaling[0][0], 1.0):
        raise AssertionError("Standardizing failed!")

    ds.transform_c_list(c_list)
    if not np.allclose(c_list[0].extra_dat[1], 0.0):
        raise AssertionError("Interval transformation failed!")

    # Test get_scaling_mul
    scaling, is_sc = ds.get_scaling_mul(0, 3)
    is_sc_exp = np.array([True, True, True])
    sc_mean_exp = np.array([1.0, 1.0, 1.0])
    if not np.array_equal(is_sc_exp, is_sc):
        raise AssertionError("get_scaling_mul failed!")
    if not np.allclose(sc_mean_exp, scaling[:, 0]):
        raise AssertionError("get_scaling_mul failed!")

    # Test ModelDataView
    dat_nan = np.array([0, 2, 3, 7, 8,
                        1, 3, 4, 8, np.nan,
                        2, 3, 4, 8, np.nan,
                        3, 3, 4, 8, np.nan,
                        4, 4, 5, 7, 8,
                        5, 4, 5, 7, 8,
                        6, 4, 5, 7, 8,
                        7, 4, 5, 7, 8,
                        8, 4, np.nan, 7, 8,
                        9, 4, np.nan, 7, 8,
                        10, 5, 6, 7, 9], dtype=np.float32).reshape((-1, 5))
    ds_nan = get_test_ds(dat_nan, c_inds, name="SyntheticTest")
    ds_nan.seq_len = 2

    # Test get_rel_data
    mdv = ModelDataView(ds_nan, "Test", 2, 7)
    mod_dat = mdv.get_rel_data()
    if not nan_array_equal(mod_dat, dat_nan[2:9]):
        raise AssertionError("Something's fucking wrong!!")

    # Test streak extraction
    mdv.extract_streak(3)
    str_dat, i = mdv.extract_streak(3)
    exp_dat = np.array([
        dat_nan[5:7],
        dat_nan[6:8],
    ])
    if not np.array_equal(str_dat, exp_dat) or not i == 3:
        raise AssertionError("Something in extract_streak is fucking wrong!!")

    # Test disjoint streak extraction
    dis_dat, dis_inds = mdv.extract_disjoint_streaks(2, 1)
    exp_dis = np.array([[
        dat_nan[4:6],
        dat_nan[5:7],
    ]])
    if not np.array_equal(dis_dat, exp_dis) or not np.array_equal(dis_inds, np.array([2])):
        raise AssertionError("Something in extract_disjoint_streaks is fucking wrong!!")

    # Test split_data
    ds_nan.val_percent = 0.33
    ds_nan.split_data()
    test_dat = ds_nan.split_dict['test'].get_rel_data()
    val_dat = ds_nan.split_dict['val'].get_rel_data()
    if not nan_array_equal(test_dat, dat_nan[7:]):
        raise AssertionError("Something in split_data is fucking wrong!!")
    if not nan_array_equal(val_dat, dat_nan[3:7]):
        raise AssertionError("Something in split_data is fucking wrong!!")

    # Test get_day
    day_dat, ns = ds_nan.get_days('val')
    exp_first_out_dat = np.array([
        [5, 5.0, 8.0],
        [6, 5.0, 8.0],
    ], dtype=np.float32)
    if ns[0] != 4 or not np.array_equal(day_dat[0][1], exp_first_out_dat):
        raise AssertionError("get_days not implemented correctly!!")

    print("Dataset test passed :)")
