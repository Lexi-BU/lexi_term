from tabulate import tabulate
import global_variables
import numpy as np


def print_time_details(file_type=None, start_time=None, end_time=None):
    """
    Prints the details of the time values in different files in the data base for both SCI and
    HK files in a nice tabular format as well as the name of the loaded files.

    Parameters
    ----------
    file_type : str or list
        Type of file. Default is None.
    start_time : float
        Start time. Default is None.
    end_time : float
        End time. Default is None.

    Returns
    -------
        None
    """
    if file_type is None:
        file_type_list = ['sci', 'hk']
    else:
        if hasattr(file_type, '__len__'):
            file_type_list = file_type
            file_type = []
        else:
            file_type_list = [file_type]
            file_type = []

    for file_type in file_type_list:
        try:
            df_all = global_variables.all_file_details[f"df_all_{file_type}"]
            # df_slice = global_variables.all_file_details[f"df_all_{file_type}"]
            file_name = global_variables.all_file_details[f"file_name_{file_type}"]
            print(f"\n Displaying values for \x1b[1;32;255m {file_name.split('/')[-1]} \x1b[0m")
            print(tabulate(
                [[f"Minimum time in the {file_type} file", df_all.index.min()],
                 [f"Maximum time in the  {file_type} file", df_all.index.max()],
                 # [f"Minimum time in the sliced  {file_type} file", df_slice.index.min()],
                 # [f"Maximum time in the sliced  {file_type} file", df_slice.index.max()],
                 ["Start time from widget", start_time],
                 ["End time from widget", end_time]],
                headers=["Parameter", "Value"], tablefmt="fancy_grid", floatfmt=".2f",
                numalign="center"))
        except Exception:
            pass


def insert_file_name(file_load_entry=None, tk=None, file_name=None):
    """
    If a new file is loaded, then insert the file name into the entry box

    Parameters
    ----------
    file_load_entry : tkinter.Entry
        The entry box where the file name is inserted
    tk : tkinter.Tk
        The main window
    file_name : str
        The file name to be inserted

    Returns
    -------
        None
    """
    file_name_short = file_name.split('/')[-1]
    file_load_entry.delete(0, tk.END)
    file_load_entry.insert(0, file_name_short)


def curve_fit_func(x, a, b, c):
    """
    Get the Gaussian curve function

    Parameters
    ----------
    x : numpy.ndarray
        The x values
    a : float
        The amplitude of the curve
    b : float
        The mean of the curve
    c : float
        The standard deviation of the curve
    """
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


def fwhm(x, y):
    """
    Calculate the full width at half maximum of a curve

    Parameters
    ----------
    x : numpy.ndarray
        The x values
    y : numpy.ndarray
        The y values
    """
    half_max = np.max(y) / 2.
    left_idx = np.where(y > half_max)[0][0]
    right_idx = np.where(y > half_max)[0][-1]
    return x[right_idx] - x[left_idx]


def ts_option_update():
    print(global_variables.all_file_details["df_slice_hk"].columns.tolist())
    return global_variables.all_file_details["df_slice_hk"].columns.tolist()


def diff_file_warnings():
    """
    Warn the user if the hk and sci files do not have the same time values
    """
    try:
        hk_t_min = global_variables.all_file_details["df_all_hk"].index.min()
        hk_t_max = global_variables.all_file_details["df_all_hk"].index.max()
        sci_t_min = global_variables.all_file_details["df_all_sci"].index.min()
        sci_t_max = global_variables.all_file_details["df_all_sci"].index.max()
        if hk_t_min != sci_t_min:
            print("\n \x1b[1;31;255m WARNING: The hk file does not have the same minimum time "
                  "values as the sci file \x1b[0m")
            print(f"The hk file has the minimum time value of \x1b[1;32;255m{hk_t_min} \x1b[0m"
                  f"and the sci file has the minimum time value of \x1b[1;32;255m{sci_t_min}"
                  f"\x1b[0m")
        if hk_t_max != sci_t_max:
            print("\n \x1b[1;31;255m WARNING: The hk file does not have the same maximum time "
                  "values as the sci file \x1b[0m")
            print(f"The hk file has the maximum time value of \x1b[1;32;255m{hk_t_max} \x1b[0m"
                  f"and the sci file has the maximum time value of \x1b[1;32;255m{sci_t_max}"
                  f"\x1b[0m")
    except Exception:
        pass


def file_name_update(file_name=None, file_type=None):
    """
    Update the file name in the global variables

    Parameters
    ----------
    file_name : str
        The file name to be updated
    """
    if file_type == "sci":
        file_name = global_variables.all_file_details["file_name_sci"].split('/')[-1]
    elif file_type == "hk":
        file_name = global_variables.all_file_details["file_name_hk"].split('/')[-1]
    elif file_type == "b":
        file_name = global_variables.all_file_details["file_name_b"].split('/')[-1]
    diff_file_warnings()

    return file_name


def PinPullerTemp_func(vpc, hk_value, lxi_unit):
    PinPullerTemp = (hk_value * vpc - 2.73) * 100
    return PinPullerTemp


def OpticsTemp_func(vpc, hk_value, lxi_unit):
    OpticsTemp = (hk_value * vpc - 2.73) * 100
    return OpticsTemp


def LEXIbaseTemp_func(vpc, hk_value, lxi_unit):
    LEXIbaseTemp = (hk_value * vpc - 2.73) * 100
    return LEXIbaseTemp


def HVsupplyTemp_func(vpc, hk_value, lxi_unit):
    HVsupplyTemp = (hk_value * vpc - 2.73) * 100
    return HVsupplyTemp


def V_Imon_5_2_func(vpc, hk_value, lxi_unit):
    if lxi_unit == 1:
        V_Imon_5_2 = (hk_value * vpc) * 1e3 / 18
    elif lxi_unit == 2:
        V_Imon_5_2 = (hk_value * vpc - 1.129) * 1e3 / 21.456
    return V_Imon_5_2


def V_Imon_10_func(vpc, hk_value, lxi_unit):
    V_Imon_10 = hk_value * vpc
    return V_Imon_10


def V_Imon_3_3_func(vpc, hk_value, lxi_unit):
    if lxi_unit == 1:
        V_Imon_3_3 = (hk_value * vpc + 0.0178) * 1e3 / 9.131
    elif lxi_unit == 2:
        V_Imon_3_3 = (hk_value * vpc - 0.029) * 1e3 / 18
    return V_Imon_3_3


def AnodeVoltMon_func(vpc, hk_value, lxi_unit):
    AnodeVoltMon = hk_value * vpc
    return AnodeVoltMon


def V_Imon_28_func(vpc, hk_value, lxi_unit):
    if lxi_unit == 1:
        V_Imon_28 = (hk_value * vpc + 0.00747) * 1e3 / 17.94
    elif lxi_unit == 2:
        V_Imon_28 = (hk_value * vpc + 0.00747) * 1e3 / 17.94
    return V_Imon_28


def ADC_Ground_func(vpc, hk_value, ADC_Ground):
    ADC_Ground = hk_value * vpc
    return ADC_Ground


def Cmd_count_func(vpc, hk_value, lxi_unit):
    Cmd_count = hk_value * vpc
    return Cmd_count


def Pinpuller_Armed_func(vpc, hk_value, lxi_unit):
    Pinpuller_Armed = hk_value
    return Pinpuller_Armed


def Unused1_func(vpc, hk_value, lxi_unit):
    Unused1 = hk_value
    return Unused1


def Unused2_func(vpc, hk_value, lxi_unit):
    Unused2 = hk_value
    return Unused2


def HVmcpAuto_func(vpc, hk_value, lxi_unit):
    HVmcpAuto = hk_value * vpc
    return HVmcpAuto


def HVmcpMan_func(vpc, hk_value, lxi_unit):
    HVmcpMan = hk_value * vpc
    return HVmcpMan


def hk_value_comp(ii=None, vpc=None, hk_value=None, hk_id=None, lxi_unit=None):
    ops = {
        "0": PinPullerTemp_func,
        "1": OpticsTemp_func,
        "2": LEXIbaseTemp_func,
        "3": HVsupplyTemp_func,
        "4": V_Imon_5_2_func,
        "5": V_Imon_10_func,
        "6": V_Imon_3_3_func,
        "7": AnodeVoltMon_func,
        "8": V_Imon_28_func,
        "9": ADC_Ground_func,
        "10": Cmd_count_func,
        "11": Pinpuller_Armed_func,
        "12": Unused1_func,
        "13": Unused2_func,
        "14": HVmcpAuto_func,
        "15": HVmcpMan_func,
    }
    chosen_func = ops.get(str(hk_id))
    return chosen_func(vpc, hk_value, lxi_unit)
