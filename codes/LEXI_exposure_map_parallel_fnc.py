import numpy as np
import pandas as pd
from text_color_fnc import text_color as tc

tc = tc()


def vignette(d):
    """
    Function to calculate the vignetting factor for a given distance from boresight

    Parameters
    ----------
    d : float
        Distance from boresight in degrees

    Returns
    -------
    f : float
        Vignetting factor
    """

    # Set the vignetting factor
    # f = 1.0 - 0.5 * (d / (LEXI_FOV * 0.5)) ** 2
    f = 1

    return f


def calculate_exposure_delta(*args):
    """
    Function to calculate the exposure delta for a given pointing step

    Parameters
    ----------
    args : tuple
        Tuple of arguments to pass to the function
        The arguments are:
            i : int
                Index of the pointing step
            x_grid_arr : numpy array
                Array of x values for the exposure map
            y_grid_arr : numpy array
                Array of y values for the exposure map
            ra : float
                RA of the pointing step
            dec : float
                DEC of the pointing step
            LEXI_FOV : float
                LEXI FOV in degrees
            step : float
                Step size in seconds

    Returns
    -------
    exposure_delt : numpy array
        Exposure delta array
    """
    i = args[0][0]
    x_grid_arr = args[0][1]
    y_grid_arr = args[0][2]
    ra = args[0][3]
    dec = args[0][4]
    LEXI_FOV = args[0][5]
    step = args[0][6]

    # Calculate the distance from the pointing to each pixel
    r = np.sqrt((x_grid_arr - ra) ** 2 + (y_grid_arr - dec) ** 2)
    exposure_delt = np.where((r < LEXI_FOV * 0.5), vignette(r) * step, 0)
    return exposure_delt


# Function to interpolate the pointing data to a given resolution
def interpolate_pointing(
    df,
    t_res="100L",
    method="ffill",
    save_df=False,
    filename="../data/LEXI_pointing_ephem_highres",
    filetype="pkl",
):
    """
    Function to interpolate the pointing data to a given resolution

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with pointing data
    res : string
        Resolution to interpolate to.  Default is 100 ms
    method : string
        Interpolation method.  Default is forward fill
    save_df : bool
        If True, save the dataframe to a file.  Default is False
    filename : string
        Filename to save the dataframe to.  Default is '../data/LEXI_pointing_ephem_highres'
    filetype : string
        Filetype to save the dataframe to.  Default is 'pkl'. Options are 'csv' or 'pkl'

    Returns
    -------
    df : pandas dataframe
        Dataframe with interpolated pointing data

    Raises
    ------
    TypeError
        If df is not a pandas dataframe
    ValueError
        If df is empty
    TypeError
        If res is not a string
    TypeError
        If index is not a datetime object
    ValueError
        If method is not 'ffill' or 'bfill'
    TypeError
        If filename is not a string
    """

    # Check if df is a pandas dataframe and is not empty
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    if df.empty:
        raise ValueError("df cannot be empty")
    if not isinstance(t_res, str):
        raise TypeError(
            "res must be a string. Options are '100L' or '1S' or any other pandas"
            "resample string"
        )

    # Check if the index is a datetime object
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("index must be a datetime object")

    if method not in ["ffill", "bfill"]:
        raise ValueError("method must be 'ffill' or 'bfill'")

    # Interpolate the even time steps to get 100 ms resolution
    # Print that the interpolation is happening
    print(
        f"\nInterpolating the pointing data to {t_res} resolution. This may take a while\n"
    )
    df_intrp = df.resample(t_res).interpolate(method=method)

    if save_df:
        save_file = f"{filename}_tres_{t_res}.{filetype}"
        print(f"Trying to save dataframe to {tc.green_text(save_file)}\n")
        # Check if filename is given, if not then print saying that default will be used
        if filename is None:
            print(f"No filename given, default will be used: {filename}")
        # Check if filename is a string
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")

        # Save the dataframe to proper file type
        if filetype == "csv":
            df_intrp.to_csv(save_file, index=False)
        elif filetype == "pkl":
            df_intrp.to_pickle(save_file)
        else:
            raise ValueError("filetype must be 'csv' or 'pkl'")
        print(f"Dataframe saved to {tc.green_text(save_file)}\n")

    return df_intrp
