import csv
import datetime
import importlib
import struct
from pathlib import Path
from tkinter import filedialog
from typing import NamedTuple

import numpy as np
import pandas as pd

import global_variables
import lxi_misc_codes as lmsc

importlib.reload(lmsc)

# Tha packet format of the science and housekeeping packets
packet_format_sci = ">II4H"
# signed lower case, unsigned upper case (b)
packet_format_hk = ">II4H"

# double precision format for time stamp from pit
packet_format_pit = "<d"


sync_lxi = b'\xfe\x6b\x28\x40'
sync_pit = b'\x54\x53'

volts_per_count = 4.5126 / 65536  # volts per increment of digitization


class sci_packet(NamedTuple):
    """
    Class for the science packet.
    The code unpacks the science packet into a named tuple. Based on the packet format, each packet
    is unpacked into following parameters:
    - Date: time of the packet as received from the PIT
    - timestamp: int (32 bit)
    - IsCommanded: bool (1 bit)
    - voltage channel1: float (16 bit)
    - voltage channel2: float (16 bit)
    - voltage channel3: float (16 bit)
    - voltage channel4: float (16 bit)

    TimeStamp is the time stamp of the packet in seconds.
    IsCommand tells you if the packet was commanded.
    Voltages 1 to 4 are the voltages of corresponding different channels.
    """
    Date: float
    is_commanded: bool
    timestamp: int
    channel1: float
    channel2: float
    channel3: float
    channel4: float

    @classmethod
    def from_bytes(cls, bytes_: bytes):
        structure_time = struct.unpack(">d", bytes_[2:10])
        structure = struct.unpack(packet_format_sci, bytes_[12:])
        return cls(
            Date=structure_time[0],
            is_commanded=bool(structure[1] & 0x40000000),  # mask to test for commanded event type
            timestamp=structure[1] & 0x3fffffff,           # mask for getting all timestamp bits
            channel1=structure[2] * volts_per_count,
            channel2=structure[3] * volts_per_count,
            channel3=structure[4] * volts_per_count,
            channel4=structure[5] * volts_per_count,
        )


class sci_packet_gsfc(NamedTuple):
    """
    Class for the science packet.
    The code unpacks the science packet into a named tuple. Based on the packet format, each packet
    is unpacked into following parameters:
    - timestamp: int (32 bit)
    - IsCommanded: bool (1 bit)
    - voltage channel1: float (16 bit)
    - voltage channel2: float (16 bit)
    - voltage channel3: float (16 bit)
    - voltage channel4: float (16 bit)

    TimeStamp is the time stamp of the packet in seconds.
    IsCommand tells you if the packet was commanded.
    Voltages 1 to 4 are the voltages of corresponding different channels.
    """
    is_commanded: bool
    timestamp: int
    channel1: float
    channel2: float
    channel3: float
    channel4: float

    @classmethod
    def from_bytes(cls, bytes_: bytes):
        structure = struct.unpack(packet_format_sci, bytes_)
        return cls(
            is_commanded=bool(structure[1] & 0x40000000),  # mask to test for commanded event type
            timestamp=structure[1] & 0x3fffffff,           # mask for getting all timestamp bits
            channel1=structure[2] * volts_per_count,
            channel2=structure[3] * volts_per_count,
            channel3=structure[4] * volts_per_count,
            channel4=structure[5] * volts_per_count,
        )


class hk_packet_cls(NamedTuple):
    """
    Class for the housekeeping packet.
    The code unpacks the HK packet into a named tuple. Based on the document and data structure,
    each packet is unpacked into
    - Date: time of the packet as received from the PIT
    - "timestamp",
    - "hk_id" (this tells us what "hk_value" stores inside it),
    - "hk_value",
    - "delta_event_count",
    - "delta_drop_event_count", and
    - "delta_lost_event_count".

    Based on the value of "hk_id", "hk_value" might correspond to value of following parameters:
    NOTE: "hk_id" is a number, and varies from 0 to 15.
    0: PinPuller Temperature
    1: Optics Temperature
    2: LEXI Base Temperature
    3: HV Supply Temperature
    4: Current Correspoding to the HV Supply (5.2V)
    5: Current Correspoding to the HV Supply (10V)
    6: Current Correspoding to the HV Supply (3.3V)
    7: Anode Voltage Monitor
    8: Current Correspoding to the HV Supply (28V)
    9: ADC Ground
    10: Command Count
    11: Pin Puller Armmed
    12: Unused
    13: Unused
    14: MCP HV after auto change
    15: MCP HV after manual change
    """
    Date: int
    timestamp: int
    hk_id: int
    hk_value: float
    delta_event_count: int
    delta_drop_event_count: int
    delta_lost_event_count: int

    @classmethod
    def from_bytes(cls, bytes_: bytes):
        structure_time = struct.unpack(">d", bytes_[2:10])
        structure = struct.unpack(packet_format_hk, bytes_[12:])
        # Check if the present packet is the house-keeping packet. Only the house-keeping packets
        # are processed.
        if structure[1] & 0x80000000:
            Date = structure_time[0]
            timestamp = structure[1] & 0x3fffffff  # mask for getting all timestamp bits
            hk_id = (structure[2] & 0xf000) >> 12  # Down-shift 12 bits to get the hk_id
            if hk_id == 10 or hk_id == 11:
                hk_value = structure[2] & 0xfff
            else:
                hk_value = (structure[2] & 0xfff) << 4  # Up-shift 4 bits to get the hk_value
            delta_event_count = structure[3]
            delta_drop_event_count = structure[4]
            delta_lost_event_count = structure[5]

            return cls(
                Date=Date,
                timestamp=timestamp,
                hk_id=hk_id,
                hk_value=hk_value,
                delta_event_count=delta_event_count,
                delta_drop_event_count=delta_drop_event_count,
                delta_lost_event_count=delta_lost_event_count,
            )


class hk_packet_cls_gsfc(NamedTuple):
    """
    Class for the housekeeping packet.
    The code unpacks the HK packet into a named tuple. Based on the document and data structure,
    each packet is unpacked into
    - "timestamp",
    - "hk_id" (this tells us what "hk_value" stores inside it),
    - "hk_value",
    - "delta_event_count",
    - "delta_drop_event_count", and
    - "delta_lost_event_count".

    Based on the value of "hk_id", "hk_value" might correspond to value of following parameters:
    NOTE: "hk_id" is a number, and varies from 0 to 15.
    0: PinPuller Temperature
    1: Optics Temperature
    2: LEXI Base Temperature
    3: HV Supply Temperature
    4: Current Correspoding to the HV Supply (5.2V)
    5: Current Correspoding to the HV Supply (10V)
    6: Current Correspoding to the HV Supply (3.3V)
    7: Anode Voltage Monitor
    8: Current Correspoding to the HV Supply (28V)
    9: ADC Ground
    10: Command Count
    11: Pin Puller Armmed
    12: Unused
    13: Unused
    14: MCP HV after auto change
    15: MCP HV after manual change
    """
    timestamp: int
    hk_id: int
    hk_value: float
    delta_event_count: int
    delta_drop_event_count: int
    delta_lost_event_count: int

    @classmethod
    def from_bytes(cls, bytes_: bytes):
        structure = struct.unpack(packet_format_hk, bytes_)
        # Check if the present packet is the house-keeping packet. Only the house-keeping packets
        # are processed.
        if structure[1] & 0x80000000:
            timestamp = structure[1] & 0x3fffffff  # mask for getting all timestamp bits
            hk_id = (structure[2] & 0xf000) >> 12  # Down-shift 12 bits to get the hk_id
            if hk_id == 10 or hk_id == 11:
                hk_value = structure[2] & 0xfff
            else:
                hk_value = (structure[2] & 0xfff) << 4  # Up-shift 4 bits to get the hk_value
            delta_event_count = structure[3]
            delta_drop_event_count = structure[4]
            delta_lost_event_count = structure[5]

            return cls(
                timestamp=timestamp,
                hk_id=hk_id,
                hk_value=hk_value,
                delta_event_count=delta_event_count,
                delta_drop_event_count=delta_drop_event_count,
                delta_lost_event_count=delta_lost_event_count,
            )


def read_binary_data_sci(
    in_file_name=None,
    save_file_name="../data/processed/sci/output_sci.csv",
    number_of_decimals=6
):
    """
    Reads science packet of the binary data from a file and saves it to a csv file.

    Parameters
    ----------
    in_file_name : str
        Name of the input file. Default is None.
    save_file_name : str
        Name of the output file. Default is "output_sci.csv".
    number_of_decimals : int
        Number of decimals to save. Default is 6.

    Raises
    ------
    FileNotFoundError :
        If the input file does not exist.
    TypeError :
        If the name of the input file or input directory is not a string. Or if the number of
        decimals is not an integer.
    Returns
    -------
        df : pandas.DataFrame
            DataFrame of the science packet.
        save_file_name : str
            Name of the output file.
    """
    if in_file_name is None:
        in_file_name = "../data/raw_data/2022_03_03_1030_LEXI_raw_2100_newMCP_copper.txt"

    # Check if the file exists, if does not exist raise an error
    if not Path(in_file_name).is_file():
        raise FileNotFoundError(
            "The file " + in_file_name + " does not exist."
        )
    # Check if the file name and folder name are strings, if not then raise an error
    if not isinstance(in_file_name, str):
        raise TypeError(
            "The file name must be a string."
        )

    # Check the number of decimals to save
    if not isinstance(number_of_decimals, int):
        raise TypeError(
            "The number of decimals to save must be an integer."
        )

    input_file_name = in_file_name

    with open(input_file_name, 'rb') as file:
        raw = file.read()

    index = 0
    packets = []

    # Check if the word 'mcp' is present in the file name
    if "mcp" in in_file_name:
        while index < len(raw) - 16:
            if raw[index:index + 4] == sync_lxi:
                packets.append(sci_packet_gsfc.from_bytes(raw[index:index + 16]))
                index += 16
                continue

            index += 1

    else:
        while index < len(raw) - 28:
            if raw[index:index + 2] == sync_pit and raw[index + 12:index + 16] == sync_lxi:
                packets.append(sci_packet.from_bytes(raw[index:index + 28]))
                index += 28
                continue

            index += 1

    # Split the file name in a folder and a file name
    output_file_name = in_file_name.split("/")[-1].split(".")[0] + "_sci_output.csv"
    output_folder_name = "/".join(in_file_name.split("/")[:-2]) + "/processed_data/sci"

    save_file_name = output_folder_name + "/" + output_file_name

    # Check if the save folder exists, if not then create it
    if not Path(output_folder_name).exists():
        Path(output_folder_name).mkdir(parents=True, exist_ok=True)

    if "mcp" in in_file_name:
        # Get the current datetime
        now = datetime.datetime.now()

        with open(save_file_name, 'w', newline='') as file:
            dict_writer = csv.DictWriter(
                file,
                fieldnames=(
                    'Date',
                    'TimeStamp',
                    'IsCommanded',
                    'Channel1',
                    'Channel2',
                    'Channel3',
                    'Channel4'
                ),
            )
            dict_writer.writeheader()
            dict_writer.writerows(
                {
                    'Date': now,
                    'TimeStamp': sci_packet.timestamp,
                    'IsCommanded': sci_packet.is_commanded,
                    'Channel1': np.round(sci_packet.channel1, decimals=number_of_decimals),
                    'Channel2': np.round(sci_packet.channel2, decimals=number_of_decimals),
                    'Channel3': np.round(sci_packet.channel3, decimals=number_of_decimals),
                    'Channel4': np.round(sci_packet.channel4, decimals=number_of_decimals)
                }
                for sci_packet in packets
            )
    else:
        with open(save_file_name, 'w', newline='') as file:
            dict_writer = csv.DictWriter(
                file,
                fieldnames=(
                    'Date',
                    'TimeStamp',
                    'IsCommanded',
                    'Channel1',
                    'Channel2',
                    'Channel3',
                    'Channel4'
                ),
            )
            dict_writer.writeheader()
            dict_writer.writerows(
                {
                    'Date': datetime.datetime.utcfromtimestamp(sci_packet.Date),
                    'TimeStamp': sci_packet.timestamp / 1e3,
                    'IsCommanded': sci_packet.is_commanded,
                    'Channel1': np.round(sci_packet.channel1, decimals=number_of_decimals),
                    'Channel2': np.round(sci_packet.channel2, decimals=number_of_decimals),
                    'Channel3': np.round(sci_packet.channel3, decimals=number_of_decimals),
                    'Channel4': np.round(sci_packet.channel4, decimals=number_of_decimals)
                }
                for sci_packet in packets
            )

    # Read the saved file data in a dataframe
    df = pd.read_csv(save_file_name)

    # Convert the date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Set index to the date
    df.set_index('Date', inplace=False)

    # Save the dataframe to a csv file
    df.to_csv(save_file_name, index=False)

    return df, save_file_name


def read_binary_data_hk(
    in_file_name=None,
    save_file_name="../data/processed/hk/output_hk.csv",
    number_of_decimals=6
):
    """
    Reads housekeeping packet of the binary data from a file and saves it to a csv file.

    Parameters
    ----------
    in_file_name : str
        Name of the input file. Default is None.
    save_file_name : str
        Name of the output file. Default is "output_hk.csv".
    number_of_decimals : int
        Number of decimals to save. Default is 6.

    Raises
    ------
    FileNotFoundError :
        If the input file does not exist.
    TypeError :
        If the name of the input file or input directory is not a string. Or if the number of
        decimals is not an integer.
    Returns
    -------
        df : pandas.DataFrame
            DataFrame of the housekeeping packet.
        save_file_name : str
            Name of the output file.

    Raises
    ------
    FileNotFoundError :
        If the input file does not exist or isn't a specified
    """
    if in_file_name is None:
        raise FileNotFoundError(
            "The input file name must be specified."
        )

    # Check if the file exists, if does not exist raise an error
    if not Path(in_file_name).is_file():
        raise FileNotFoundError(
            "The file " + in_file_name + " does not exist."
        )
    # Check if the file name and folder name are strings, if not then raise an error
    if not isinstance(in_file_name, str):
        raise TypeError(
            "The file name must be a string."
        )

    # Check the number of decimals to save
    if not isinstance(number_of_decimals, int):
        raise TypeError(
            "The number of decimals to save must be an integer."
        )

    input_file_name = in_file_name

    with open(input_file_name, 'rb') as file:
        raw = file.read()

    index = 0
    packets = []

    if "mcp" in in_file_name:
        while index < len(raw) - 16:
            if raw[index:index + 4] == sync_lxi:
                packets.append(hk_packet_cls_gsfc.from_bytes(raw[index:index + 16]))
                index += 16
                continue

            index += 1
    else:
        while index < len(raw) - 28:
            if raw[index:index + 2] == sync_pit and raw[index + 12:index + 16] == sync_lxi:
                packets.append(hk_packet_cls.from_bytes(raw[index:index + 28]))
                index += 28
                continue

            index += 1

    # Get only those packets that have the HK data
    hk_idx = []
    for idx, hk_packet in enumerate(packets):
        if hk_packet is not None:
            hk_idx.append(idx)

    Date = np.full(len(hk_idx), np.nan)
    TimeStamp = np.full(len(hk_idx), np.nan)
    HK_id = np.full(len(hk_idx), np.nan)
    PinPullerTemp = np.full(len(hk_idx), np.nan)
    OpticsTemp = np.full(len(hk_idx), np.nan)
    LEXIbaseTemp = np.full(len(hk_idx), np.nan)
    HVsupplyTemp = np.full(len(hk_idx), np.nan)
    V_Imon_5_2 = np.full(len(hk_idx), np.nan)
    V_Imon_10 = np.full(len(hk_idx), np.nan)
    V_Imon_3_3 = np.full(len(hk_idx), np.nan)
    AnodeVoltMon = np.full(len(hk_idx), np.nan)
    V_Imon_28 = np.full(len(hk_idx), np.nan)
    ADC_Ground = np.full(len(hk_idx), np.nan)
    Cmd_count = np.full(len(hk_idx), np.nan)
    Pinpuller_Armed = np.full(len(hk_idx), np.nan)
    Unused1 = np.full(len(hk_idx), np.nan)
    Unused2 = np.full(len(hk_idx), np.nan)
    HVmcpAuto = np.full(len(hk_idx), np.nan)
    HVmcpMan = np.full(len(hk_idx), np.nan)
    DeltaEvntCount = np.full(len(hk_idx), np.nan)
    DeltaDroppedCount = np.full(len(hk_idx), np.nan)
    DeltaLostEvntCount = np.full(len(hk_idx), np.nan)

    all_data_dict = {"Date": Date, "TimeStamp": TimeStamp, "HK_id": HK_id,
                     "0": PinPullerTemp, "1": OpticsTemp, "2": LEXIbaseTemp, "3": HVsupplyTemp,
                     "4": V_Imon_5_2, "5": V_Imon_10, "6": V_Imon_3_3, "7": AnodeVoltMon,
                     "8": V_Imon_28, "9": ADC_Ground, "10": Cmd_count, "11": Pinpuller_Armed,
                     "12": Unused1, "13": Unused2, "14": HVmcpAuto, "15": HVmcpMan,
                     "DeltaEvntCount": DeltaEvntCount, "DeltaDroppedCount": DeltaDroppedCount,
                     "DeltaLostEvntCount": DeltaLostEvntCount}

    selected_keys = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
                     "15"]

    # Check if "unit_1" or "unit1" is in the file name, if so then the data is from the unit 1
    if "unit_1" in input_file_name or "unit1" in input_file_name:
        lxi_unit = 1
    elif "unit_2" in input_file_name or "unit2" in input_file_name:
        lxi_unit = 2
    else:
        # Print warning that unit is defaulted to 1
        print(
            "\n FileName Warning: \033[91m \nThe unit is defaulted to 1 because the name of the "
            "file does not contain \"unit_1\" or \"unit1\" or \"unit_2\" or \"unit2\". \033[0m \n"
        )
        lxi_unit = 1

    if "mcp" in input_file_name:
        now = datetime.datetime.now()
        # Convert now to a float
        now = now.timestamp()
    for ii, idx in enumerate(hk_idx):
        hk_packet = packets[idx]
        # Convert to seconds from milliseconds for the timestamp
        if "mc" in input_file_name:
            all_data_dict["Date"][ii] = now
        else:
            all_data_dict["Date"][ii] = hk_packet.Date
        all_data_dict["TimeStamp"][ii] = hk_packet.timestamp / 1e3
        all_data_dict["HK_id"][ii] = hk_packet.hk_id
        key = str(hk_packet.hk_id)
        if key in selected_keys:
            all_data_dict[key][ii] = lmsc.hk_value_comp(ii=ii,
                                                        vpc=volts_per_count,
                                                        hk_value=hk_packet.hk_value,
                                                        hk_id=hk_packet.hk_id,
                                                        lxi_unit=lxi_unit
                                                        )

        all_data_dict["DeltaEvntCount"][ii] = hk_packet.delta_event_count
        all_data_dict["DeltaDroppedCount"][ii] = hk_packet.delta_drop_event_count
        all_data_dict["DeltaLostEvntCount"][ii] = hk_packet.delta_lost_event_count

    # Create a dataframe with the data
    df_key_list = ["Date", "TimeStamp", "HK_id", "PinPullerTemp", "OpticsTemp", "LEXIbaseTemp",
                   "HVsupplyTemp", "+5.2V_Imon", "+10V_Imon", "+3.3V_Imon", "AnodeVoltMon",
                   "+28V_Imon", "ADC_Ground", "Cmd_count", "Pinpuller_Armed", "Unused1", "Unused2",
                   "HVmcpAuto", "HVmcpMan", "DeltaEvntCount", "DeltaDroppedCount",
                   "DeltaLostEvntCount"]

    Date_datetime = [datetime.datetime.utcfromtimestamp(x) for x in all_data_dict["Date"]]

    df = pd.DataFrame(columns=df_key_list)
    for ii, key in enumerate(df_key_list):
        df[key] = all_data_dict[list(all_data_dict.keys())[ii]]

    # For the dataframe, replace the nans with the value from the previous index.
    # This is to make sure that the file isn't inundated with nans.
    for key in df.keys():
        for ii in range(1, len(df[key])):
            if np.isnan(df[key][ii]):
                df[key][ii] = df[key][ii - 1]

    # Set the index to the TimeStamp
    df["Date"] = Date_datetime

    # Split the file name in a folder and a file name
    output_file_name = in_file_name.split("/")[-1].split(".")[0] + "_hk_output.csv"
    output_folder_name = "/".join(in_file_name.split("/")[:-2]) + "/processed_data/hk"

    save_file_name = output_folder_name + "/" + output_file_name

    # Check if the save folder exists, if not then create it
    if not Path(output_folder_name).exists():
        Path(output_folder_name).mkdir(parents=True, exist_ok=True)

    # Save the dataframe to a csv file
    df.to_csv(save_file_name, index=False)

    return df, save_file_name


def open_file_sci(start_time=None, end_time=None):
    # define a global variable for the file name

    file_val = filedialog.askopenfilename(initialdir="../data/processed_data/sci/",
                                          title="Select file",
                                          filetypes=(("csv files", "*.csv"),
                                                     ("all files", "*.*"))
                                          )
    # Cut path to the file off
    file_name_sci = file_val.split('/')[-1]
    global_variables.all_file_details['file_name_sci'] = file_val

    df_all_sci, df_slice_sci = read_csv_sci(file_val=file_val, t_start=start_time, t_end=end_time)
    global_variables.all_file_details['df_slice_sci'] = df_slice_sci
    global_variables.all_file_details['df_all_sci'] = df_all_sci
    print(f"\n \x1b[1;32;255m Loaded {file_name_sci} in the data base \x1b[0m")

    return file_val


def open_file_hk():
    # define a global variable for the file name
    file_val = filedialog.askopenfilename(initialdir="../data/processed_data/hk/",
                                          title="Select file",
                                          filetypes=(("csv files", "*.csv"),
                                                     ("all files", "*.*"))
                                          )
    # Cut path to the file off
    file_name_hk = file_val.split('/')[-1]
    global_variables.all_file_details['file_name_hk'] = file_val

    df_all_hk, df_slice_hk = read_csv_hk(file_val)
    global_variables.all_file_details['df_slice_hk'] = df_slice_hk
    global_variables.all_file_details['df_all_hk'] = df_all_hk
    print(f"\n \x1b[1;32;255m Loaded {file_name_hk} in the data base \x1b[0m")
    return file_val


def open_file_b():
    # define a global variable for the file name
    file_val = filedialog.askopenfilename(initialdir="../data/raw_data/",
                                          title="Select file",
                                          filetypes=(("all files", "*.*"),
                                                     ("text files", "*.txt"))
                                          )

    # Cut path to the file off
    file_name_b = file_val
    (df_slice_hk, file_name_hk, df_slice_sci, file_name_sci, df_all_hk, df_all_sci
     ) = read_binary_file(file_val)
    global_variables.all_file_details["file_name_b"] = file_name_b
    global_variables.all_file_details["file_name_hk"] = file_name_hk
    global_variables.all_file_details["file_name_sci"] = file_name_sci

    global_variables.all_file_details["df_slice_hk"] = df_slice_hk
    global_variables.all_file_details["df_slice_sci"] = df_slice_sci
    global_variables.all_file_details["df_all_hk"] = df_all_hk
    global_variables.all_file_details["df_all_sci"] = df_all_sci

    print(
        f"\n Loaded \x1b[1;32;255m{file_name_b}\x1b[0m in the data base,\n  and the csv file for "
        f"\x1b[1;32;255m HK \x1b[0m and \x1b[1;32;255m SCI \x1b[0m data have been saved to \n "
        f"HK File : \x1b[1;32;255m{file_name_hk} \x1b[0m \n and \n Sci File: "
        f"\x1b[1;32;255m{file_name_sci}\x1b[0m")

    return file_val


def lin_correction(x, y, M_inv=np.array([[0.98678, 0.16204],
                                         [0.11385, 0.993497]]),
                   b=np.array([0.00195, 0.56355])):
    """
    Function to apply nonlinearity correction to MCP position x/y data
    # TODO: Add correct M_inv matrix and the offsets
    """
    x_lin = (x * M_inv[0, 0] + y * M_inv[0, 1]) - b[0]
    y_lin = (x * M_inv[1, 0] + y * M_inv[1, 1])

    return x_lin, y_lin


def volt_to_mcp(x, y):
    """
    Function to convert voltage coordinates to MCP coordinates
    """
    x_mcp = (x - 0.544) * 78.55
    y_mcp = (y - 0.564) * 78.55

    return x_mcp, y_mcp


def compute_position(v1=None, v2=None, n_bins=401, bin_min=0, bin_max=4):
    """
    The function computes the position of the particle in the xy-plane. The ratios to compute
    both the x and y position are taken from Dennis' code. The code computes the offset of the
    four voltages as measured by LEXI. We then subtract the offset from the four voltages to get
    the shifted voltages and then compute the position of the particle based on the shifted
    voltages.

    Parameters
    ----------
    v1 : float
        Voltage of the first channel. Default is None.
    v2 : float
        Voltage of the second channel. Default is None.
    n_bins : int
        Number of bins to compute the position. Default is 401.
    bin_min : float
        Minimum value of the bin. Default is 0.
    bin_max : float
        Maximum value of the bin. Default is 4.

    Returns
    -------
    particle_pos : float
        position of the particle along one of the axis. Whether it gives x or y position depends
        on which voltages were provided. For example, if v1 and v3 were provided, then the x
        position is returned. Else if v4 and v2 were provided, then the y position is returned.
        It is important to note that the order of the voltages is important.
    v1_shift: float
        Offset corrected voltage of the first channel.
    v2_shift: float
        Offset corrected voltage of the second channel.
    """
    bin_size = (bin_max - bin_min) / (n_bins - 1)

    # make 1-D histogram of all 4 channels
    hist_v1 = np.histogram(v1, bins=n_bins, range=(bin_min, bin_max))
    hist_v2 = np.histogram(v2, bins=n_bins, range=(bin_min, bin_max))

    xx = bin_min + bin_size * np.arange(n_bins)

    # Find the index where the histogram is the maximum
    # NOTE/TODO: I don't quite understand why the offset is computed this way. Need to talk to
    # Dennis about this and get an engineering/physics reason for it.
    max_index_v1 = np.argmax(hist_v1[0][0:int(n_bins / 2)])
    max_index_v2 = np.argmax(hist_v2[0][0:int(n_bins / 2)])

    z1_min = 1000 * xx[max_index_v1]
    z2_min = 1000 * xx[max_index_v2]

    n1_z = z1_min / 1000
    n2_z = z2_min / 1000

    v1_shift = v1 - n1_z
    v2_shift = v2 - n2_z

    particle_pos = v2_shift / (v2_shift + v1_shift)

    return particle_pos, v1_shift, v2_shift


def read_csv_sci(file_val=None, t_start=None, t_end=None):
    """
    Reads a csv file and returns a pandas dataframe for the selected time range along with x and
    y-coordinates.

    Parameters
    ----------
    file_val : str
        Path to the input file. Default is None.
    t_start : float
        Start time of the data. Default is None.
    t_end : float
        End time of the data. Default is None.
    """

    df = pd.read_csv(file_val, index_col=False)

    # Check all the keys and find out which one has the word "time" in it
    for key in df.keys():
        if "time" in key.lower():
            time_col = key
            break
    # Rename the time column to TimeStamp
    df.rename(columns={time_col: 'TimeStamp'}, inplace=True)

    # Convert the Date column from string to datetime in utc
    try:
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
    except Exception:
        # Convert timestamp to datetime and set it to Date
        df['Date'] = pd.to_datetime(df['TimeStamp'], unit='s', utc=True)

    # Set the index to the time column
    df.set_index('Date', inplace=True)
    # Sort the dataframe by timestamp
    df = df.sort_index()

    if t_start is None:
        t_start = df.index.min()
    if t_end is None:
        t_end = df.index.max()

    # Select dataframe from timestamp t_start to t_end
    df_slice_sci = df.loc[t_start:t_end]

    # For both the sliced and entire dataframes, compute the x and y-coordinates and the shift in
    # the voltages
    x_slice, v1_shift_slice, v3_shift_slice = compute_position(v1=df_slice_sci['Channel1'],
                                                               v2=df_slice_sci['Channel3'],
                                                               n_bins=401, bin_min=0, bin_max=4)

    x, v1_shift, v3_shift = compute_position(v1=df['Channel1'], v2=df['Channel3'], n_bins=401,
                                             bin_min=0, bin_max=4)

    y_slice, v4_shift_slice, v2_shift_slice = compute_position(v1=df_slice_sci['Channel4'],
                                                               v2=df_slice_sci['Channel2'],
                                                               n_bins=401, bin_min=0, bin_max=4)

    y, v4_shift, v2_shift = compute_position(v1=df['Channel4'], v2=df['Channel2'], n_bins=401,
                                             bin_min=0, bin_max=4)

    # Correct for the non-linearity in the positions
    x_lin_slice, y_lin_slice = lin_correction(x_slice, y_slice)
    x_lin, y_lin = lin_correction(x, y)

    # Get the x,y value in mcp units
    x_mcp_slice, y_mcp_slice = volt_to_mcp(x_slice, y_slice)
    x_mcp, y_mcp = volt_to_mcp(x, y)
    x_mcp_lin_slice, y_mcp_lin_slice = volt_to_mcp(x_lin_slice, y_lin_slice)
    x_mcp_lin, y_mcp_lin = volt_to_mcp(x_lin, y_lin)

    # Add the x-coordinate to the dataframe
    df_slice_sci.loc[:, 'x_val'] = x_slice
    df_slice_sci.loc[:, 'x_val_lin'] = x_lin_slice
    df_slice_sci.loc[:, 'x_mcp'] = x_mcp_slice
    df_slice_sci.loc[:, 'x_mcp_lin'] = x_mcp_lin_slice
    df_slice_sci.loc[:, 'v1_shift'] = v1_shift_slice
    df_slice_sci.loc[:, 'v3_shift'] = v3_shift_slice

    df.loc[:, 'x_val'] = x
    df.loc[:, 'x_val_lin'] = x_lin
    df.loc[:, 'x_mcp'] = x_mcp
    df.loc[:, 'x_mcp_lin'] = x_mcp_lin
    df.loc[:, 'v1_shift'] = v1_shift
    df.loc[:, 'v3_shift'] = v3_shift

    # Add the y-coordinate to the dataframe
    df_slice_sci.loc[:, 'y_val'] = y_slice
    df_slice_sci.loc[:, 'y_val_lin'] = y_lin_slice
    df_slice_sci.loc[:, 'y_mcp'] = y_mcp_slice
    df_slice_sci.loc[:, 'y_mcp_lin'] = y_mcp_lin_slice
    df_slice_sci.loc[:, 'v4_shift'] = v4_shift_slice
    df_slice_sci.loc[:, 'v2_shift'] = v2_shift_slice

    df.loc[:, 'y_val'] = y
    df.loc[:, 'y_val_lin'] = y_lin
    df.loc[:, 'y_mcp'] = y_mcp
    df.loc[:, 'y_mcp_lin'] = y_mcp_lin
    df.loc[:, 'v4_shift'] = v4_shift
    df.loc[:, 'v2_shift'] = v2_shift

    return df, df_slice_sci


def read_csv_hk(file_val=None, t_start=None, t_end=None):
    """
    Reads a csv file and returns a pandas dataframe for the selected time range along with x and
    y-coordinates.

    Parameters
    ----------
    file_val : str
        Path to the input file. Default is None.
    t_start : float
        Start time of the data. Default is None.
    t_end : float
        End time of the data. Default is None.
    """

    global df_slice_hk
    df = pd.read_csv(file_val)

    # Check all the keys and find out which one has the word "time" in it
    for key in df.keys():
        if "time" in key.lower():
            time_col = key
            break
    # Rename the time column to TimeStamp
    df.rename(columns={time_col: 'TimeStamp'}, inplace=True)

    # Convert the Date column from string to datetime in utc
    try:
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
    except Exception:
        # Convert timestamp to datetime and set it to Date
        df['Date'] = pd.to_datetime(df['TimeStamp'], unit='s', utc=True)

    # Set the index to the time column
    df.set_index('Date', inplace=True)
    # Sort the dataframe by timestamp
    df = df.sort_index()

    if t_start is None:
        t_start = df.index.min()
    if t_end is None:
        t_end = df.index.max()

    # Select dataframe from timestamp t_start to t_end
    df_slice_hk = df.loc[t_start:t_end]

    return df, df_slice_hk


def read_binary_file(file_val=None, t_start=None, t_end=None):
    """
    Reads the binary file using functions saved in the file "lxi_read_binary_data.py" and returns
    a pandas dataframe for the selected time range along with x and y-coordinates.

    Parameters
    ----------
    file_val : str
        Path to the input file. Default is None.
    t_start : float
        Start time of the data. Default is None.
    t_end : float
        End time of the data. Default is None.

    Returns
    -------
    df_slice_hk : pandas.DataFrame
        The Housekeeping dataframe for the selected time range.
    df_slice_sci : pandas.DataFrame
        The Science dataframe for the selected time range.
    df_hk : pandas.DataFrame
        The Housekeeping dataframe for the entire time range in the file.
    df_sci : pandas.DataFrame
        The Science dataframe for the entire time range in the file.
    file_name_hk : str
        The name of the Housekeeping file.
    file_name_sci : str
        The name of the Science file.
    """

    # Read the housekeeping data
    df_hk, file_name_hk = read_binary_data_hk(
        in_file_name=file_val,
        save_file_name=None,
        number_of_decimals=6
    )

    # Read the science data
    df_sci, file_name_sci = read_binary_data_sci(
        in_file_name=file_val,
        save_file_name=None,
        number_of_decimals=6
    )

    # Replace index with timestamp
    df_hk.set_index('Date', inplace=True)
    df_sci.set_index('Date', inplace=True)

    # Sort the dataframe by timestamp
    df_hk = df_hk.sort_index()
    df_sci = df_sci.sort_index()

    if t_start is None:
        t_start = df_sci.index.min()
    if t_end is None:
        t_end = df_sci.index.max()

    df_sci, df_slice_sci = read_csv_sci(
        file_val=file_name_sci, t_start=t_start, t_end=t_end)

    df_hk, df_slice_hk = read_csv_hk(
        file_val=file_name_hk, t_start=t_start, t_end=t_end)

    # Select only those where "IsCommanded" is True
    df_slice_sci = df_slice_sci[df_slice_sci['IsCommanded']==False]
    df_sci = df_sci[df_sci['IsCommanded']==False]
    """
    # Select dataframe from timestamp t_start to t_end
    df_slice_hk = df_hk.loc[t_start:t_end]
    df_slice_sci = df_sci.loc[t_start:t_end]

    # For both the sliced and entire dataframes, compute the x and y-coordinates and the shift in
    # the voltages
    x_slice, v1_shift_slice, v3_shift_slice = compute_position(v1=df_slice_sci['Channel1'],
                                                               v2=df_slice_sci['Channel3'],
                                                               n_bins=401, bin_min=0, bin_max=4)

    x, v1_shift, v3_shift = compute_position(v1=df_sci['Channel1'], v2=df_sci['Channel3'],
                                             n_bins=401, bin_min=0, bin_max=4)

    # Add the x-coordinate to the dataframe
    df_slice_sci.loc[:, 'x_val'] = x_slice
    df_slice_sci.loc[:, 'v1_shift'] = v1_shift_slice
    df_slice_sci.loc[:, 'v3_shift'] = v3_shift_slice

    df_sci.loc[:, 'x_val'] = x
    df_sci.loc[:, 'v1_shift'] = v1_shift
    df_sci.loc[:, 'v3_shift'] = v3_shift

    y_slice, v4_shift_slice, v2_shift_slice = compute_position(v1=df_slice_sci['Channel4'],
                                                               v2=df_slice_sci['Channel2'],
                                                               n_bins=401, bin_min=0, bin_max=4)

    y, v4_shift, v2_shift = compute_position(v1=df_sci['Channel4'], v2=df_sci['Channel2'],
                                             n_bins=401, bin_min=0, bin_max=4)

    # Add the y-coordinate to the dataframe
    df_slice_sci.loc[:, 'y_val'] = y_slice
    df_slice_sci.loc[:, 'v4_shift'] = v4_shift_slice
    df_slice_sci.loc[:, 'v2_shift'] = v2_shift_slice

    df_sci.loc[:, 'y_val'] = y
    df_sci.loc[:, 'v4_shift'] = v4_shift
    df_sci.loc[:, 'v2_shift'] = v2_shift
    """

    return df_slice_hk, file_name_hk, df_slice_sci, file_name_sci, df_hk, df_sci
