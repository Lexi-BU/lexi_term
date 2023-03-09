import csv
import struct
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# Tha packet format of the science and housekeeping packets
packet_format_sci = ">II4H"
# signed lower case, unsigned upper case (b)
packet_format_hk = ">II4H"

sync = b'\xfe\x6b\x28\x40'
volts_per_count = 0.000068817  # volts per increment of digitization


class sci_packet(NamedTuple):
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

    while index < len(raw) - 16:
        if raw[index:index + 4] == sync:
            packets.append(sci_packet.from_bytes(raw[index:index + 16]))
            index += 16
            continue

        index += 1

    # Split the file name in a folder and a file name
    output_file_name = in_file_name.split("/")[-1].split(".")[0] + "_sci_output.csv"
    output_folder_name = "/".join(in_file_name.split("/")[:-2]) + "/processed_data/sci"

    save_file_name = output_folder_name + "/" + output_file_name

    # Check if the save folder exists, if not then create it
    if not Path(output_folder_name).exists():
        Path(output_folder_name).mkdir(parents=True, exist_ok=True)

    with open(save_file_name, 'w', newline='') as file:
        dict_writer = csv.DictWriter(
            file,
            fieldnames=(
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
                'TimeStamp': sci_packet.timestamp,
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

    # Save the dataframe to a csv file and set index to time stamp
    df.to_csv(save_file_name, index=True)

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

    while index < len(raw) - 16:
        if raw[index:index + 4] == sync:
            packets.append(hk_packet_cls.from_bytes(raw[index:index + 16]))
            index += 16
            continue

        index += 1

    # Get only those packets that have the HK data
    hk_idx = []
    for idx, hk_packet in enumerate(packets):
        if hk_packet is not None:
            hk_idx.append(idx)

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
    DeltaLostevntCount = np.full(len(hk_idx), np.nan)

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

    for ii, idx in enumerate(hk_idx):
        hk_packet = packets[idx]
        TimeStamp[ii] = hk_packet.timestamp
        HK_id[ii] = hk_packet.hk_id
        if hk_packet.hk_id == 0:
            PinPullerTemp[ii] = (hk_packet.hk_value * volts_per_count - 2.73) * 100
        elif hk_packet.hk_id == 1:
            OpticsTemp[ii] = (hk_packet.hk_value * volts_per_count - 2.73) * 100
        elif hk_packet.hk_id == 2:
            LEXIbaseTemp[ii] = (hk_packet.hk_value * volts_per_count - 2.73) * 100
        elif hk_packet.hk_id == 3:
            HVsupplyTemp[ii] = (hk_packet.hk_value * volts_per_count - 2.73) * 100
        elif hk_packet.hk_id == 4:
            if lxi_unit == 1:
                V_Imon_5_2[ii] = (hk_packet.hk_value * volts_per_count) * 1e3 / 18
            elif lxi_unit == 2:
                V_Imon_5_2[ii] = (hk_packet.hk_value - 1.129) / 21.456
        elif hk_packet.hk_id == 5:
            V_Imon_10[ii] = hk_packet.hk_value * volts_per_count
        elif hk_packet.hk_id == 6:
            if lxi_unit == 1:
                V_Imon_3_3[ii] = (hk_packet.hk_value * volts_per_count + 0.0178) * 1e3 / 9.131
            elif lxi_unit == 2:
                V_Imon_3_3[ii] = (hk_packet.hk_value * volts_per_count + 0.029) * 1e3 / 18
        elif hk_packet.hk_id == 7:
            AnodeVoltMon[ii] = hk_packet.hk_value * volts_per_count
        elif hk_packet.hk_id == 8:
            if lxi_unit == 1:
                V_Imon_28[ii] = (hk_packet.hk_value * volts_per_count + 0.00747) * 1e3 / 17.94
            elif lxi_unit == 2:
                V_Imon_28[ii] = (hk_packet.hk_value * volts_per_count + 0.00747) * 1e3 / 17.94
        elif hk_packet.hk_id == 9:
            ADC_Ground[ii] = hk_packet.hk_value * volts_per_count
        elif hk_packet.hk_id == 10:
            Cmd_count[ii] = hk_packet.hk_value
        elif hk_packet.hk_id == 11:
            Pinpuller_Armed[ii] = hk_packet.hk_value
        elif hk_packet.hk_id == 12:
            Unused1[ii] = hk_packet.hk_value
        elif hk_packet.hk_id == 13:
            Unused2[ii] = hk_packet.hk_value
        elif hk_packet.hk_id == 14:
            HVmcpAuto[ii] = hk_packet.hk_value * volts_per_count
        elif hk_packet.hk_id == 15:
            HVmcpMan[ii] = hk_packet.hk_value * volts_per_count

        DeltaEvntCount[ii] = hk_packet.delta_event_count
        DeltaDroppedCount[ii] = hk_packet.delta_drop_event_count
        DeltaLostevntCount[ii] = hk_packet.delta_lost_event_count

    # For observations which get their values from "HK_value", go through the whole array and
    # replace the nans at any index with the value from the previous index.
    # This is to make sure that the file isn't inundated with nans.
    for ii in range(1, len(TimeStamp)):
        if np.isnan(PinPullerTemp[ii]):
            PinPullerTemp[ii] = PinPullerTemp[ii - 1]
        if np.isnan(OpticsTemp[ii]):
            OpticsTemp[ii] = OpticsTemp[ii - 1]
        if np.isnan(LEXIbaseTemp[ii]):
            LEXIbaseTemp[ii] = LEXIbaseTemp[ii - 1]
        if np.isnan(HVsupplyTemp[ii]):
            HVsupplyTemp[ii] = HVsupplyTemp[ii - 1]
        if np.isnan(V_Imon_5_2[ii]):
            V_Imon_5_2[ii] = V_Imon_5_2[ii - 1]
        if np.isnan(V_Imon_10[ii]):
            V_Imon_10[ii] = V_Imon_10[ii - 1]
        if np.isnan(V_Imon_3_3[ii]):
            V_Imon_3_3[ii] = V_Imon_3_3[ii - 1]
        if np.isnan(AnodeVoltMon[ii]):
            AnodeVoltMon[ii] = AnodeVoltMon[ii - 1]
        if np.isnan(V_Imon_28[ii]):
            V_Imon_28[ii] = V_Imon_28[ii - 1]
        if np.isnan(ADC_Ground[ii]):
            ADC_Ground[ii] = ADC_Ground[ii - 1]
        if np.isnan(Cmd_count[ii]):
            Cmd_count[ii] = Cmd_count[ii - 1]
        if np.isnan(Pinpuller_Armed[ii]):
            Pinpuller_Armed[ii] = Pinpuller_Armed[ii - 1]
        if np.isnan(Unused1[ii]):
            Unused1[ii] = Unused1[ii - 1]
        if np.isnan(Unused2[ii]):
            Unused2[ii] = Unused2[ii - 1]
        if np.isnan(HVmcpAuto[ii]):
            HVmcpAuto[ii] = HVmcpAuto[ii - 1]
        if np.isnan(HVmcpMan[ii]):
            HVmcpMan[ii] = HVmcpMan[ii - 1]

    # Split the file name in a folder and a file name
    output_file_name = in_file_name.split("/")[-1].split(".")[0] + "_hk_output.csv"
    output_folder_name = "/".join(in_file_name.split("/")[:-2]) + "/processed_data/hk"

    save_file_name = output_folder_name + "/" + output_file_name

    # Check if the save folder exists, if not then create it
    if not Path(output_folder_name).exists():
        Path(output_folder_name).mkdir(parents=True, exist_ok=True)

    with open(save_file_name, 'w', newline='') as file:
        dict_writer = csv.DictWriter(
            file,
            fieldnames=(
                "TimeStamp",
                "HK_id",
                "PinPullerTemp",
                "OpticsTemp",
                "LEXIbaseTemp",
                "HVsupplyTemp",
                "+5.2V_Imon",
                "+10V_Imon",
                "+3.3V_Imon",
                "AnodeVoltMon",
                "+28V_Imon",
                "ADC_Ground",
                "Cmd_count",
                "Pinpuller_Armed",
                "Unused1",
                "Unused2",
                "HVmcpAuto",
                "HVmcpMan",
                "DeltaEvntCount",
                "DeltaDroppedCount",
                "DeltaLostevntCount"
            ),
        )
        dict_writer.writeheader()

        dict_writer.writerows(
            {
                "TimeStamp": TimeStamp[ii],
                "HK_id": HK_id[ii],
                "PinPullerTemp": PinPullerTemp[ii],
                "OpticsTemp": OpticsTemp[ii],
                "LEXIbaseTemp": LEXIbaseTemp[ii],
                "HVsupplyTemp": HVsupplyTemp[ii],
                "+5.2V_Imon": V_Imon_5_2[ii],
                "+10V_Imon": V_Imon_10[ii],
                "+3.3V_Imon": V_Imon_3_3[ii],
                "AnodeVoltMon": AnodeVoltMon[ii],
                "+28V_Imon": V_Imon_28[ii],
                "ADC_Ground": ADC_Ground[ii],
                "Cmd_count": Cmd_count[ii],
                "Pinpuller_Armed": Pinpuller_Armed[ii],
                "Unused1": Unused1[ii],
                "Unused2": Unused2[ii],
                "HVmcpAuto": HVmcpAuto[ii],
                "HVmcpMan": HVmcpMan[ii],
                "DeltaEvntCount": DeltaEvntCount[ii],
                "DeltaDroppedCount": DeltaDroppedCount[ii],
                "DeltaLostevntCount": DeltaLostevntCount[ii],
            }
            for ii in range(len(hk_idx))
        )
    return packets


in_file_path = "../data/GSFC/2022_04_21_1431_LEXI_HK_unit_1_mcp_unit_1_eBox_1987_hk_/"
in_file_name = "../data/GSFC/2022_04_21_1431_LEXI_HK_unit_1_mcp_unit_1_eBox_1987_hk_/2022_04_21_1431_LEXI_raw_LEXI_unit_1_mcp_unit_1_eBox-1987.txt"
save_file_path_sci = "../data/processed_data/sci/"
save_file_path_hk = "../data/processed_data/hk/"
save_file_name = f"{in_file_name[:-4]}_qudsi.csv"

pkts = read_binary_data_hk(
    in_file_name=in_file_name,
    save_file_name=save_file_name,
    number_of_decimals=6
)

pkts = read_binary_data_sci(
    in_file_name=in_file_name,
    save_file_name=save_file_name,
    number_of_decimals=6
)
