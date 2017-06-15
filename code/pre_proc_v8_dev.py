import os
import time
import numpy as np
import pandas as pd
import datetime as dt

start_row = 0  # some file do not contain 100 lines, set this value to 0
neg_days = 720
pos_days = 7
all_days = neg_days + pos_days
resample_time = 10  # in seconds
resample_str = str(resample_time) + 'S'
resample_num = 24 * 60 * 60 / resample_time

# in_root_dir = '/mnt/disks/data'
in_root_dir = '/Users/zhangrao/PycharmProjects/341preprocess/data/iotdata'
# in_dirs = ["BG_Data_Part2", "BG_Data_Part3", "BG_Data_Part5", "BG-Data_Part11", "BG-Data_Part12"]
# in_dirs = ["BG-Data_Part11"]
# in_dirs = ["BG_Data_Part2"]
# in_dirs = ["BG_Data_Part95"]
in_dirs = ["BG_Data_Part911"]

# out_root_dir = '/mnt/disks/filldata'
out_root_dir = '/Users/zhangrao/PycharmProjects/341preprocess/filldata'

attributes_str = '''
Date
Time
System_time
ECU-Type
ECU_SW-Version
ECU_Identification
Code_plug_number
Nominal_(maximum)_burner_power
Minimal_Burner_load
Hot_water_system
Power_setpoint
Actual_Power
Current_fault_display_code
Current_fault_cause_code
Number_of_burner_starts
Working_time_total_system
Working_time_total_of_the_burner
Working_time_central_heating
Working_time_DHW
ClipIn_FW_Type
ClipIn_Bosch_FW_mainversion
ClipIn_Bosch_FW_subversion
Operating_status:_Central_heating_active
Operating_status:_Hot_water_active
Operating_status:_Chimmney_sweeper_active
Operating_status:_Flame
Operating_status:_Heatup_phase
Operating_status:_Error_Locking
Operating_status:_Error_Blocking
Operating_status:_Maintenance_request
Anti_fast_cycle_time_DT25
PWM_pump_present
Fluegas_sensor_present
Pressure_sensor_present
Return_sensor_present
Relay_status:_Gasvalve
Relay_status:_Fan
Relay_status:_Ignition
Relay_status:_CH_pump
Relay_status:_internal_3-way-valve
Relay_status:_HW_circulation_pump
External_cut_off_switch
Safety_Temperature_Limiter_MAX
RTH_switch_(external_on/off_control)
Outdoor_temperature
Number_of_starts_central_heating
Supply_temperature_(primary_flow_temperature)_setpoint
Supply_temperature_(primary_flow_temperature)
CH_pump_modulation
Low_loss_header_temperature
(Primary)_Return_temperature
Central_heating_blocked
Floor_drying_mode_active
System_water_pressure
Programmer_channel_for_central_heating_active
Central_heating_switch_on/off
Maximum_supply_(primary_flow)_temperature
Maximum_central_heating_power
Supply_temp__Pos._tolerance
Supply_temp__Neg._tolerance
Anti_fast_cycle_time_DT22
Pump_functionality_switch
Pump_post_purge_time
Pump_head_selection
Heat_request_status:_Central_heating_via_EMS-bus
Heat_request_status:_Central_heating_via_Switch
Heat_request_status:_Central_heating_Frost
Heat_request_status:_Hot_water_Frost
Heat_request_status:_Hot_water_detection_internal
Heat_request_status:_Hot_water_detection_via_EMS-bus
Number_of_starts_hot_water
Hot_water_temperature_setpoint
Hot_water_outlet_temperature
Hot_water_storage_temperature
Hot_water_flow_sensor_(turbine)
Programmer_channel_for_hot_water_active
Hot_water_installed_at_appliance
Hot_water_switch
Hot_water_supply_temperature_offset
Hot_water_circulation_pump
Circulation_pump_starts_per_Hour
Thermal_disinfection_setpoint
Diverter_valve_or_Chargepump
DHW_priority
Hot_water_day_function
Hot_water_one_time_loading
Hot_water_thermal_disinfection
Hot_water_system_is_being_heated
Hot_water_system_is_being_post_heated
Hot_water_setpoint_is_reached
Hot_water_priority_status
Error_status_byte_DHW_Hot_water_sensor_1_is_defect
Error_status_byte_DHW_Thermal_Disinfection_did_not_work
Service_Request_Setting
Service_after_burner_operating_time
Service_after_date:_Day
Service_after_date:_Month
Service_after_date:_Year
Service_after_appliance_operating_time
Voltage_measured_1-2-4_connection
Temperature_combustion_chamber
Used_fan_map
Actual_flow_rate_turbine
External_frost_thermostat_230Vac
On_Off_HW_demand_230Vac
On_Off_room_thermostat_230Vac_is_Y_S_compliant
Flame_current
Fan_speed
Fan_speed_setpoint
'''

columns_str = '''
Operating_status:_Error_Locking
Date
Time
Actual_Power
Number_of_burner_starts
Operating_status:_Central_heating_active
Operating_status:_Hot_water_active
Operating_status:_Flame
Relay_status:_Gasvalve
Relay_status:_Fan
Relay_status:_Ignition
Relay_status:_CH_pump
Relay_status:_internal_3-way-valve
Relay_status:_HW_circulation_pump
Supply_temperature_(primary_flow_temperature)_setpoint
Supply_temperature_(primary_flow_temperature)
CH_pump_modulation
Maximum_supply_(primary_flow)_temperature
Hot_water_temperature_setpoint
Hot_water_outlet_temperature
Actual_flow_rate_turbine
Fan_speed
'''


# ---- Above is global variable ----
# ---- Below is program ------------


def read_file(data_root_dir, data_dirs):
    all_files = []
    for one_dir in data_dirs:
        for one_file in os.listdir(os.path.join(data_root_dir, one_dir)):
            if one_file.endswith('.csv') and not one_file.startswith('sim'):
                file_name = os.path.join(data_root_dir, one_dir, one_file)
                if os.stat(file_name).st_size == 0:  # check empty file without headers
                    continue
                else:
                    all_files.append(file_name)
    return all_files


def save_npy(root_name, label_name, heater_name, date_str, var):
    root_dir = root_name
    sub_dir = label_name
    dirs = os.path.join(root_dir, sub_dir, heater_name.split('.')[0])
    dir_exists = os.path.exists(dirs)
    if not dir_exists:
        os.makedirs(dirs)
    file_name = dirs + '/' + date_str
    file_exists = os.path.isfile(file_name)
    if not file_exists:
        np.save(file_name, var)


def save_npz(root_name, label_name, heater_name, var):
    '''
    root_dir = root_name
    sub_dir = label_name
    dirs = os.path.join(root_dir, sub_dir)
    dir_exists = os.path.exists(dirs)
    if not dir_exists:
        os.makedirs(dirs)
    file_name = dirs + '/' + heater_name
    file_exists = os.path.isfile(file_name)

    if not file_exists:
        np.savez_compressed(file_name, var)
    '''
    np.savez_compressed("test_fill", *var)


def fill_file(file_name, save_type='y'):
    # save type check
    if save_type not in ('y', 'z'):
        raise ValueError('save_type can only be set for \'y\' or \'z\'')

    # build data type dict
    type_dict = {}
    attributes = attributes_str.strip().split('\n')
    for attribute in attributes:
        if attribute == 'Date' or attribute == 'Time':
            type_dict[attribute] = object
        else:
            type_dict[attribute] = np.float32

    # read in file
    df = pd.read_csv(file_name, sep='\t', dtype=type_dict)

    # fill data
    df.fillna({'Operating_status:_Error_Locking': 0}, inplace=True)  # 1. fill label with 0
    df.fillna(method='ffill', inplace=True)  # 2. forward fill values
    df.fillna(method='bfill', inplace=True)  # 3. backward fill to start row
    df.fillna(float(0), inplace=True)  # 4. fill empty columns

    # select columns
    columns = columns_str.strip().split('\n')
    df1 = df.loc[start_row:, columns]

    # extract date and time to form datetime
    datetime_str = df1['Date'] + ' ' + df1['Time']
    datetime = pd.to_datetime(datetime_str, format="%d.%m.%Y %H:%M:%S", errors='coerce')

    # drop illegal Date & Time
    df1.dropna(axis=0, how='any', inplace=True)

    # set date
    df1['Time'] = datetime
    df1['Date'] = datetime.dt.date

    # # save to a new csv file
    # cvs_to_save = os.path.join(in_root_dir, 'filled_' + os.path.basename(file_name))
    # cvs_exists = os.path.isfile(cvs_to_save)
    # if not cvs_exists:
    #     df1.to_csv(cvs_to_save, index=False)

    # extract labels and group by dates
    dfa1 = df1[['Date', 'Operating_status:_Error_Locking']].groupby('Date').sum().reset_index()
    dfa2 = (dfa1[dfa1['Operating_status:_Error_Locking'] > 0])['Date']  # only index, date and label > 0

    print dfa2

    # heater_name
    heater_name = (os.path.basename(file_name)).split('.')[0]  # for saving file

    # store the heater
    frames = []

    if not dfa2.empty:
        for idx in dfa2.index:
            idx_date = dfa2[idx]
            for day in xrange(all_days):
                prev_idx = idx - (day + 2)
                curr_idx = idx - (day + 1)
                if prev_idx >= 0 and dfa1.at[
                    prev_idx, 'Operating_status:_Error_Locking'] == 0:  # 1. guarantee to exist and not error date
                    curr_date = dfa1.at[curr_idx, 'Date']
                    if curr_date >= idx_date - dt.timedelta(days=pos_days) and dfa1.at[
                        curr_idx, 'Operating_status:_Error_Locking'] == 0:  # 2. guarantee to be within range
                        # join to select date
                        dfa3 = df1[df1['Date'] == curr_date].copy()

                        # stretch start and end, deal with only 1 sample / day
                        start_sec = pd.to_datetime(curr_date)
                        end_sec = pd.to_datetime(curr_date + pd.DateOffset(hours=23, minutes=59, seconds=59))
                        dfa3.set_value(dfa3.index[0], 'Time', start_sec)
                        if len(dfa3) == 1:
                            dfa3 = dfa3.append(dfa3, ignore_index=True)
                        dfa3.set_value(dfa3.index[-1], 'Time', end_sec)

                        # data processing
                        dfa3.drop_duplicates(subset='Time', keep='last', inplace=True)  # drop duplicates
                        dfa3.set_index('Time', inplace=True, verify_integrity=True)  # set index for resample
                        dfa4 = dfa3.resample(rule=resample_str).pad()  # resample cannot be done in-place
                        dfa4.reset_index(inplace=True)

                        # fill label columns:
                        dfa4['Operating_status:_Error_Locking'] = 1.0

                        # sanity check for re-sample number
                        if len(dfa4) != resample_num:
                            raise ValueError('re-sample number is inconsistent')

                        # save file
                        if save_type == 'y':
                            # drop Time and Data column
                            dfa4.drop(['Time', 'Date'], axis=1, inplace=True)       # drop time and date
                            x = dfa4.as_matrix()
                            date_str = curr_date.strftime('%Y-%m-%d')
                            save_npy(out_root_dir, 'pos_days', heater_name, date_str, x)
                        elif save_type == 'z':
                            frames.append(dfa4)

                    elif curr_date >= idx_date - dt.timedelta(days=all_days) and dfa1.at[
                        curr_idx, 'Operating_status:_Error_Locking'] == 0:
                        # join to select date
                        dfa3 = df1[df1['Date'] == curr_date].copy()

                        # stretch start and end, deal with only 1 sample / day
                        start_sec = pd.to_datetime(curr_date)
                        end_sec = pd.to_datetime(curr_date + pd.DateOffset(hours=23, minutes=59, seconds=59))
                        dfa3.set_value(dfa3.index[0], 'Time', start_sec)
                        if len(dfa3) == 1:
                            dfa3 = dfa3.append(dfa3, ignore_index=True)
                        dfa3.set_value(dfa3.index[-1], 'Time', end_sec)

                        # data processing
                        dfa3.drop_duplicates(subset='Time', keep='last', inplace=True)  # drop duplicates
                        dfa3.set_index('Time', inplace=True, verify_integrity=True)  # set index for resample
                        dfa4 = dfa3.resample(rule=resample_str).pad()  # resample cannot be done in-place
                        dfa4.reset_index(inplace=True)

                        # fill label columns:
                        dfa4['Operating_status:_Error_Locking'] = 0

                        # sanity check for re-sample number
                        if len(dfa4) != resample_num:
                            raise ValueError('re-sample number is inconsistent')

                        # save files
                        if save_type == 'y':
                            # drop Time and Data column
                            dfa4.drop(['Time', 'Date'], axis=1, inplace=True)       # drop time and date
                            x = dfa4.as_matrix()
                            date_str = curr_date.strftime('%Y-%m-%d')
                            save_npy(out_root_dir, 'neg_days', heater_name, date_str, x)
                        elif save_type == 'z':
                            frames.append(dfa4)

                else:
                    break
        if save_type == 'z' and frames:
            # sort by time
            result = pd.concat(frames, ignore_index='True')
            result.sort_values(by='Time', inplace=True)             # sort by time, not date

            # save npz files
            npz_array = []
            result_date = result.loc[:, 'Date'].copy()
            result_date.drop_duplicates(inplace=True)
            for ind in result_date.index:
                per_day = result[result['Date'] == result_date[ind]].copy()
                # drop Date column
                per_day.drop(['Date'], axis=1, inplace=True)        # drop date
                npz_array.append(per_day.as_matrix())
            save_npz(out_root_dir, 'pos_days', heater_name, npz_array)
        return len(dfa2)  # total error times
    else:  # all dates in this file are negative
        for idx in dfa1.index:
            # join to select date
            idx_date = dfa1.at[idx, 'Date']
            dfa3 = df1[df1['Date'] == idx_date].copy()

            # stretch start and end, deal with only 1 sample / day
            start_sec = pd.to_datetime(idx_date)
            end_sec = pd.to_datetime(idx_date + pd.DateOffset(hours=23, minutes=59, seconds=59))
            dfa3.set_value(dfa3.index[0], 'Time', start_sec)
            if len(dfa3) == 1:
                dfa3 = dfa3.append(dfa3, ignore_index=True)
            dfa3.set_value(dfa3.index[-1], 'Time', end_sec)

            # data processing
            dfa3.drop_duplicates(subset='Time', keep='last', inplace=True)  # drop duplicates
            dfa3.set_index('Time', inplace=True, verify_integrity=True)  # set index for resample
            dfa4 = dfa3.resample(rule=resample_str).pad()  # resample cannot be done in-place
            dfa4.reset_index(inplace=True)

            # fill label columns:
            dfa4['Operating_status:_Error_Locking'] = 0

            # sanity check for re-sample number
            if len(dfa4) != resample_num:
                raise ValueError('re-sample number is inconsistent')

            # save files
            if save_type == 'y':
                dfa4.drop(['Time', 'Date'], axis=1, inplace=True)   # drop date and time
                x = dfa4.as_matrix()
                date_str = idx_date.strftime('%Y-%m-%d')
                save_npy(out_root_dir, 'neg_days', heater_name, date_str, x)
            elif save_type == 'z':
                frames.append(dfa4)

        if save_type == 'z' and frames:
            # sort by time
            result = pd.concat(frames, ignore_index='True')
            result.sort_values(by='Time', inplace=True)             # sort by time, not date

            # save npz files
            npz_array = []
            result_date = result.loc[:, 'Date'].copy()
            result_date.drop_duplicates(inplace=True)
            for ind in result_date.index:
                per_day = result[result['Date'] == result_date[ind]].copy()

                # drop Date column
                per_day.drop(['Date'], axis=1, inplace=True)        # drop date
                npz_array.append(per_day.as_matrix())
            save_npz(out_root_dir, 'neg_days', heater_name, npz_array)

        return 0


if __name__ == '__main__':
    all_files = read_file(in_root_dir, in_dirs)
    file_cnt = 1
    tic_total = time.clock()

    # fill_file("../iotdata/BG-Data_Part11/2029717810162171906.csv", save_type='z')

    for one_file in all_files:
        print 'processing:', one_file
        print 'processing:', file_cnt, 'in', len(all_files), 'files'
        file_cnt += 1

        errors = fill_file(one_file, save_type='z')  # this is the function need to be called

        print 'processed', errors, 'error_locking are found in', os.path.basename(one_file)
        print ''
    toc_total = time.clock()
    print "[%ds] TASK done!" % (toc_total - tic_total)
