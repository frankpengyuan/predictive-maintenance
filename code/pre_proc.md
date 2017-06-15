read csv file by heater

- data type: float32 to accelarate parallelity
- select 19 sensors, date, time and error_locking

fill the data
- fill error_locking column domain with 0
- forward fill
- backward fill
- fill blank column with 0

deal with time
- extract Date and Time
- drop time rows with wrong conversion
- construct Python datetime and overwrite Time



__at this point, denote the data as df1__



deal with label

- group data by 'Date'
- winnow error_locking date which label > 0
- check whether the previous 7 days are positive
- check whether the previous >= 8 days are negative
- continues going back until the file start or previous error_locking
- skip the date after the error_locking day and first day
- set label positive to be 1, negative to be 0

extract data by date
- join the selected date with df1
- drop duplicate time point in one day
- set the first tuple of this day to 00:00:00
- set the last tuple of this day to 23:59:59
- resample the data to align each day



__if there is no error, all the data in this file are saved__



store the data by date

- drop useless 'Date' column
- save file with date as its name
- save pos and neg into different folders
- save different heaters' data into different folders
- save different days' data into npy files (one file -> one day/one heater)
- save npz file in sorted order by 'Time'
- save npz file with all they days for the heater (one file -> one heater)


optimization

- in-place update data if possible
- reduce deep-copy to save memory










test

程序大概分为两部分

- 填充空白数据
- 筛选pos和neg的数据



填充空白数据

- 读取csv文件
- 填充数据
  - label项所有都填充0 （'Operating_status:_Error_Locking'）
  - 首先forward填充数据
  - 其余的backward填充数据
  - 剩下整列为空的直接填0
- 提取目标列
- 时间日期转换
  - 转换为date、time格式
  - 转换失败的成为NaN，直接drop掉
- 保存处理后的csv文件



筛选pos和neg数据项

- 提取Date和Label两列
  - 根据Date进行group
  - 选择包含Label 1的Date
- 检查该Date是否有效
  - 保证之前有14天数据
  - 数据列14项之前的日期应该为14天前（保证之前日期没有跳跃）
  - 当前为1，之前的14天都为0
- 写入文件
  - 7天为neg sample，Label为0不变
  - 7天为pos sample，Label全部置为1
  - 保存成npy格式
