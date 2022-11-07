import numpy as np

# data_type = np.float32
#
# data_batch = np.fromfile("batch1.dat", dtype=data_type)
#
# print(data_batch[0])

# lon , lat = data.reshape(2,2288,2288)

import pandas as pd

data_batch_fname = ["batch1.dat", "batch2.dat", "batch3.dat", "batch4.dat", "batch5.dat",
                    "batch6.dat", "batch7.dat", "batch8.dat", "batch9.dat", "batch10.dat"]

# data_batch_fname = ["batch1.dat"]

x_buf = []
y_buf = []

for fname in data_batch_fname:

    data_batch = pd.read_csv(fname, header=None, encoding="utf-8", delimiter=" ")

    # print(data_batch.shape)
    # print(type(data_batch[1][0]))
    # print((data_batch[0][0]))

    row, col = data_batch.shape
    data_batch_float = np.zeros((row, col-1))
    label_batch_int = np.zeros((row,1)).astype(np.long)

    for index2 in range(col): # 129
        for index1 in range(row): # 445
            x = data_batch[index2][index1]
            if index2 == 0:
                label_batch_int[index1] = x-1
            else:
                strt = x.find(":") + 1
                data_batch_float[index1, index2-1] = float(x[strt:])

    x_buf.append(data_batch_float)
    y_buf.append(label_batch_int)
    # print(data_batch_float)
    # print(label_batch_int)
    print(fname + " finished!")

x_buf_array = np.concatenate(x_buf, axis=0)
y_buf_array = np.concatenate(y_buf, axis=0)
xy_buf_array = np.concatenate([x_buf_array, y_buf_array], axis=1)
print(x_buf_array.shape)
print(y_buf_array.shape)

header = [str(x) for x in range(x_buf_array.shape[1])] + ["target"]
xy_buf_dataframe = pd.DataFrame(xy_buf_array)

print(xy_buf_dataframe)

xy_buf_dataframe.to_csv("gas.csv", header=header, index=None)