import numpy as np
import os

file_name = 'antiderivative_unaligned_train'
data = np.load(f'./antiderivative/{file_name}.npz')
# make a folder in './csvdata' directory
PATH = f'./csvdata/{file_name}'
# if exists, skip
if not os.path.exists(PATH):
  os.mkdir(PATH)

npz_file = {}
for key in data.keys():
  if key == 'X_train1':
    my_array = data[key]
    # expand the dimension of the array by concatenating the array with itself
    # my_array = np.concatenate((my_array, my_array, my_array), axis=1)
    np.savetxt(f'{PATH}/{key}.csv', my_array, delimiter=',')
    npz_file[key] = my_array
  elif key == 'X_train0':
    my_array = data[key]
    # expand the dimension ten-times of the array by concatenating the array with itself
    # my_array = np.concatenate((my_array, my_array, my_array, my_array, my_array, my_array, my_array, my_array, my_array, my_array), axis=1)
    np.savetxt(f'{PATH}/{key}.csv', my_array, delimiter=',')
    npz_file[key] = my_array
  else:
    my_array = data[key]
    np.savetxt(f'{PATH}/{key}.csv', my_array, delimiter=',')
    npz_file[key] = my_array

print(npz_file)
np.savez(f'{PATH}/{file_name}.npz', **npz_file)
