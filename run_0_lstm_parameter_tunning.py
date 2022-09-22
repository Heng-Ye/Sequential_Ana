import os


for i in range(9):
  num=300+(i+1)*10
  print('num:',num)
  exe='python dl_0_LSTM_parameter_tunning.py '+str(num)
  os.system(exe)

