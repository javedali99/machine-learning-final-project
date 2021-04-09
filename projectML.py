
#  Created by Amanda Barroso on 3/26/21.
#  
import scipy.io as sio
import matplotlib.pyplot as plt

matstruct_contents = sio.loadmat('WaterLevel_St_Pete_hourly.mat')

raw_waterlevel= matstruct_contents['WaterLevel']
time = matstruct_contents['Time']

plt.plot(time,raw_waterlevel)
plt.show()

detrended = sio.loadmat('hourlydt.mat')

dt_water = detrended['hourlydt']
plt.plot(time,dt_water)
plt.show()