import glob
import shutil

short_list = glob.glob("/darkdata/Sony/short/*.ARW")
long_list = glob.glob("/darkdata/Sony/long/*.ARW")

short_list.sort()
long_list.sort()

dest = "Sony_test/"

n = 20

for i in range(n):
    shutil.copy(long_list[i],dest+"long")
    
for i in range(n*12+1):
    shutil.copy(short_list[i],dest+"short")
    
