import datetime
import os
import time
today = str(datetime.date.today())

#filename = "currentReadings"+today+".csv"
dir = "./Pi_captures/"

#f = open(filename, 'w')
while True:
    stamp = int(time.time())
    pic = dir+"image_"+str(stamp)+".jpg"
    cmd = "raspistill -n -t 100 -tl 0 -dt -o " + pic
    print (cmd)
    os.system(cmd)
    #current = -99999#getReading(pic)
    #f.write(str(stamp)+","+str(current)+"\n")

