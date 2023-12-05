import datetime
import os
import time
import argparse
today = str(datetime.date.today())

#parser = argparse.parser()
#args = arg

#filename = "currentReadings"+today+".csv"
dir = "~/Pi_captures/"

#f = open(filename, 'w')
while True:
    if os.path.exists("~/Desktop/.stop"):
        print("stop file detected, stopping takePiPics.py")
        os.remove("~/Desktop/.stop")
        break
    stamp = int(time.time())
    pic = dir+"image_"+str(stamp)+".jpg"
    cmd = "raspistill -n -t 1000  -dt -o " + pic
    print (cmd)
    os.system(cmd)
    copyCmd = "rsync "+dir+"*.jpg pmfreeman@tau.physics.ucsb.edu:/net/cms26/cms26r0/pmfreeman/Pi_captures/ --r>
    os.system(copyCmd)
    time.sleep(100)
    #current = -99999#getReading(pic)
    #f.write(str(stamp)+","+str(current)+"\n")
