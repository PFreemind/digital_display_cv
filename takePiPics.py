import datetime
import os
import time
import argparse
today = str(datetime.date.today())

#parser = argparse.parser()
#args = arg

parser = argparse.ArgumentParser(description='read still captured by RPi camera')
#parser.add_argument('-o', '--output', help='output directory',type=str, default="/net/cms26/cms26r0/pmfreeman/Pi_captures/" )
parser.add_argument('-d', '--delay', help='delay time between captures',type=int, default = 1)
parser.add_argument('-r', '--rsync', action ='store_true', help = 'bool for running rsync (no rsyncing if not set)' )
parser.add_argument('-o', '--output', help='output directory',type=str, default="~/Pi_captures/" )
args = parser.parse_args()
delay = args.delay
output  = args.output
rsync = args.rsync
#filename = "currentReadings"+today+".csv"
IDfile  = "/home/dirpi19/dirpi/metadata/ID.txt"
try:
    with open(IDfile, 'r') as file:
        first_line = file.readline()
        if first_line:
            ID = str(first_line.split("\n")[0])
            print("the dirpi ID is : "+ID)
        else:
            print("The ID file is empty.")
except FileNotFoundError:
    print(f"File not found: {IDfile}")
except Exception as e:
    print(f"An error occurred: {e}")

#f = open(filename, 'w')
while True:
    if os.path.exists("/home/dirpi"+str(ID)+"/digital_display_cv/.stop"):
        print("stop file detected, stopping takePiPics.py")
        os.remove("/home/dirpi"+str(ID)+"/digital_display_cv/.stop")
        break
    stamp = int(time.time())
    pic = output+"image_"+str(stamp)+".jpg"
    cmd = "raspistill -n -t 1 -rot 180 -roi 0.25,0.25,0.5,0.5 -o " + pic
    print (cmd)
    os.system(cmd)
    copyCmd = "rsync "+output+"*.jpg pmfreeman@tau.physics.ucsb.edu:/net/cms26/cms26r0/pmfreeman/Pi_captures/dirpi"+str(ID)+"/ --remove-source-files"
    if rsync:
        print(copyCmd)
        os.system(copyCmd)
    if delay > 0:
        time.sleep(delay)
    #current = -99999#getReading(pic)
    #f.write(str(stamp)+","+str(current)+"\n")
