import os
import time
import readZoomVideo as z
import cv2
import glob
import datetime
import numpy as np
import pytesseract
import matplotlib.pyplot as plot
import imutils
import atexit
import ssocr

def combineCropped(list, dir="."):
    img = Image.open(dir+"/combinedCroppedPi.jpg")
    #for image in list, add it to the image
    print("combining images")
    for pic in list:
        input = imread(pic, im)
        img.Paste(im, img.size[1], img.size[0])
    
        

def getDigit( input_image, x1, x2, y1, y2, contours = False):
    # Load the 7x5 character templates for digits 0-9
    template = cv2.imread(f'digit_templates/7by5-digits_modified.jpeg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test', template )
    templates = []
    h, w  = template.shape
    _, binary = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY)

# Invert the binary image
    inverted_binary = binary#cv2.bitwise_not(binary)
    step = float(w-17)/10.
    intercept = 5
    numerals = []
    for i in range (10):
        
     #   print( int( (w*i) /10.),"  ", int( ((i+1) *w) /10.))
        digit = cv2.imread(f"digit_templates/Hugh/"+str(i)+".jpeg", cv2.IMREAD_GRAYSCALE)
        digit = cv2.adaptiveThreshold( digit,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -2) # adaptive thresholding
      #  digit = cv2.threshold(digit, 128, 255, cv2.THRESH_BINARY)
    #    print(digit)
        templates.append( digit)#inverted_binary[  15:h-15,  max(0, int( (step*float(i) + intercept)))  : int( (step*float(i+1) + intercept) ) ] ) #int( ((w+1) *i) /10.) ] )# int( (w*i) /10.) : int( ((w+1)*i )/10) ]  )
    #    print(templates[i])
        #extract contours
      #  cnts = cv2.findContours(template.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
      #  cnts = imutils.grab_contours(cnts)
      #  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
       # numerals.append(cnts[0])
        #cv2.drawContours(templates[i], cnts, -1, (128, 0, 0), 4)
        cv2.imshow('digit'+str(i), templates[i])
    # Load your input image

    # Initialize a list to store the matching results
    matches = []
    #scale input image to match templates
    cropped_image = input_image[y1:y2, x1:x2]
    # Loop through the templates and perform convolution
    for template in templates:
        scaled_image = cv2.resize(cropped_image, (template.shape[1], template.shape[0]))
        result = cv2.matchTemplate(scaled_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        matches.append(max_val)
   #     print(max_val)

    if contours:
        matches = []
        cnts = cv2.findContours(input_image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cnt = cnts[0]
        for numeral in numerals:
            ret = cv2.matchShapes(numeral,cnt,1,0.0)
            print( ret )
            matches.append(ret)
    
    # Find the index of the best matching template
    best_match_index = np.argmax(matches)

    # set a threshold to determine if a match is valid
    threshold = 0.35
 #   print(matches[best_match_index])
    if matches[best_match_index] >= threshold:
        recognized_digit = best_match_index
      #  print ("Look at me, I did it!")
    else:
        recognized_digit = None

    # Print the recognized digit (or None if no valid match)
 #   print("Recognized Digit:", recognized_digit)
    
    return recognized_digit




######## main functionality ####################

#run over a directory of images

today = str(datetime.date.today())

stamp = time.time()
filename = "currentReadings"+today+".csv"
file = open(filename, 'w')
atexit.register(file.close)

currents = []
times = []
dir ="/Users/patfreeman/Desktop/Pi_captures/Hugh2/*jpg"
outDir ="/Users/patfreeman/Desktop/Pi_captures/Hugh2/cropped/"
pics =sorted(glob.glob(dir))
x1 = 692#1430 - 4
x2 = 1572
x22 = 831#1615 +4
y1 = 128#388 - 1
y2 = 440
y22 = 157 #430
for pic in pics:
    stamp = pic.split("/")[-1].split("_")[1].split(".")[0]
    if int(stamp) > 1700181337:
        x1 = 694#1430 - 4
        x2 = 1575
        x22 = 827#1615 +4
        y1 = 123+1#388 - 1
        y2 = 440
        y22 = 154 #430
        print("woo")
    #print(stamp)
    im = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    roi = im[y1:y22, x1:x22] #crop
    cv2.imshow('cropped', roi)#
    rot =  cv2.rotate(roi, cv2.ROTATE_180)#roate 180 deg
  #  rect = np.array([[24,230], [1253,50], [1227,449], [11,569] ])
    angle = 2
# Calculate the rotation matrix
    height, width = rot.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
# Apply the rotation to the imageangle = 40
# Calculate the rotation matrix
    rotated_image = cv2.warpAffine(rot, rotation_matrix, (width, height))

# Apply the rotation to the image
    warped = rotated_image#z.four_point_transform(rot, rect)# correct for skew
    gaussian_3 = cv2.GaussianBlur(warped, (7, 7), 4.0)
    unsharp_image = warped#cv2.addWeighted(warped, 2.0, gaussian_3, -1.0, 0)
    cv2.imshow('sharpened', unsharp_image)
    
 #   rect = np.array([[24,230], [1253,50], [1227,449], [11,569] ])
    binary = cv2.adaptiveThreshold( unsharp_image,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -5) # adaptive thresholding
    
    #save binary image to file
    cv2.imwrite(outDir+"/PiCropped_"+stamp+".jpg", binary)
    
    chars = []
    ytrim = 0
    h, w = binary.shape
    nChars = 7
#    w = w2 - (x22 - x2)
 #   dot = 10
    chars.append(binary [0 +  ytrim: h-  ytrim, 0:int(w/nChars) ]  )
    chars.append(binary [0 +  ytrim: h-  ytrim, int(w/nChars) : 2*int(w/nChars) ]  )
    chars.append(binary [0 +  ytrim: h-  ytrim,2*int(w/nChars) : 3*int(w/nChars) ]  )
    chars.append(binary [0 +  ytrim: h-  ytrim, 3*int(w/nChars) : int(4* w/nChars) ]  )
    chars.append(binary [0 +  ytrim: h-  ytrim, 4*int(w/nChars) : 5*int(w/nChars) ]  )
    chars.append(binary [0 +  ytrim: h-  ytrim, 5*int(w/nChars) : 6*int(w/nChars)  ]  )
    chars.append(binary [0 +  ytrim: h-  ytrim, 6*int(w/nChars) : int(w)  ]  )
#    chars.append(binary [0 +  ytrim: h-  ytrim, 7*int(w/nChars) : w  ]  )
  #  chars.append(binary [0 +  ytrim: h-  ytrim, w +dot : w2 ]  )

    reading =""
    treading=""
    noneFlag = 0
    i=0
    config = ' -c tessedit_char_whitelist=0123456789 --psm 7 --psm 10'
   # extracted_text = pytesseract.image_to_string(binary, config=config)
    for char in chars:
        #extracted_text = pytesseract.image_to_string(char, config=config)
        xx1=0
        yy1=0
        xx2 = char.shape[1]
        yy2 = char.shape[0]
        if i == 1:
            extracted_text = "."
        else:
            extracted_text = getDigit(char, xx1, xx2, yy1, yy2, False) #pytesseract.image_to_string(char, config=config) #getDigit(xx1, xx2, yy1, yy2, True)
        tesseracted_text =  pytesseract.image_to_string(char, config=config)
       
        reading+=str(extracted_text)
        treading+=str(tesseracted_text)
      #  print("tess: "+tesseracted_text)
       
            #reading+="."
        cv2.imshow('char'+str(i), char )
        if extracted_text is None:
            noneFlag = 1
            i+=1
            continue
        i+=1
    reading = ssocr.main( cv2.bitwise_not(binary) )

    print("reading: "+str(reading))
   # print("treading: "+str(treading))
    config = ' -c tessedit_char_whitelist=0123456789 --psm 7 '
    extracted_text = pytesseract.image_to_string(binary, config=config)
    cv2.putText(im, str(reading), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.imshow('binary', binary)#
    cv2.imshow('input image', im)#
    times.append(times)
  #  current = getReading(pic)
    current = 0#int(extracted_text)
    print(current)
    file.write(str(stamp)+", " +  reading +"\n")
    file.flush()
    currents.append(current)
 #   if cv2.waitKey(100) & 0xFF == ord('q'):
  #      break

list = glob.glob(outDir+"/*.jpg")
combineCropped(list)
    #chars

file.close()

#plot.plot(times, currents)

#take images and process them 1-by-1, fill a .csv with timestamps and current readings
'''
with  open(file, 'w') as f:
    stamp = time.time()
    
    pic = "image_"+str(stamp)+".jpg"
    cmd = "raspistill -n -t 1000 - tl 1000 -dt -o " + pic
    os.system(cmd)
    f.write(stamp)
    f.write(",")
    current = getReading(pic)
    f.write(str(stamp)+","+str(current)+"\n")
'''



