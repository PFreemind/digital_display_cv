import cv2
import matplotlib.pyplot as plot
import pytesseract
import numpy as np
import imutils
import ssocr
import glob

# import the necessary packages
import numpy as np
import cv2
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect
#for image rotation
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def getChar(img, minArea = 200): #crop a character exactly around its borders for aligned comparison
    cnts = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#    cv2.drawContours(img, cnts, -1, (128, 0, 0), 4)
        
    # create a combined bounding box
    ymin = 1e5
    xmin = 1e5
    xmax = 0
    ymax = 0
    for cnt in cnts:
        if cv2.contourArea(cnt) < minArea : #stop once the contours are too small
            break
        [x,y,w,h ] = cv2.boundingRect(cnt)
        if x < xmin: xmin = x
        if y < ymin: ymin = y
        if x + w > xmax : xmax = x + w
        if y + h > ymax: ymax = y + h

    #new char coordinates are xmin, ymin, xmax, ymax
    # return the cropped char
    try:
        char = img [ymin:ymax, xmin:xmax]
    except:
        print("could not extract char (likely due to contour area threshold), returning orignal image")
        char = img
    return char
    
def getDigit(input_image, x1, x2, y1, y2):
    # Load the 7-segment character templates for digits 0-9
    template = cv2.imread(f'digit_templates/lcd-7segment-digits.jpeg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test', template )
    templates = []
    h, w  = template.shape
    _, binary = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY)

# Invert the binary image
    inverted_binary = cv2.bitwise_not(binary)
    step = float(w-17)/10.
    intercept = 5

    for i in range (10):
        im = cv2.imread("digit_templates/dirpi19/"+str(i)+".jpg" ,   cv2.IMREAD_GRAYSCALE )
        binary = cv2.adaptiveThreshold( im, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31, -5)
        binary = getChar(binary)
        templates.append(binary)
     #   print( int( (w*i) /10.),"  ", int( ((i+1) *w) /10.))
        #templates.append( inverted_binary[  40:h-25,  max(0, int( (step*float(i) + intercept)))  : int( (step*float(i+1) + intercept) ) ] ) #int( ((w+1) *i) /10.) ] )# int( (w*i) /10.) : int( ((w+1)*i )/10) ]  )
    #    print(templates[i])
        cv2.imshow('digit'+str(i), templates[i])
    # Load your input image

    # Initialize a list to store the matching results
    matches = []
    #scale input image to match templates
    cropped_image = input_image[y1:y2, x1:x2]
    scaled_image = cv2.resize(cropped_image, (templates[0].shape[1], templates[0].shape[0]))
    cv2.imshow('scaled', scaled_image)
    # Loop through the templates and perform convolution
    for template in templates:
        scaled_image = cv2.resize(cropped_image, (template.shape[1], template.shape[0]))
        result = cv2.matchTemplate(scaled_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        matches.append(max_val)

    # Find the index of the best matching template
    best_match_index = np.argmax(matches)

    # You can set a threshold to determine if a match is valid
    threshold = 0.45
 #   print(matches[best_match_index])
    if matches[best_match_index] >= threshold:
        recognized_digit = best_match_index
      #  print ("Look at me, I did it!")
    else:
        recognized_digit = None

    # Print the recognized digit (or None if no valid match)
 #   print("Recognized Digit:", recognized_digit)
    
    return recognized_digit
    
    

# Path to the Tesseract OCR executable (change this to match your installation)
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='read still captured by RPi camera')
    parser.add_argument('-x1', '--x1', help='crop limit for 4 digits (TL)',type=int, default = 930)
    parser.add_argument('-x2', '--x2', help='crop limit (TR)',type=int, default = 1660)
    parser.add_argument('-x3', '--x3', help='crop limit (BR)',type=int, default = 1657)
    parser.add_argument('-x4', '--x4', help='crop limit (BL)',type=int, default = 918)
    parser.add_argument('-y1', '--y1', help='crop limit (TL)',type=int, default = 1148)
    parser.add_argument('-y2', '--y2', help='crop limit (TR)',type=int, default = 999)
    parser.add_argument('-y3', '--y3', help='crop limit (BR)',type=int, default = 1206)
    parser.add_argument('-y4', '--y4', help='crop limit (BL)',type=int, default = 1355)
    parser.add_argument('-d', '--dirpi', help='ID number of dirpi device',type=int, default = 19)
    parser.add_argument('-s', '--show',  action ='store_true', help = 'bool for showing images as they are processeed' )
    parser.add_argument('-a', '--angle', help='angle of rotation correction in degrees', type=float, default = 0)
    parser.add_argument('-i', '--input', help='input directory',type=str, default="/Users/patfreeman/Desktop/Pi_captures/dirpi19/" )
    parser.add_argument('-o', '--output', help='output directory',type=str, default="/Users/patfreeman/Desktop/Pi_captures/dirpi19/cropped/" )
    parser.add_argument('-t', '--throttle', type = int, default ='100',  help='throttle the video reading, 1/throttle frames of the video are processed'  )

    args = parser.parse_args()
    throttle = args.throttle
    x1 = args.x1   # crop window limits
    x2 = args.x2
    x3 = args.x3   # crop window limits
    x4 = args.x4
    y1 = args.y1
    y2 = args.y2
    y3 = args.y3
    y4 = args.y4
    show = args.show
    dirpi = args.dirpi
    dir = args.input
    outDir = args.output
    # Define the coordinates of the ROI (top-left and bottom-right)
    roi_x1, roi_y1, roi_x2, roi_y2 = 0,0,4000,4000#1000, 500, 2400, 1400  # Adjust these coordinates as needed

    # Initialize the video capture object

    frameOfCameraBump = 38250

    # Get the frame rate of the video
    # Calculate the frame index to start from

    # Set the frame index to the calculated starting frame
    iFrame = 0
    readings = []
    filename = "dirpiCameraReadings_dirpi"+str(dirpi)+".txt"
    files = glob.glob(dir+"*.jpg")
    with open(filename, 'w') as output:
        for f in files:
            frame = cv2.imread(f)
            cv2.imshow("original", frame)
            if iFrame % throttle != 0:
                iFrame+=1
                continue
            if iFrame%100 == 0:
                print("reading image number "+str(iFrame))
                print("reading image file "+str(f))
            # Read a frame from the video
            # Check if we have reached the end of the video
            #image pre-processing
            # Extract the ROI from the frame
            time = int( f.split("/")[-1].split("_")[-1].split(".")[0] ) #float(iFrame)/float(frame_rate) + skip_start_time
           # roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            # Convert the ROI to grayscale for better OCR accuracy
            gray_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('gray', gray_roi)

            # ret,thresh = cv2.threshold(gray_roi,70,255,0)
            rect = np.zeros((4, 2), dtype = "float32")
            rect = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4] ])
            warped = gray_roi[y1:y3, x4:x2 ] # four_point_transform(gray_roi, rect)
            cv2.imshow('warped', warped)
            if dirpi == 17:
                warped =  cv2.rotate(warped, cv2.ROTATE_180)#roate 180 deg since image upside-down
            cv2.imshow('rotated', warped)
            '''
            if iFrame >  frameOfCameraBump:
                xBump = 30
                yBump = -20
                smidge = 8
                rect = np.array([[24 + xBump - smidge, 230 + yBump], [1253 + xBump, 50 + yBump - smidge], [1227+ xBump, 449 + yBump - smidge], [11 + xBump, 569 + yBump] ])
            '''
            #could add noise reduction with fastNlMeansDenoisingColored ()
            #also, smarter edge detection?
           # foo,binary=cv2.threshold(warped, 160, 255, cv2.THRESH_BINARY)
            blurred = cv2.GaussianBlur(warped, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 200, 255)

            gaussian_3 = cv2.GaussianBlur(warped, (51, 51), 4.0)
            unsharp_image = cv2.addWeighted(warped, 2.0, gaussian_3, -1.0, 0)
            cv2.imshow('sharpened', unsharp_image)
            binary = cv2.adaptiveThreshold( warped,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31, -4)
            if dirpi == 17:
                    binary = cv2.adaptiveThreshold( unsharp_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,31, -2)
            cnts = cv2.findContours(binary.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            cv2.drawContours(binary, cnts, -1, (128, 0, 0), 4)
            cv2.imshow('binary', binary)#
            cv2.imwrite(outDir+"/PiCropped_"+str(time)+".jpg", binary)
            # only keep the largest contours, remove the little blobs
            # get alignment better, using contours?
            # use joe's code
        #    cv2.imshow('foo',binary)
            #plot.imshow(gray_roi)
            # Use Tesseract OCR to extract text from the ROI
    #        extracted_text = pytesseract.image_to_string(gray_roi)
        #split the frame into 4 characters
            
            h, w = binary.shape
            chars = []
            offset = 0
            y0shift = 0
            x0shift = 0
            y1shift = 0

            
            if iFrame >  frameOfCameraBump:
                xBump = 30
                yBump = 30
                chars.append( binary[ offset + yBump -y0shift :h - offset*2 - y0shift+ yBump, 0+ offset + xBump- x0shift :int(w/4 - 60 ) + xBump  + x0shift ] )
                chars.append( binary[ offset+ yBump - y1shift:h - offset*2+ yBump - y1shift, int(w/4)+ offset + xBump :int(w/2) - offset*2 + xBump ] )
                chars.append( binary[ offset+ yBump:h - offset*2+ yBump, int(w/2)+ offset + xBump :int(3*w/4) -offset*2 + xBump ] )
                chars.append( binary[ offset+ yBump:h - offset*2+ yBump, int(3*w/4)+ offset + xBump :w - offset*2 + xBump ] )
            else:
                chars.append( binary[ offset:h - offset*2, 0+ offset :int(w/4)  ] )
                chars.append( binary[ offset:h - offset*2, int(w/4)+ offset:int(w/2) - offset*2] )
                chars.append( binary[ offset:h - offset*2, int(w/2)+ offset:int(3*w/4) -offset*2] )
                chars.append( binary[ offset:h - offset*2, int(3*w/4)+ offset:w - offset*2] )
           
            shift = 4
            '''
            for j in range( min(4, len(cnts)) ):
                [x,y,w,l ] = cv2.boundingRect(cnts[j])
                
                chars.append( binary[ y  : y + l , x  : x + w ] )
            '''
            i=-1
        
            config = ' --psm 7 -c tessedit_char_whitelist=0123456789 '
            reading =""
            noneFlag = 0
            for char in chars:
                i+=1
            #    extracted_text = pytesseract.image_to_string(char, config=config)
                char = getChar(char)
                xx1=0
                yy1=0
                xx2 = char.shape[1]
                yy2 = char.shape[0]
                extracted_text = getDigit(char, xx1, xx2, yy1, yy2)
                if i < 4:
                    reading+=str(extracted_text)
                if i == 0:
                    reading+="."
                cv2.imshow('char'+str(i), char )
                readings.append(reading )
                if extracted_text is None:
                    noneFlag = 1
                    continue
           #     reading = ssocr.main( cv2.bitwise_not(binary) )

              #  cv2.imshow('char'+str(i), char )
                cv2.putText(char, str(extracted_text), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                cv2.imshow('char'+str(i), char )

                cv2.putText(binary, str(reading), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                cv2.imshow('binary', binary )
      #      extracted_text = pytesseract.image_to_string(binary, config=config)

            # You can add additional processing here to filter and extract numbers from 'extracted_text'
            if noneFlag == 0:
               # f.write(str(time)+", ")
                #f.write("test\n")
                #f.write(reading+"\n")
                #f.write("\n")
                output.write(str(time)+", ")
                output.write(reading+"\n")
                reading = float(reading)
                readings.append(reading )
            # Display the extracted text on the frame
            
         #   foo,gray_roi=cv2.threshold(roi_frame, 127, 255, cv2.THRESH_BINARY)
            # Display the frame with extracted text
            #cv2.accumulateWeighted(dst, binary, 0.1)
            h=binary.shape[0]
            w=binary.shape[1]
            dst=np.zeros([h, w], dtype=np.uint8)
            '''
            alpha =0.2
            for i in range(h):
                for j in range(w):
                    dst[i,j] = dst[i,j] * (1-alpha) + binary[i,j] * (alpha)
                    
            '''
            if show:
                input()
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            foo,out=cv2.threshold(binary, 1, 255, cv2.THRESH_BINARY)
           # out= cv2.adaptiveThreshold(dst,160,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            #dst = dst*(1-alpha) + binary*alpha
           # extracted_text = pytesseract.image_to_string(out, config=config)
            #cv2.putText(out, extracted_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0))
         #   cv2.imshow('Frame', out)
            # Break the loop if 'q' is pressed
           
           
            
            ##hmmmm
            #break image into 4 parts, check each letter individually
            #can write something specific for 7-segment letters
            #calculate mean in 7 roi, get pieces, match to mapping
            #id bad matches
            #1 =[0,1,1,0,0,0,0]
            #2=[1,1,0,1,1,0,1]
            #3=[1,1,1,1,0,0,1]
            #etc
            #really, same as the easrlier comment... removing short contours would help tho
            # can parse for numbers via contours
            #then check each number with nasty 7-segment explicit defintions. that's a decent amount of development...
            iFrame+=1

    # Release the video capture object and close all OpenCV windows
    cv2.destroyAllWindows()

