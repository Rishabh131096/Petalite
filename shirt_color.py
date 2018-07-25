import cv2
import numpy as np
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Various color types for detected shirt colors.
#enum                             {cBLACK=0,cWHITE, cGREY, cRED, cORANGE, cYELLOW, cGREEN, cAQUA, cBLUE, cPURPLE, cPINK,  NUM_COLOR_TYPES};
color = ["Black", "White","Grey","Red","Orange","Yellow","Green","Aqua","Blue","Purple","Pink"]
color_values = [[0,0,0],[255,255,255],[128,128,128],[0,0,255],[0,128,255],[0,255,255],[0,255,0],[255,0,0],[153,0,153],[255,0,255]]

def draw_rectangle(img, rect,color=(0,255,0)):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
	
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)	
	
def dist(a,b):
	return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]))
	
def get_pixel_color_type(pix,img):

	V = pix[2]
	S = pix[1]
	H = pix[0]
	
	#print(str(H) + " " + str(S) + " " + str(V));
	#draw_text(img,str(H) + " " + str(S) + " " + str(V),40,40)
	
	if (V < 45):
		color = 0 #BLACK
	elif (V > 165 and S < 27):
		color = 1 #WHITE
	elif (S < 53 and V <= 165):
		color = 2 #GREY
	else :
		if (H < 14):
			color = 3 #RED
		elif (H < 25):
			color = 4 #ORANGE
		elif (H < 34):
			color = 5 #YELLOW
		elif (H < 73):
			color = 6 #GREEN
		elif (H < 102):
			color = 7 #AQUA
		elif (H < 127):
			color = 8 #BLUE
		elif (H < 149):
			color = 9 #PURPLE
		elif (H < 175):
			color = 10 #PINK
		else: 
			color = 3 #RED;	// back to Red
			
	return color;
	

	
	
def detect_shirt_color(f,img):
	initialConfidence = 1.0

	# Create the shirt region, to be below the detected face and of similar size.
	SHIRT_DY = 1.4	# Distance from top of face to top of shirt region, based on detected face height.
	SHIRT_SCALE_X = 0.6	# Width of shirt region compared to the detected face
	SHIRT_SCALE_Y = 0.6	# Height of shirt region compared to the detected face
	(x,y,w,h) = f
	#print("x="+str(x)+"y="+str(y)+"w="+str(w)+"h="+str(h))

	X = x + (int)(0.5 * (1.0-SHIRT_SCALE_X) * w)
	Y = y + (int)(SHIRT_DY * h) + (int)(0.5 * (1.0-SHIRT_SCALE_Y) * h)
	W = (int)(SHIRT_SCALE_X * w)
	H = (int)(SHIRT_SCALE_Y * h)
	
	#print("updated X="+str(X)+"Y="+str(Y)+"W="+str(W)+"H="+str(H))
	draw_rectangle(img,(x,y,w,h),(255,0,0))
	bottom = Y+H-1
	#print("bottom= "+str(bottom))
	imgh=img.shape[0]
	imgw=img.shape[1]
	if(bottom > imgh - 1):
		SHIRT_DY = 0.95	# Distance from top of face to top of shirt region, based on detected face height.
		SHIRT_SCALE_Y = 0.3	# Height of shirt region compared to the detected face
		initialConfidence = initialConfidence * 0.5	# Since we are using a smaller region, we are less confident about the results now.
		Y = y + (int)(SHIRT_DY * h) + (int)(0.5 * (1.0-SHIRT_SCALE_Y) * h)
		H = (int)(SHIRT_SCALE_Y * h)
		#print("H in first if = "+str(H))
	
	#Try once again if it is partly below the image.
	bottom = Y+H-1;
	if(bottom > imgh-1): 
		bottom = imgh-1	# Limit the bottom
		#print("bottom = " + str(bottom) + " Y = " + str(Y))
		H = bottom - (Y-1)	# Adjust the height to use the new bottom
		#print("H in second if = "+str(H))
		initialConfidence = initialConfidence * 0.7	#Since we are using a smaller region, we are less confident about the results now.
		
	#print("H = "+str(H))

	# Make sure the shirt region is in the image
	if(H <= 1):
		draw_text(img,"Shirt not in Image",20,20)
	else:
		shirt = (X,Y,W,H)
		#taking image in hsv format
		shirt_img = img[Y:Y+H, X:X+W]
		shirt_img = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2HSV)
		draw_rectangle(img,shirt)
		tallyColors=[0,0,0,0,0,0,0,0,0,0,0]
		
		
		# Scan the shirt image to find the tally of pixel colors
		for j in range(0,H):
			for i in range(0,W):
				#Get the BGR pixel components

				#Determine what type of color the BGR pixel is.
				ctype = get_pixel_color_type(shirt_img[j][i],img)
				#Keep count of these colors.
				tallyColors[ctype]+=1


		# Print a report about color types, and find the max tally
		tallyMaxIndex = 0
		tallyMaxCount = -1
		pixels = W*H
		for i in range(0,len(color)):
			v = tallyColors[i]
			if (v > tallyMaxCount):
				tallyMaxCount = tallyColors[i]
				tallyMaxIndex = i
				
				
			#Display the color type over the shirt in the image.
		percentage = initialConfidence * (tallyMaxCount * 100 / pixels)
		#print("confidence "+str(percentage))
		draw_text(img,color[tallyMaxIndex]+" (" +str(percentage) +"% confidence).",20,20)
	
	return img


def detect_faces(test_img):
	img = test_img.copy()
	#########################################
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

	if (len(faces) == 0):
		return img
	if(len(faces)!=1):
		return img
	
	for f in faces:
		(x, y, w, h) = f
		gray_img = gray[y:y+w, x:x+h]
		rect = (x,y,w,h)
		#draw_rectangle(img, rect)
		img = detect_shirt_color(f,img);
		#print("loop")
	#####################################
	return img

cap = cv2.VideoCapture(0)

End_of_Video = False

while(1):
	
	ret, img = cap.read()
	if ret==False:
		End_of_Video = True
		break 
	
	img=cv2.flip(img,1)
	w=320
	h=480
	img = img[0:h, 0:w]
	predicted_img = detect_faces(img)
	
	cv2.imshow("VIDEO",predicted_img)
	
	k = cv2.waitKey(1)
	if k == 27:
		break


#img = cv2.imread("man.jpg",1)
#predicted_img = detect_faces(img)
#cv2.imshow("VIDEO",predicted_img)
#cv2.waitKey(0)
cv2.destroyAllWindows()
