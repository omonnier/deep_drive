1) Install Raspberry et Cars
	https://www.sunfounder.com/learn/category/Smart-Video-Car-for-Raspberry-Pi.html


2) Start VNC Session using Raspberry IP address on EmployeeHotspot
	User 	 : pi
	Password : rapberry
	
	You can retreive IP Address connecting display on Raspberry HDMI connector 
	
	Creer .bash_aliases et ajouter :
	alias si='cd /home/pi/deep_drive/raspberry/mjpg-streamer/mjpg-streamer;./start.sh &'
	alias st='cd /home/pi/deep_drive/raspberry/server;sudo python tcp_server.py'
	alias sc='cd /home/pi/deep_drive/raspberry/server;sudo python cali_server.py' 

	To change Display resolution :  
	Links :
		https://support.realvnc.com/Knowledgebase/Article/View/523/2/troubleshooting-vnc-server-on-the-raspberry-pi 
		https://www.raspberrypi.org/documentation/configuration/config-txt.md
		
	File to be modified :  /boot/config.txt
	
	sudo vi /boot/config.txt
		Uncomment hdmi_force_hotplug=1
		Uncomment hdmi_group and set it to 2 (DTM mode)
		Uncomment hdmi_mode and set 
			to 16 to force a resolution of 1024x768 at 60Hz
			to 23 to force a resolution of 1280x768	at 60Hz
			to 35 to force a resolution of 1280x1024 at 60Hz
	
3) Install source code on your PC and Raspberry :
	Repo to clone : git clone https://github.com/muratory/deep_drive.git
	
	On Raspberry, clone repo in $HOME
	Raspberry code is located in $HOME/deep_drive/raspberry
	
	Clone repo on your PC in <path> 
	Computer source code is located in <path>/deep_drive/computer
	
4)  Calibration :
	On Raspberry :
		Folder 		 : $HOME/deep_drive/raspberry/server 
		Launch 		 : sudo python cali_server.py
		or use alias : sc
 
	On your PC :
		Folder 		: <path>\deep_drive\computer\client_sf\
		Script 		: client_App.py 
			HOST = 'xxx.xxx.xxx.xxx'    # Server(Raspberry Pi) IP address
		Launch 		: python cali_client.py
	
5) Check Camera :
	Folder 		 : /home/pi/deep_drive/raspberry/mjpg-streamer/mjpg-streamer 
	Launch 		 : ./start.sh &
	or use alias : si
	
	On your computer launch following address :
	http://10.246.51.227:8080/stream.html
	
6) Raspberry TCP server : 
	Folder 		 : $HOME/deep_drive/raspberry/server
	Launch 		 : sudo python tcp_server.py
	or use alias : st

7) Intall Python2.7.x in case it is not already install on your computer
	If installed, check Python version : python --version
	
8) Check commands from PC to Raspberry : 
	On Raspberry : 
		Starts ftp and camera server (si and st aliases)
	
	On your PC :
		Modify  <path>\deep_drive\computer\client_sf\client_App.py to set Raspberry IP Address
			HOST = 'xxx.xxx.xxx.xxx'    # Server(Raspberry Pi) IP address
		python <path>\deep_drive\computer\client_sf\client_App.py
	
9) Training sequence : 
	Folder : <path>\deep_drive\computer
	Script : collect_training_data_sf.py
	Temporary folder to be created : deep_drive\computer\training_data_temp
	
	Prerequisite to be done on EmployeeHotspot :
	Pip 	: python -m pip install -U pip
	numpy 	: python -m pip install numpy
	pygame  : python -m pip install pygame
	cv2 	: 
		download from https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.13/opencv-2.4.13.exe/download
		install procedure : http://docs.opencv.org/trunk/d5/de5/tutorial_py_setup_in_windows.html
		resume : Once downloaded double-click to extract it and copy opencv/build/python/2.7/xx/cv2.pyd to C:/Python27/lib/site-packages
		
	Use arrow to control the car :
	up arrow 	: forward
	down arrow 	: stop
	right arrow : turn right
	left arrow 	: turn left

	When training sequence is complete use "a" key (AZERTY) or "q" (QWERTY) to end the training sequence.
	test08.npz file created in deep_drive\computer\training_data_temp
	
10) Train Neural Network
	copy .npz in deep_drive\computer\training_data
	
	folder : deep_drive\computer
	Script : mlp_training.py
	
	command : python mlp_training.py
	Output : deep_drive\computer\mlp_xml\mlp.xml
	
	Info :
	For 105KB NPZ file size with arround 352 frames, training duration is less than 5 minutes. 
	
11) Replay
	folder : deep_drive\computer
	Script : rc_driver_sf.py
	
	Note : Change HOST variable with you raspberry IP address
	