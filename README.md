
# Track the Constellation

This project is a fun game where you can use your hand gesture to track the stars and check your accuracy.


# Tech used:

- OpenCV
- Mediapipe
- Some more libraries of python


## Deployment

To deploy this project run

```bash
git clone https://github.com/amod404/Constellation_Project.git
```
then run
```
cd Constellation_Project
```
then download the libraries
```
pip install -r requirements.txt
```

## important note : 
 - if you are facing problem in downloading media pipe, then it is because some of the libraries are only supported in python 3.12's version not in 3.13+
 - to fix it you can make virtual environment where you can run older version of python
 - download the python.exe file of the required version and copy its path
 now in Constellation_Project terminal run 
 ```
 {path to .exe} -m venv myenv
 ```
 ```
 myenv\Scripts\activate
 ```
 ```
 pip install -r requirements.txt
 ```


# How to use:

- you will find two variales in your constellation.py file.
  - 1.) map - which is used to select the question number, can varry from 1-6, you can add more images however 
  - 2.) mode - 0 means you are solving, and 1 means you are storing data that will be compared in solving mode to calculate the accuracy
- then just simply run your code
- Now the main parts come there are two kinds of gestures:
  - 1.) one is when your all five fingers are open (normal mode)
  - 2.) second is when only index finger is open and all other are closed (drawing mode)
- You have to track you cursor to the first star and then go to draw mode, and draw the constellation slowly.
- make sure to move your fingers parallel to screen and don't tilt them.
- then open up all the fingers and wait for 3 second, it will show the answer if you win :)


# Improvement

~ You need to track the constellation in a definite path. That is one thing I am thinking upon.


## Support

Special Thanks to Astronomy Club of IIT-BHU who let me dispay my project at welcoming stall for freshers fo 2025-26.
Thanks to my friend Arpit, who also helped me to debug.

For support, email amod.yadav.cse24@itbhu.ac.in