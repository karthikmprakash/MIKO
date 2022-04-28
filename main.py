import cv2
import mediapipe as mp
import numpy as np
import sys
from scipy.spatial import distance
from numpy import mean, sqrt, square
from scipy.stats import entropy
import time


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(sys.argv[1])

if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

outdir, inputflnm = sys.argv[1][:sys.argv[1].rfind(
    '/')+1], sys.argv[1][sys.argv[1].rfind('/')+1:]
inflnm, inflext = inputflnm.split('.')
out_filename = f'{outdir}{inflnm}_annotated.{inflext}'
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
    
i = 0

def calculate_rms_energy(distance_metric):
    rms_sqr_energy = 0
    for i in range(0,len(distance_metric)-1):
        rms_sqr_energy += pow((distance_metric[i] - distance_metric[i+1]),2)
        rms_energy = np.sqrt(rms_sqr_energy/len(distance_metric))
    return rms_energy
    
def get_distance_metric(landmarks):
    
    nose = (landmarks[mp_pose.PoseLandmark.NOSE.value].x , landmarks[mp_pose.PoseLandmark.NOSE.value].y , landmarks[mp_pose.PoseLandmark.NOSE.value].z)
    left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x , landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y , landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z)
    right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y , landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z)
    left_elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x , landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y , landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z)
    right_elbow = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x ,  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y , landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z)
    left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x , landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y , landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z)
    right_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y , landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z)
    left_ankle= (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x , landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y , landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z)
    right_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y , landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z)
    
    nose_vis = landmarks[mp_pose.PoseLandmark.NOSE.value].visibility
    lw_vis = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility
    rw_vis = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility
    le_vis = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
    re_vis = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
    lk_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    rk_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
    la_vis = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility
    ra_vis = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility
    
    # computing the euclidean distance
    nose_lw_distance = distance.euclidean(nose, left_wrist) * nose_vis * lw_vis
    #print('Euclidean Distance b/w nose and left wrist is: ', nose_lw_distance)

    # computing the euclidean distance
    nose_rw_distance = distance.euclidean(nose, right_wrist) * nose_vis * rw_vis
    #print('Euclidean Distance b/w nose and right wrist is: ', nose_rw_distance)

    # computing the euclidean distance
    nose_le_distance = distance.euclidean(nose, left_elbow) * nose_vis * le_vis
    #print('Euclidean Distance b/w nose and left elbow is: ', nose_le_distance)

    # computing the euclidean distance
    nose_re_distance = distance.euclidean(nose, right_elbow) * nose_vis * re_vis
    #print('Euclidean Distance b/w nose and right elbow is: ', nose_re_distance )

    # computing the euclidean distance
    lw_rw_distance = distance.euclidean(right_wrist, left_wrist) * rw_vis * lw_vis
    #print('Euclidean Distance b/w right wrist and left wrist is: ', lw_rw_distance)

    # computing the euclidean distance
    nose_rk_distance = distance.euclidean(nose, right_knee) * nose_vis * rk_vis
    #print('Euclidean Distance b/w nose and right knee is: ', nose_rk_distance)

    # computing the euclidean distance
    nose_lk_distance = distance.euclidean(nose, left_knee) * nose_vis * lk_vis
    #print('Euclidean Distance b/w nose and left knee is: ', nose_lk_distance )

    # computing the euclidean distance
    nose_ra_distance = distance.euclidean(nose, right_ankle) * nose_vis * ra_vis
    #print('Euclidean Distance b/w nose and right ankle is: ', nose_ra_distance)

    # computing the euclidean distance
    nose_la_distance = distance.euclidean(nose, left_ankle) * nose_vis * la_vis
    #print('Euclidean Distance b/w nose and left ankle is: ', nose_la_distance)

    # computing the euclidean distance
    la_ra_distance = distance.euclidean(left_ankle, right_ankle) * la_vis * ra_vis
    #print('Euclidean Distance b/w left ankle and right ankle is: ', la_ra_distance)
    
    # Energy Calculation 
    distance_metric = np.sum([nose_lw_distance,nose_rw_distance,nose_le_distance,nose_re_distance,lw_rw_distance,nose_rk_distance,nose_lk_distance,nose_ra_distance,nose_la_distance,la_ra_distance])/ np.sum([nose_vis, lw_vis, rw_vis, le_vis, re_vis, lk_vis, rk_vis, la_vis, ra_vis])
    

    return distance_metric


rms_energy=0
i=0
distance_metric = []

# Set the window size 
window_size = 10

while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detection
        start = time.time()
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if i % 6 == 0:
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                print(i)
                
                if len(distance_metric) < window_size:
                    distance_metric.append(get_distance_metric(landmarks))
                    
                elif len(distance_metric) == window_size:
                    distance_metric.pop(0)
                    distance_metric.append(get_distance_metric(landmarks))
                    rms_energy = calculate_rms_energy(distance_metric)
                    end = time.time()
                    print("Time is", end-start)
                    #print("RMS Energy" ,rms_energy)
                        
            except:
                pass
        
        
        
        ##Energy status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        # Energy data
        cv2.putText(image, 'RMS ENERGY', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(round(rms_energy*100,2)), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               

        out.write(image)
        
        if i<30:
            i+=1
        else:
            i=0
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

pose.close()
cap.release()
out.release()


