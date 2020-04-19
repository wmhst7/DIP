import cv2
import numpy as np


def preload(name):
    with open('view morphing/'+name+'.json') as f:
        d = eval(f.read())
        item = ['left_eye_lower_left_quarter', 'left_eye_lower_right_quarter'
                ,'left_eye_pupil'
                ,'left_eye_upper_left_quarter'
                , 'left_eye_upper_right_quarter'
                , 'left_eyebrow_lower_left_quarter'
                , 'left_eyebrow_lower_right_quarter'
                , 'left_eyebrow_upper_left_quarter'
                , 'left_eyebrow_upper_right_quarter'
                , 'mouth_lower_lip_top'
                , 'mouth_upper_lip_bottom'
                , 'right_eye_lower_left_quarter'
                , 'right_eye_lower_right_quarter'
                , 'right_eye_pupil'
                , 'right_eye_upper_left_quarter'
                , 'right_eye_upper_right_quarter'
                , 'right_eyebrow_lower_left_quarter'
                , 'right_eyebrow_lower_right_quarter'
                , 'right_eyebrow_upper_left_quarter'
                , 'right_eyebrow_upper_right_quarter'
                ]
        for ii in item:
            del d[ii]
        with open('view morphing/'+name+'_point.json', 'w') as ff:
            l = [[d[x]['x'], d[x]['y']] for x in d]
            print('length:', len(l))
            ff.write(str(l))


if __name__ == '__main__':
    for name in ['source1', 'source2', 'target1', 'target2']:
        preload(name)
