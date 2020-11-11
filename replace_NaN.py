'''
Here, we read npy data and check if a file contains NaN. If so, replace it with 0 using np.nan_to_num()
'''
import os
import glob
import numpy as np

input_path = 'data/bigan/'
phases = os.listdir(input_path)
for p in ['test']:
    print('phase: ', p)
    phase_path = os.path.join(input_path, p)
    classes = os.listdir(phase_path)
    for c in classes:
        print('class: ', c)
        class_path = os.path.join(phase_path, c)
        cases = glob.glob(os.path.join(class_path, '*.npy'))
        print('number of cases: ', len(cases))
        for case in cases:
            print('case: ', case)
            ar = np.load(case)
            print('loaded shape: ', ar.shape)
            bol = np.any(np.isnan(ar))
            if(bol):
                print('yes, it contains NaNs')
                ar = np.nan_to_num(ar)
                np.save(case, ar)
                print('saved again')
            else:
                print('no NaN in this file')
                
    
