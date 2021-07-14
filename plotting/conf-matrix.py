'''
#########       Parallel Confusion Matrix       #########
	        ang     hap     sad     neu     fea     sur
	^ang|   184     10      3       20      1       11
	^hap|   4       232     7       16      1       1
	^sad|   5       17      147     58      1       2
	^neu|   53      45      64      244     3       8
	^fea|   0       0       0       0       0       0
	^sur|   0       0       0       0       0       0


#### . Sequential
ccuracy: 0.7836879432624113
--------------------
              precision    recall  f1-score   support

           0       0.85      0.74      0.79       222
           1       0.90      0.93      0.92       333
           2       0.78      0.50      0.61       195
           3       0.67      0.88      0.76       353
           6       1.00      0.11      0.20         9
           7       0.00      0.00      0.00        16

    accuracy                           0.78      1128
   macro avg       0.70      0.53      0.55      1128
weighted avg       0.79      0.78      0.77      1128

--------------------
#########	 Confusion Matrix 	#########
	ang	hap	sad	neu	fea	sur	
^ang|	164	1	22	6	0	1	
^hap|	2	309	2	25	0	4	
^sad|	12	0	98	10	3	2	
^neu|	44	23	73	312	5	9	
^fea|	0	0	0	0	1	0	
^sur|	0	0	0	0	0	0
'''
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
emotions = ['Anger', 'Happy', 'Sad', 'Neutral', 'Fear', 'Surprise']


def draw_parallel():
    parallel_matrix = [[184, 10, 3,  20, 1,  11],
                       [4,  232, 7,  16, 1,  1],
                       [5,  17, 147, 58, 1,  2],
                       [53, 45, 64, 244, 3,  8],
                       [0,  0,  0,  0,  0,  0],
                       [0,  0,  0,  0,  0,  0]]
    matrix = pd.DataFrame(parallel_matrix, index=emotions, columns=emotions)
    plt.figure(figsize=(8, 8))
    sn.heatmap(matrix, annot=True, cmap='BuGn_r', fmt='g')
    plt.savefig(
        f'/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/scripts/plotting/conf-matrix/parallel_conf_matrix')


draw_parallel()
