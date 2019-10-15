## DVM code

import time
import numpy as np
from EDGE_4_3_4 import EDGE
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score
import plotly.graph_objs as go
import plotly
import scipy as sp
import scipy.ndimage
#import pickle

# Data Value Metric 
def DVM(X, Y=None, T=10, alpha= 0.1, beta=0.1,  method=None, X_continuous = True, Z_continuous = True, Y_continuous = False, problem_type='supervised', **kwarg):
    # T is the number of different subsets of the original dataset
    # beta is the normalizing factor of the complexity

    MI_score = np.zeros(T)
    method_runtime = np.zeros(T)
    method_accuracy = np.zeros(T)
    I1_array = np.zeros(T)
    I2_array = np.zeros(T)
    I3_array = np.zeros(T)

    for i in range(T):
        
        if problem_type == 'unsupervised':
            num_sample = X.shape[0]
            num_sample_train = int(num_sample/3)
            I=range(num_sample)

            # Create I_test
            I_test = np.random.choice(I, size=num_sample_train, replace=False, p=None)
            I_train = np.array([r for r in I if not r in I_test])
            I_train1 = I_train[0:num_sample_train]
            I_train2 = I_train[num_sample_train:]

            # Train model and predict
            start_time = time.clock()
            Z1 = method(X_train=X[I_train1], X_test=X[I_test], **kwarg)
            Z2 = method(X_train=X[I_train2], X_test=X[I_test], **kwarg)

            method_runtime[i] = time.clock() - start_time
            
            ####discreete variable
            if X_continuous:
                gamma_X = 1.0
            else:
                gamma_X = 0.001

            if Z_continuous:
                gamma_Z = 1
            else:
                gamma_Z = 0.001

            if Y_continuous:
                gamma_Y = 1
            else:
                gamma_Y = 0.001

            I1 = EDGE(Z1, Z2, gamma = [gamma_Y,gamma_Z])
            I2 = EDGE(X[I_test], Z1, gamma = [gamma_X,gamma_Z])
            I3 = EDGE(X[I_test], Z2, gamma = [gamma_X,gamma_Y])

            MI_score[i] = (I1 - beta * (I2-I1))/I3
            #MI_score[i] = 2*EDGE(Z1,Z2,gamma=[0.001,0.001])/ (EDGE(Z1,Z1,gamma=[0.001,0.001])+EDGE(Z2,Z2,gamma=[0.001,0.001])) 
            
            method_accuracy[i] = (silhouette_score(X[I_test],Z1,metric='euclidean') + 1) / 2
            
            I1_array[i] = I1
            I2_array[i] = I2
            I3_array[i] = I3
        
        if problem_type == 'supervised':
            num_sample = X.shape[0]
            num_sample_train = int(num_sample/2)
            I=range(num_sample)

            I_train = np.random.choice(I, size=num_sample_train, replace=False, p=None)
            I_test = np.array([r for r in I if not r in I_train])

        # Train model and predict
            start_time = time.clock()
            Z = method(X_train=X[I_train], Y_train=Y[I_train], X_test=X[I_test], **kwarg)
            method_runtime[i] = time.clock() - start_time
            
        ####Special for lasso
        #Y_pred = Y_pred.round().astype(int)
        
        ####discreete variable
            if X_continuous:
                gamma_X = 1.0
            else:
                gamma_X = 0.001

            if Z_continuous:
                gamma_Z = 1
            else:
                gamma_Z = 0.001

            if Y_continuous:
                gamma_Y = 1
            else:
                gamma_Y = 0.001

            I1 = EDGE(Y[I_test], Z, gamma = [gamma_Y,gamma_Z])
            I2 = EDGE(X[I_test], Z, gamma = [gamma_X,gamma_Z])
            I3 = EDGE(X[I_test], Y[I_test], gamma = [gamma_X,gamma_Y])

            MI_score[i] = (I1 - beta * (I2-I1))/I3
        
        #continuous variable
        #MI_score[i] = EDGE(Y[I_test],Y_pred,gamma = [0.9,1])/ EDGE(Y[I_test],Y[I_test],gamma = [0.9,1])
       
            if not Y_continuous:
                method_accuracy[i] = accuracy_score(Y[I_test],Z)
            else:
                method_accuracy[i] = r2_score(Y[I_test],Z)
        
        
    accuracy = np.mean(MI_score)
    ## Theoretical computational complexity
    #complexity = n_neighbors* N * math.log(N)* 10**(-5)
    complexity = np.mean(method_runtime)
    #beta = 1
    DVM= accuracy - alpha*complexity/num_sample
    DVM = max(0,DVM)
    model_accuracy = np.mean(method_accuracy)
    I1 = np.mean(I1_array)
    I2 = np.mean(I2_array)
    I3 = np.mean(I3_array)
    
    q = 5
    percentile_upper = np.percentile(MI_score, q)
    percentile_lower = np.percentile(MI_score, 100-q)
    confidence_band = (-percentile_upper+percentile_lower)/2.0
    
    return {"DVM": DVM, "Accuracy": model_accuracy, "MI":accuracy,
            "Complexity":complexity,"Confidence Band":confidence_band,
            "I1":I1,"I2":I2,"I3":I3}

# Data Value Metric Graph 
def plot_DVM(X, Y=None,X_continuous = False,Y_continuous = False,
             sample_len = 10, feature_len = 10, 
             sample_start = None,feature_start = None,
             T=10, alpha = 0.1, beta=1,sigma_x = 0, sigma_y = 0,method=None,
             filename = "tempo",problem_type='supervised',
             plot_type = '3D',
             **kwarg):
    
    num_sample = X.shape[0]
    num_feature = X.shape[1]
    if sample_start == None:
        sample_start = round(num_sample * 0.1)
    
    if feature_start == None:
        feature_start = round(num_feature * 0.1)
        
    sample_seq = [int(i) for i in np.linspace(sample_start,num_sample,sample_len)]
    feature_seq = [int(i) for i in np.linspace(feature_start,num_feature,feature_len)]
    
    DVM_value = np.zeros((sample_len,feature_len))
    MI = np.zeros((sample_len,feature_len))
    Accuracy = np.zeros((sample_len,feature_len))
    Complexity = np.zeros((sample_len,feature_len))
    confidence_band = np.zeros((sample_len,feature_len))
    I1_matrix = np.zeros((sample_len,feature_len))
    I2_matrix = np.zeros((sample_len,feature_len))
    I3_matrix = np.zeros((sample_len,feature_len))
    
    MI_row_id = 0
    for s in sample_seq:
        I_s = np.random.choice(range(X.shape[0]), size=s, replace=False, p=None)
        MI_col_id = 0
        for f in feature_seq:
            I_f = np.random.choice(range(X.shape[1]), size=f, replace=False, p=None)
            if problem_type == 'supervised':
                X_sub=X[I_s,:]
                X_sub=X_sub[:,I_f]
                Y_sub=Y[I_s]
                
                result = DVM(X = X_sub, Y = Y_sub,  T = T,alpha = alpha, beta = beta, 
                             method=method,X_continuous = X_continuous,
                             Y_continuous = Y_continuous, problem_type = problem_type,**kwarg)
            if problem_type == 'unsupervised':
                X_sub =X[I_s,:]
                X_sub =X_sub[:,I_f]
                result = DVM(X = X_sub,T = T,alpha = alpha, beta = beta, 
                             method=method,X_continuous = X_continuous,
                             Y_continuous = Y_continuous, problem_type = problem_type,**kwarg)
            DVM_value[MI_row_id, MI_col_id] = result["DVM"]
            Accuracy[MI_row_id, MI_col_id] = result["Accuracy"]
            MI[MI_row_id, MI_col_id] = result["MI"]
            Complexity[MI_row_id, MI_col_id] = result["Complexity"]
            confidence_band[MI_row_id, MI_col_id] = result["Confidence Band"]
            I1_matrix[MI_row_id, MI_col_id] = result["I1"]
            I2_matrix[MI_row_id, MI_col_id] = result["I2"]
            I3_matrix[MI_row_id, MI_col_id] = result["I3"]
            
            MI_col_id += 1
        MI_row_id += 1
    
    sigma = [sigma_y, sigma_x]
    DVM_value = sp.ndimage.filters.gaussian_filter(DVM_value, sigma, mode='constant')
    Accuracy = sp.ndimage.filters.gaussian_filter(Accuracy, sigma, mode='constant')
    
    trace3D_DVM = go.Surface(x = feature_seq,y = sample_seq, z=DVM_value, 
                             colorscale='Viridis',name = 'DVM')
    trace3D_Accuracy = go.Surface(x = feature_seq,y = sample_seq, z=Accuracy, colorscale='Portland',
                                  opacity=0.7,name = 'Accuracy')
    layout3D = go.Layout(
            scene = dict (
            xaxis = dict(title = "Number of Features",),
            yaxis = dict(title = "Number of Samples",),
            zaxis = dict(title = "DVM and Accuracy"),),)
    
    trace2D_sample_DVM = go.Scatter(x = sample_seq,
                    y = DVM_value[:,-1],
                    mode = 'lines',
                    name = 'DVM')
    trace2D_sample_Accuracy = go.Scatter(x = sample_seq,
                    y = Accuracy[:,-1],
                    mode = 'lines',
                    name = 'Accuracy')
    layout2D_sample = go.Layout(
        title = 'DVM and Accuracy v.s Sample Number',
        xaxis = dict(title = 'Sample Number'),
        yaxis = dict(title = 'DVM & Accuracy'),
        )
    
    trace2D_feature_DVM = go.Scatter(x = feature_seq,
                    y = DVM_value[-1,:],
                    mode = 'lines',
                    name = 'DVM')
    trace2D_feature_Accuracy = go.Scatter(x = feature_seq,
                    y = Accuracy[-1,:],
                    mode = 'lines',
                    name = 'Accuracy')
    layout2D_feature = go.Layout(
        title = 'DVM and Accuracy v.s Feature Number',
        xaxis = dict(title = 'Feature Number'),
        yaxis = dict(title = 'DVM & Accuracy'),
        )
    #layout = go.Layout(scene = dict(
            #zaxis = dict(
                   #range = [0.99990,1],),))
    #plotly.offline.plot({"data": [trace1],"layout": layout}, auto_open=True, filename = filename)
    if plot_type == '3D':
        plotly.offline.plot({"data": [trace3D_DVM,trace3D_Accuracy],"layout":layout3D}, 
                         auto_open=False, filename = filename + '3D'+'.html')
    if plot_type == '2D':
        plotly.offline.plot({"data": [trace2D_sample_DVM,trace2D_sample_Accuracy],
                         "layout":layout2D_sample}, auto_open=False, filename = filename + '2D_Sample' + '.html')
        plotly.offline.plot({"data": [trace2D_feature_DVM,trace2D_feature_Accuracy],
                         "layout":layout2D_feature}, auto_open=False, filename = filename + '2D_feature' + '.html')
    if plot_type == 'Both':
        plotly.offline.plot({"data": [trace3D_DVM,trace3D_Accuracy],"layout":layout3D}, 
                         auto_open=False, filename = filename + '3D'+'.html')
        plotly.offline.plot({"data": [trace2D_sample_DVM,trace2D_sample_Accuracy],
                         "layout":layout2D_sample}, auto_open=False, filename = filename + '2D_Sample' + '.html')
        plotly.offline.plot({"data": [trace2D_feature_DVM,trace2D_feature_Accuracy],
                         "layout":layout2D_feature}, auto_open=False, filename = filename + '2D_feature' + '.html')   
    
    return {"Accuracy":Accuracy,"MI":MI,"Complexity" : Complexity,"DVM":DVM_value,
            "Sample Number":sample_seq, "Feature Number":feature_seq,
            "Confidence band":confidence_band,"I1":I1_matrix,"I2":I2_matrix,"I3":I3_matrix}
    
