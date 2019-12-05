# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 14:37:38 2019

@author: sunym
"""

import time
import numpy as np
from EDGE_4_4_1 import EDGE
from sklearn.metrics import accuracy_score
#from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score
from sklearn import metrics
import plotly.graph_objs as go
import plotly
import scipy as sp
import scipy.ndimage
from sklearn.metrics import mean_squared_error
#import pickle

# Data Value Metric 
def DVM(X, Y=None, T=10, alpha= 0.1, beta=0.1, 
        normalization = 'MI' ,method=None, X_continuous = True, 
        Z_continuous = True, Y_continuous = False, problem_type='supervised', 
        GiveAccuracy = True,**kwarg):
    # T is the number of different subsets of the original dataset
    # beta is the normalizing factor of the complexity

    MI_score = np.zeros(T)
    method_runtime = np.zeros(T)
    method_accuracy = np.zeros(T)
    I1_array = np.zeros(T)
    I2_array = np.zeros(T)
    I3_array = np.zeros(T)

    for i in range(T):
        print('i/T', i/T)
        
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
            
            #method_accuracy[i] = (silhouette_score(X[I_test],Z1,metric='euclidean') + 1) / 2
            if GiveAccuracy:
                method_accuracy[i] = metrics.calinski_harabasz_score(X[I_test],Z1) 
            
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
                gamma_X = 0.35
            else:
                gamma_X = 0.001

            if Z_continuous:
                gamma_Z = 0.35
            else:
                gamma_Z = 0.001

            if Y_continuous:
                gamma_Y = 0.35
            else:
                gamma_Y = 0.001

            if Z_continuous:
                I1 = EDGE(Y[I_test], Z, L_ensemble=1, gamma = [gamma_Y,gamma_Z],hashing='TSNE')
            else: 
                I1 = EDGE(Y[I_test], Z, L_ensemble=1, gamma = [gamma_Y,gamma_Z])

            I2 = EDGE(X[I_test], Z, L_ensemble=1, gamma = [gamma_X,gamma_Z],hashing='TSNE')
            
            if normalization == 'MI':
                I3 = EDGE(X[I_test], Y[I_test], L_ensemble=1, gamma = [gamma_X,gamma_Y],hashing='TSNE')
            elif normalization =='Entropy':
                I3 = EDGE(X[I_test], Y[I_test], L_ensemble=1, gamma = [0.001*gamma_X,gamma_Y],hashing='TSNE')

            MI_score[i] = (I1 - beta * (I2-I1))/I3
            
            I1_array[i] = I1
            I2_array[i] = I2
            I3_array[i] = I3
        
        #continuous variable
        #MI_score[i] = EDGE(Y[I_test],Y_pred,gamma = [0.9,1])/ EDGE(Y[I_test],Y[I_test],gamma = [0.9,1])
       
            if GiveAccuracy:
                if not Y_continuous:
                    method_accuracy[i] = accuracy_score(Y[I_test],Z)
                else:
                    method_accuracy[i] = mean_squared_error(Y[I_test],Z)
        
        
    accuracy = np.mean(MI_score)
    ## Theoretical computational complexity
    #complexity = n_neighbors* N * math.log(N)* 10**(-5)
    complexity = np.mean(method_runtime)
    #beta = 1
    DVM= accuracy - alpha*complexity/num_sample
    DVM = min(max(0,DVM),1)
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
             plot_type = '3D',GiveAccuracy = True,
             RemoveSigularities = True,
             **kwarg):
    
    num_sample = X.shape[0]
    num_feature = X.shape[1]
    if sample_start == None:
        sample_start = round(num_sample * 0.1)
    
    if feature_start == None:
        feature_start = round(num_feature * 0.1)
        
    sample_seq = [int(i) for i in np.linspace(sample_start,num_sample,sample_len)]
    feature_seq = [int(i) for i in np.linspace(feature_start,num_feature,feature_len)]
    
    print('sample_seq',sample_seq)
    print('feature_seq',feature_seq)
    
    if plot_type == '2D':
        DVM_value_sample = np.zeros((sample_len,))
        MI_sample = np.zeros((sample_len,))
        Accuracy_sample = np.zeros((sample_len,))
        Complexity_sample = np.zeros((sample_len,))
        confidence_band_sample = np.zeros((sample_len,))
        I1_sample = np.zeros((sample_len,))
        I2_sample = np.zeros((sample_len,))
        I3_sample = np.zeros((sample_len,))
        
        DVM_value_feature = np.zeros((feature_len,))
        MI_feature = np.zeros((feature_len,))
        Accuracy_feature = np.zeros((feature_len,))
        Complexity_feature = np.zeros((feature_len,))
        confidence_band_feature = np.zeros((feature_len,))
        I1_feature = np.zeros((feature_len,))
        I2_feature = np.zeros((feature_len,))
        I3_feature = np.zeros((feature_len,))
        
        sample_id = 0
        for s in sample_seq:
            print('s in sample_seq: ', s)
            I_s = np.random.choice(range(X.shape[0]), size=s, replace=False, p=None)
            I_f = np.random.choice(range(X.shape[1]), size=feature_seq[-1], replace=False, p=None)
            if problem_type == 'supervised':
                X_sub=X[I_s,:]
                X_sub=X_sub[:,I_f]
                Y_sub=Y[I_s]
                
                result = DVM(X = X_sub, Y = Y_sub,  T = T,alpha = alpha, beta = beta, 
                             method=method,X_continuous = X_continuous,
                             Y_continuous = Y_continuous, problem_type = problem_type,
                             GiveAccuracy=GiveAccuracy,**kwarg)
            if problem_type == 'unsupervised':
                X_sub =X[I_s,:]
                X_sub =X_sub[:,I_f]
                result = DVM(X = X_sub,T = T,alpha = alpha, beta = beta, 
                             method=method,X_continuous = X_continuous,
                             Y_continuous = Y_continuous, problem_type = problem_type,
                             GiveAccuracy=GiveAccuracy,**kwarg)
            DVM_value_sample[sample_id,] = result["DVM"]
            Accuracy_sample[sample_id,] = result["Accuracy"]
            MI_sample[sample_id, ] = result["MI"]
            Complexity_sample[sample_id, ] = result["Complexity"]
            confidence_band_sample[sample_id, ] = result["Confidence Band"]
            I1_sample[sample_id, ] = result["I1"]
            I2_sample[sample_id, ] = result["I2"]
            I3_sample[sample_id, ] = result["I3"]
            sample_id += 1
            
        feature_id = 0
        for f in feature_seq:
            print('f in feature_seq: ', f)
            I_s = np.random.choice(range(X.shape[0]), size=sample_seq[-1], replace=False, p=None)
            I_f = np.random.choice(range(X.shape[1]), size=f, replace=False, p=None)
            if problem_type == 'supervised':
                X_sub=X[I_s,:]
                X_sub=X_sub[:,I_f]
                Y_sub=Y[I_s]
                
                result = DVM(X = X_sub, Y = Y_sub,  T = T,alpha = alpha, beta = beta, 
                             method=method,X_continuous = X_continuous,
                             Y_continuous = Y_continuous, problem_type = problem_type,
                             GiveAccuracy=GiveAccuracy,**kwarg)
            if problem_type == 'unsupervised':
                X_sub =X[I_s,:]
                X_sub =X_sub[:,I_f]
                result = DVM(X = X_sub,T = T,alpha = alpha, beta = beta, 
                             method=method,X_continuous = X_continuous,
                             Y_continuous = Y_continuous, problem_type = problem_type,
                             GiveAccuracy=GiveAccuracy,**kwarg)
            DVM_value_feature[feature_id,] = result["DVM"]
            Accuracy_feature[feature_id,] = result["Accuracy"]
            MI_feature[feature_id, ] = result["MI"]
            Complexity_feature[feature_id, ] = result["Complexity"]
            confidence_band_feature[feature_id, ] = result["Confidence Band"]
            I1_feature[feature_id, ] = result["I1"]
            I2_feature[feature_id, ] = result["I2"]
            I3_feature[feature_id, ] = result["I3"]
            feature_id += 1
        
        if Y_continuous == True:
            Accuracy_sample = Accuracy_sample/np.max(Accuracy_sample)
            Accuracy_feature = Accuracy_feature/np.max(Accuracy_feature)
            
        trace2D_sample_DVM = go.Scatter(x = sample_seq,
                    y = DVM_value_sample,
                    mode = 'lines',
                    name = 'DVM')
        if Y_continuous == True:
             trace2D_sample_Accuracy = go.Scatter(x = sample_seq,
                        y = Accuracy_sample,
                        mode = 'lines',
                        name = 'MSE')
             layout2D_sample = go.Layout(
                    title = go.layout.Title(x=0.5,text = 'DVM and MSE vs. Number of Samples',xanchor="center"),
                    xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Samples",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                            tickfont = dict(size = 20)),
                    yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM & MSE",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                            tickfont = dict(size = 20)),
                    plot_bgcolor = 'rgb(255,255,255,0)')
             trace2D_feature_Accuracy = go.Scatter(x = feature_seq,
                        y = Accuracy_feature,
                        mode = 'lines',
                        name = 'MSE')
             layout2D_feature = go.Layout(
                title = go.layout.Title(x=0.5,text = 'DVM and MSE vs. Number of Features',xanchor="center"),
                xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Features",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                        tickfont = dict(size = 20)),
                yaxis =  go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM & MSE",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                         tickfont =dict(size = 20)),
                plot_bgcolor = 'rgb(255,255,255,0)')
        else:
            trace2D_sample_Accuracy = go.Scatter(x = sample_seq,
                        y = Accuracy_sample,
                        mode = 'lines',
                        name = 'Accuracy')
            layout2D_sample = go.Layout(
                    title = go.layout.Title(x=0.5,text = 'DVM and Accuracy vs. Number of Features',xanchor="center"),
                    xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Features",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                            tickfont = dict(size = 20)),
                    yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM & Accuracy",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                            tickfont = dict(size = 20)),
                    plot_bgcolor = 'rgb(255,255,255,0)')
            trace2D_feature_Accuracy = go.Scatter(x = feature_seq,
                        y = Accuracy_feature,
                        mode = 'lines',
                        name = 'Accuracy')
            layout2D_feature = go.Layout(
                title = go.layout.Title(x=0.5,text = 'DVM and Accuracy vs. Number of Features',xanchor="center"),
                xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Features",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                        tickfont = dict(size = 20)),
                yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM & Accuracy",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                        tickfont = dict(size = 20)),
                plot_bgcolor = 'rgb(255,255,255,0)')
            
        layout2D_sample_NoAccuracy = go.Layout(
                title = go.layout.Title(x=0.5,text = 'DVM vs. Number of Samples',xanchor="center"),
                xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Samples",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                        tickfont = dict(size = 20)),
                yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                        tickfont = dict(size = 20)),
                plot_bgcolor = 'rgb(255,255,255,0)')
        trace2D_feature_DVM = go.Scatter(x = feature_seq,
                        y = DVM_value_feature,
                        mode = 'lines',
                        name = 'DVM')
        layout2D_feature_NoAccuracy = go.Layout(
            title = go.layout.Title(x=0.5,text = 'DVM vs. Number of Features',xanchor="center"),
            xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Features",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                    tickfont = dict(size = 20)),
            yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                    tickfont = dict(size = 20)),
            plot_bgcolor = 'rgb(255,255,255,0)')
        if GiveAccuracy:
            plotly.offline.plot({"data": [trace2D_sample_DVM,trace2D_sample_Accuracy],
                         "layout":layout2D_sample}, auto_open=False, filename = filename + '2D_Sample' + '.html')
            plotly.offline.plot({"data": [trace2D_feature_DVM,trace2D_feature_Accuracy],
                         "layout":layout2D_feature}, auto_open=False, filename = filename + '2D_feature' + '.html')
        else:
            plotly.offline.plot({"data": [trace2D_sample_DVM],
                         "layout":layout2D_sample_NoAccuracy}, auto_open=False, filename = filename + '2D_sample' + '.html')#filename + '2D_Sample' + '.html')
            plotly.offline.plot({"data": [trace2D_feature_DVM],
                         "layout":layout2D_feature_NoAccuracy}, auto_open=False, filename = filename + '2D_feature' + '.html')            
        
        return {"Accuracy_sample":Accuracy_sample,"Accuracy_feature":Accuracy_feature,
                "MI_sample":MI_sample,"MI_feature":MI_feature,
                "Complexity_sample" : Complexity_sample,"Complexity_feature":Complexity_feature,
                "DVM_sample":DVM_value_sample,"DVM_feature":DVM_value_feature,
                "Sample Number":sample_seq, "Feature Number":feature_seq,
                "Confidence band Sample":confidence_band_sample,
                "Confidence band feature":confidence_band_feature,
                "I1_sample":I1_sample,"I2_sample":I2_sample,"I3_sample":I3_sample,
                "I1_feature":I1_feature,"I2_feature":I2_feature,"I3_feature":I3_feature}       
    else:
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
            print('s in sample_seq ', s)
            I_s = np.random.choice(range(X.shape[0]), size=s, replace=False, p=None)
            MI_col_id = 0
            for f in feature_seq:
                print('f in feature_seq: ', f)
                I_f = np.random.choice(range(X.shape[1]), size=f, replace=False, p=None)
                if problem_type == 'supervised':
                    X_sub=X[I_s,:]
                    X_sub=X_sub[:,I_f]
                    Y_sub=Y[I_s]
                    
                    result = DVM(X = X_sub, Y = Y_sub,  T = T,alpha = alpha, beta = beta, 
                                 method=method,X_continuous = X_continuous,
                                 Y_continuous = Y_continuous, problem_type = problem_type,
                                 GiveAccuracy=GiveAccuracy,**kwarg)
                if problem_type == 'unsupervised':
                    X_sub =X[I_s,:]
                    X_sub =X_sub[:,I_f]
                    result = DVM(X = X_sub,T = T,alpha = alpha, beta = beta, 
                                 method=method,X_continuous = X_continuous,
                                 Y_continuous = Y_continuous, problem_type = problem_type,
                                 GiveAccuracy=GiveAccuracy,**kwarg)
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
        
        if Y_continuous == True:
            Accuracy = Accuracy/np.max(Accuracy)
        
        if RemoveSigularities:
            Gt1 = 0
            for i in range(sample_len):
                for j in range(feature_len):
                    if DVM_value[i,j] > 1:
                        Gt1 = Gt1 + 1
                        
            if Gt1 < 0.1 * sample_len * feature_len and Gt1 > 0:
                M = np.mean(DVM_value[DVM_value <= 1])
                for i in range(sample_len):
                    for j in range(feature_len):
                        if DVM_value[i,j] > 1:
                            DVM_value[i,j] = M
         
        if Y_continuous == True:
            feature_seq_v,sample_seq_v = np.meshgrid(feature_seq,sample_seq)
            feature_seq_v = feature_seq_v.astype('<59U')
            sample_seq_v = sample_seq_v.astype('<59U')
            my_text1 = np.round(Accuracy,decimals=4)
            my_text1 = my_text1.astype('<U59')
            for i in range(sample_len):
                for j in range(feature_len):
                    my_text1[i,j] = 'Sample: ' + sample_seq_v[i,j] + '<br>' + 'Feature: ' + feature_seq_v[i,j] + '<br>' +'MSE: ' + my_text1[i,j]
            #my_text1 = my_text1.tolist()
            trace3D_Accuracy = go.Surface(x = feature_seq,y = sample_seq, z=Accuracy,text=my_text1,
                                          hoverinfo = 'text',
                                        colorscale = 'Portland',showscale=False,opacity=0.7,name = 'MSE')
            layout3D = go.Layout(
                scene = dict (
                xaxis = dict(title = "Number of Features",titlefont=dict(size=17),gridcolor='rgb(230, 230, 230)',backgroundcolor='rgb(255, 255,255)'),
                yaxis = dict(title = "Number of Samples",titlefont=dict(size=17),gridcolor='rgb(230, 230, 230)',backgroundcolor='rgb(255, 255,255)'),
                zaxis = dict(title = "DVM and MSE",titlefont = dict(size=17),gridcolor='rgb(230, 230, 230)',backgroundcolor='rgb(255, 255,255)'),),)
            trace2D_sample_Accuracy = go.Scatter(x = sample_seq,
                        y = Accuracy[:,-1],
                        mode = 'lines',
                        name = 'MSE')
            layout2D_sample = go.Layout(
                    title = go.layout.Title(x=0.5,text = 'DVM and MSE vs. Number of Samples',xanchor="center"),
                    xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Samples",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                            tickfont = dict(size = 20)),
                    yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM & MSE",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                            tickfont = dict(size = 20)),
                    plot_bgcolor = 'rgb(255,255,255,0)')
            trace2D_feature_Accuracy = go.Scatter(x = feature_seq,
                        y = Accuracy[-1,:],
                        mode = 'lines',
                        name = 'MSE')
            layout2D_feature = go.Layout(
                title = go.layout.Title(x=0.5,text = 'DVM and MSE vs. Number of Features',xanchor="center"),
                xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Features",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                        tickfont = dict(size = 20)),
                yaxis =  go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM & MSE",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                         tickfont = dict(size = 20)),
                plot_bgcolor = 'rgb(255,255,255,0)')
        else:
            feature_seq_v,sample_seq_v = np.meshgrid(feature_seq,sample_seq)
            feature_seq_v = feature_seq_v.astype('<59U')
            sample_seq_v = sample_seq_v.astype('<59U')
            my_text2 = np.round(Accuracy,decimals=4)
            my_text2 = my_text2.astype('<U59')
            for i in range(sample_len):
                for j in range(feature_len):
                    my_text2[i,j] = 'Sample: ' + sample_seq_v[i,j] + '<br>' + 'Feature: ' + feature_seq_v[i,j] + '<br>' +'Accuracy: ' + my_text2[i,j]
            #my_text2 = my_text2.tolist()
            trace3D_Accuracy = go.Surface(x = feature_seq,y = sample_seq, z=Accuracy,text = my_text2,
                                          hoverinfo = 'text',
                                      colorscale = 'Portland',showscale=False,opacity=0.7,name = 'Accuracy')
            layout3D = go.Layout(
                scene = dict (
                xaxis = dict(title = "Number of Features",titlefont=dict(size=17),gridcolor='rgb(230, 230, 230)',backgroundcolor='rgb(255, 255,255)'),
                yaxis = dict(title = "Number of Samples",titlefont=dict(size=17),gridcolor='rgb(230, 230, 230)',backgroundcolor='rgb(255, 255,255)'),
                zaxis = dict(title = "DVM and Accuracy",titlefont = dict(size=17),gridcolor='rgb(230, 230, 230)',backgroundcolor='rgb(255, 255,255)'),
                ),)
            trace2D_sample_Accuracy = go.Scatter(x = sample_seq,
                        y = Accuracy[:,-1],
                        mode = 'lines',
                        name = 'Accuracy')
            layout2D_sample = go.Layout(
                    title = go.layout.Title(x=0.5,text = 'DVM and Accuracy vs. Number of Samples',xanchor="center"),
                    xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Samples",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                            tickfont = dict(size = 20)),
                    yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM & Accuracy",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                            tickfont = dict(size = 20)),
                    plot_bgcolor = 'rgb(255,255,255,0)')
            trace2D_feature_Accuracy = go.Scatter(x = feature_seq,
                        y = Accuracy[-1,:],
                        mode = 'lines',
                        name = 'Accuracy')
            layout2D_feature = go.Layout(
                title = go.layout.Title(x=0.5,text = 'DVM and Accuracy vs. Number of Features',xanchor="center"),
                xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Features",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                        tickfont = dict(size = 20)),
                yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM & Accuracy",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                        tickfont = dict(size = 20)),
                plot_bgcolor = 'rgb(255,255,255,0)')

            
                   
        my_text3 = np.round(DVM_value,decimals=4)
        my_text3 = my_text3.astype('<U59')
        for i in range(sample_len):
            for j in range(feature_len):
                my_text3[i,j] = 'Sample: ' + sample_seq_v[i,j] + '<br>' + 'Feature: ' + feature_seq_v[i,j] + '<br>' +'DVM: ' + my_text3[i,j]
        #my_text3 = my_text3.tolist()
        #print(my_text3)
        trace3D_DVM = go.Surface(x = feature_seq,y = sample_seq, z=DVM_value,text = my_text3,
                                 hoverinfo = 'text',
                                 colorscale='Viridis',showscale = False,name = 'DVM')
        layout3D_NoAccuracy = go.Layout(
                scene = dict (
                xaxis = dict(title = "Number of Features",titlefont=dict(size=17),gridcolor='rgb(230, 230, 230)',backgroundcolor='rgb(255, 255,255)'),
                yaxis = dict(title = "Number of Samples",titlefont=dict(size=17),gridcolor='rgb(230, 230, 230)',backgroundcolor='rgb(255, 255,255)'),
                zaxis = dict(title = "DVM",titlefont = dict(size=17),gridcolor='rgb(230, 230, 230)',backgroundcolor='rgb(255, 255,255)'),),)
        
        trace2D_sample_DVM = go.Scatter(x = sample_seq,
                    y = DVM_value[:,-1],
                    mode = 'lines',
                    name = 'DVM')
        
        
        
        layout2D_sample_NoAccuracy = go.Layout(
            title = go.layout.Title(x=0.5,text = 'DVM vs. Number of Samples',xanchor="center"),
            xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Samples",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                    tickfont = dict(size = 20)),
            yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                    tickfont = dict(size = 20)),
            plot_bgcolor = 'rgb(255,255,255,0)')
        
        trace2D_feature_DVM = go.Scatter(x = feature_seq,
                        y = DVM_value[-1,:],
                        mode = 'lines',
                        name = 'DVM')
        
        
        layout2D_feature_NoAccuracy = go.Layout(
            title = go.layout.Title(x=0.5,text = 'DVM vs. Feature Number',xanchor="center"),
            xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Number of Features",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                    tickfont = dict(size = 20)),
            yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "DVM",font = dict(size=25)),gridcolor='rgb(230,230,230)',
                                    tickfont = dict(size = 20)),
            plot_bgcolor = 'rgb(255,255,255,0)')
        
        if plot_type == '3D':
            if GiveAccuracy:
                plotly.offline.plot({"data": [trace3D_DVM,trace3D_Accuracy],"layout":layout3D}, 
                             auto_open=False, filename = filename + '3D'+'.html')
            else:
                plotly.offline.plot({"data": [trace3D_DVM],"layout":layout3D_NoAccuracy}, 
                             auto_open=False, filename = filename + '3D'+'.html')
        if plot_type == 'Both':
            if GiveAccuracy:
                plotly.offline.plot({"data": [trace3D_DVM,trace3D_Accuracy],"layout":layout3D}, 
                                     auto_open=False, filename = filename + '3D'+'.html')
                plotly.offline.plot({"data": [trace2D_sample_DVM,trace2D_sample_Accuracy],
                                     "layout":layout2D_sample}, auto_open=False, filename = filename + '2D_Sample' + '.html')
                plotly.offline.plot({"data": [trace2D_feature_DVM,trace2D_feature_Accuracy],
                                     "layout":layout2D_feature}, auto_open=False, filename = filename + '2D_feature' + '.html') 
            else:
                plotly.offline.plot({"data": [trace3D_DVM],"layout":layout3D_NoAccuracy}, 
                                     auto_open=False, filename = filename + '3D'+'.html')
                plotly.offline.plot({"data": [trace2D_sample_DVM],
                                     "layout":layout2D_sample_NoAccuracy}, auto_open=False, filename = filename + '2D_Sample' + '.html')
                plotly.offline.plot({"data": [trace2D_feature_DVM],
                                     "layout":layout2D_feature_NoAccuracy}, auto_open=False, filename = filename + '2D_feature' + '.html') 
        return {"Accuracy":Accuracy,"MI":MI,"Complexity" : Complexity,"DVM":DVM_value,
                "Sample Number":sample_seq, "Feature Number":feature_seq,
                "Confidence band":confidence_band,"I1":I1_matrix,"I2":I2_matrix,"I3":I3_matrix}