import pandas as pd
import numpy as np
import argparse 
from sklearn.model_selection import train_test_split

import keras
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.python.keras.callbacks import History

def data_preprocessing(raw_data):
        
    #team = raw_data['team']
    #data = 0
    
    #for i in range (len(raw_data['ts'])):
        #if team[i]=='team_ball':
            #team[i]='team_2'
        
    #team = np.array(team)
    #team_processed = np.zeros([len(team)])
    
    #for i in range (len(team)): 
        #team_processed[i] = int(team[i][5])
            
    num_time_steps = raw_data['ts'].shape[0]
    data = np.concatenate((raw_data['x'],raw_data['y'],raw_data['z']))
    data = data.reshape(3, num_time_steps)
    player=[]
    for i in range (11):
        player1 = data[:, i:num_time_steps:11]
        player.append(player1)
    
    data =  np.array(player)
    
    return data 

def create_dataset(num_past_steps, raw_data):
    """
    Arguments:
    num_past_steps: number of steps needed to predict the next time step.
    data: raw data to be preprocessed.
    Returns: time series of dimension = () , labels of dimension = ()
    """
    data=data_preprocessing(raw_data)
    data=data.reshape(33, 95325)
    x = []
    y = []
    for i in range(95224):                           #num_time_steps/11 = 95325  
        #print(i)
        p = np.matrix.transpose(data[:,i:i+num_past_steps])    #training data
        x.append(p)
        q = np.matrix.transpose(data[:,i+num_past_steps])    #test data
        y.append(q)
        
    return np.array(x), np.array(y).reshape(len(y), 1, 33)

def RNN(past_time_steps):
    """
    Arguments:
    past_time_steps: number of steps needed to predict the next time step.
    Returns: Model architecture 
    
    """
    inputs = Input(name='inputs',shape=(past_time_steps, 33))  
    #layer = BatchNormalization()(inputs)
    layer = LSTM(64, return_sequences=True)(inputs)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)   
    #layer = LSTM(32, return_sequences=False)(layer)
    layer = Dense(128,name='FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(33,name='out_layer')(layer)       
    model = Model(inputs=inputs,outputs=layer)
    return model
    
######Test how well the trained model fits to new training datasets###########
def test_variance(model, data1, data2, num_past_steps, begin, end):
    """
    Arguments:
    model: trained model.
    data1, data2: time series different from the time series the model was trained on.
    begin: beginning step of the time series.
    end: ending step of the time series.
    num_past_steps: number of steps needed to predict the next time step.
    """
    
    print('Data processing of "uta" and "gsw" data')
    data_X_uta, label_uta = create_dataset(num_past_steps, data1)
    #print(data_X_uta[0])
    #print(label_uta[0])
    X_test_uta, y_test_uta = data_X_uta[begin:end], label_uta[begin:end]

    data_X_gsw, label_gsw = create_dataset(num_past_steps, data2)
    #print(data_X_gsw[0])
    #print(label_gsw[0])
    X_test_gsw, y_test_gsw = data_X_gsw[begin:end], label_gsw[begin:end]

    print('Evaluation of the trained model on different time series')
    print('Test statistics for UTA:')
    model.evaluate(X_test_uta, y_test_uta, batch_size=64)
    print('Test statistics for GSW:')
    model.evaluate(X_test_gsw, y_test_gsw, batch_size=64)



if __name__=="__main__":
        
    parser = argparse.ArgumentParser(description="Training_and_evaluation")
    parser.add_argument("-episodes", type=int, default=50, help="number of episodes")
    parser.add_argument("-num_past_steps", type=int, default=25, help="number of time steps in the past to forecast next step")
    parser.add_argument("-batch_size", type=int, default=64, help="batch size for training and testing")
    parser.add_argument("-begin", type=int, default=0, help="beginning time step for testing trained model on different time series")
    parser.add_argument("-end", type=int, default=1000, help="ending time step for testing trained model on different time series")
    parser.add_argument("-load_model", type=int, default=1, help="flag for loading pretrained model. 0:False, 1:True")  
    args=parser.parse_args()
    print(args.episodes)
    print(args.load_model)
    flag = args.load_model
    raw_data_bos=pd.read_csv('2019040102_nba-bos_TRACKING.csv')
    
    #Create dataset
    data_X_bos, label_bos = create_dataset(args.num_past_steps, raw_data_bos) #can be experimented with any number of past time steps
    #print(data_X_bos[0])
    #print(label_bos[0])
    X_train_bos, X_test_bos, y_train_bos, y_test_bos = train_test_split(data_X_bos, label_bos, test_size=0.10, shuffle='True')     
       
    if flag==0:
        #Compile the model
        model = RNN(args.num_past_steps)
        model.summary(args.num_past_steps)
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        model.fit(X_train_bos, y_train_bos, batch_size=args.batch_size, epochs=args.episodes, validation_split=0.2, shuffle='True')
    if flag==1: 
        model = keras.models.load_model("my_model"+str(args.num_past_steps))
        print('Pre-trained model loaded')
        
    print('Evaluation on the time series data the model has been trained on (nba-bos)')
    model.evaluate(X_test_bos, y_test_bos, batch_size=args.batch_size)
        
    raw_nba_uta = pd.read_csv('2019040129_nba-uta_TRACKING.csv')
    raw_nba_gsw = pd.read_csv('2019051610_nba-gsw_TRACKING.csv')

    test_variance(model, raw_nba_uta, raw_nba_gsw, args.num_past_steps, args.begin, args.end)
        
        