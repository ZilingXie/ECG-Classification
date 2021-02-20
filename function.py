from scipy import signal
import numpy as np
from sklearn import svm, metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from time import process_time
from keras.models import Sequential
from keras import layers,Input,Model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from tensorflow.python.keras.utils.np_utils import to_categorical


def butterBandPassFilter(lowcut, highcut, samplerate, order):

    semiSampleRate = samplerate*0.5
    low = lowcut / semiSampleRate
    high = highcut / semiSampleRate
    b,a = signal.butter(order,[low,high],btype='bandpass')
    # print("bandpass:","b.shape:",b.shape,"a.shape:",a.shape,"order=",order)
    # print("b=",b)
    # print("a=",a)
    return b,a

def butterBandStopFilter(lowcut, highcut, samplerate, order):
    
    semiSampleRate = samplerate*0.5
    low = lowcut / semiSampleRate
    high = highcut / semiSampleRate
    from scipy import signal
    b,a = signal.butter(order,[low,high],btype='bandstop')
    # print("bandstop:","b.shape:",b.shape,"a.shape:",a.shape,"order=",order)
    # print("b=",b)
    # print("a=",a)
    return b,a

def normalize(data):
    normalizeData = data.copy()

    rows = data.shape[0]
    cols = data.shape[1]

    for j in range(cols):
        maxElement = np.amax(data[:, j])
        minElement = np.amin(data[:, j])

        for i in range(rows):
            normalizeData[i][j] = (data[i][j] - minElement) / (maxElement - minElement)
    return normalizeData

# correct outliers
def outliers(data):
    correctData = data.copy()
    rows = data.shape[0]
    cols = data.shape[1]
    for i in range(rows):
        for j in range(cols):
            if data[i][j] < -1:
                correctData[i][j] = -1
            elif data[i][j] > 1:
                correctData[i][j] = 1

    return correctData

def svmClassifier (train_data, test_data, train_labels,test_labels):
    t1 = process_time()
    model = svm.SVC(decision_function_shape='ovo',kernel='rbf')
    model.fit(train_data, train_labels)
    model.score(train_data, train_labels)
    pred = model.predict(test_data)

    # get confusion matrix that has the highest accuracy
    array = metrics.confusion_matrix(test_labels, pred)
    f1 = f1_score(test_labels,pred, average='macro')
    precision = precision_score(test_labels, pred, average='macro')
    recall = recall_score (test_labels, pred, average='macro')
    accuracy = accuracy_score(test_labels,pred)
    metrics.plot_confusion_matrix(model, test_data, test_labels)
    # plt.title("SVM Classifier Sigmoid")
    # plt.savefig("SVMsigmoid.pdf")
    # plt.close()
    t2 = process_time()
    time = t2 - t1
    return round(f1,5), round(precision,5), round(recall,5), accuracy, time



def MLPclassifier (train_data, train_labels,test_data,test_labels):
    t1 = process_time()
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(140,)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    learning_rate = 0.01
    sgd = SGD(lr=learning_rate)
    model.compile(optimizer=sgd, loss='mse')

    model.fit(train_data,train_labels,epochs=100,batch_size=10,verbose=1)
    pred = model.predict(test_data)
    pred = np.argmax(pred,axis=1)
    pred=to_categorical(pred)

    f1 = f1_score(test_labels,pred, average='macro')
    precision = precision_score(test_labels,pred,average='macro')
    recall = recall_score(test_labels,pred,average='macro')
    accuracy = accuracy_score(test_labels,pred)
    t2 = process_time()
    time = t2 - t1
    return round(f1,5), round(precision,5), round(recall,5), accuracy, time

def CNNclassifier (train_data, train_label,test_data, test_labels):
    t1 = process_time()
    inputECG = Input(batch_shape=(None, 140, 1))
    x = layers.Conv1D(64, 3, activation='relu', padding='valid')(inputECG)
    x1 = layers.MaxPooling1D(2)(x)
    x2 = layers.Conv1D(32, 3, activation='relu', padding='valid')(x1)
    x3 =layers.MaxPooling1D(2)(x2)
    flat = layers.Flatten()(x3)
    encoded = Dense(32, activation='relu')(flat)

    model_encoder = Model(inputECG, encoded)
    model = Sequential()
    model.add(model_encoder.layers[0])
    model.add(model_encoder.layers[1])
    model.add(model_encoder.layers[2])
    model.add(model_encoder.layers[3])
    model.add(model_encoder.layers[4])
    model.add(model_encoder.layers[5])
    model.add(model_encoder.layers[6])
    #
    model.add(layers.Dense(2, activation='softmax'))
    #
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    train_data = train_data[...,None]
    test_data = test_data[...,None]

    model.fit(train_data,train_label,epochs=100, batch_size=100)
    pred = model.predict(test_data)
    pred = np.argmax(pred, axis=1)
    pred = to_categorical(pred)

    f1 = f1_score(test_labels, pred, average='macro')
    precision = precision_score(test_labels, pred, average='macro')
    recall = recall_score(test_labels, pred, average='macro')
    accuracy = accuracy_score(test_labels, pred)
    t2 = process_time()
    time = t2 - t1
    return round(f1,5), round(precision,5), round(recall,5), accuracy, time


def lstmAE(train_data, train_label,test_data, test_labels):
    t1 = process_time()
    inputs = Input(shape=(140, 1))
    encoded = layers.LSTM(64)(inputs)
    model_lstmAE_encoder = Model(inputs, encoded)

    model = Sequential()
    model.add(model_lstmAE_encoder.layers[0])
    model.add(model_lstmAE_encoder.layers[1])
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    train_data = train_data[..., None]
    test_data = test_data[..., None]
    model.fit(train_data, train_label, epochs=10, batch_size=10)
    pred = model.predict(test_data)
    pred = np.argmax(pred, axis=1)
    pred = to_categorical(pred)

    f1 = f1_score(test_labels, pred, average='macro')
    precision = precision_score(test_labels, pred, average='macro')
    recall = recall_score(test_labels, pred, average='macro')
    accuracy = accuracy_score(test_labels, pred)
    t2 = process_time()
    time = t2 - t1
    return round(f1,5), round(precision,5), round(recall,5), accuracy, time
