import numpy as np
from scipy import signal
import plotly.graph_objects as go
import function as fc
from sklearn.model_selection import train_test_split




# open file
dataRaw = []
DataFile = open("ecg.csv", "r")

while True:
    thisLine = DataFile.readline()
    if len(thisLine) == 0:
        break
    readData = thisLine.split(",")
    for position in range(len(readData)):
        readData[position] = float(readData[position])
    dataRaw.append(readData)
DataFile.close()
data = np.array(dataRaw)

labels = data[:, -1]
data = data[: , 0:-1]

#Filter
datad = []
iSampleRate = 250	#each sampled at 250 samples per second
# #进行带通滤波
b, a = fc.butterBandPassFilter(3 , 70, iSampleRate, order=4)
#进行带阻滤波
d, c = fc.butterBandStopFilter(48, 52, iSampleRate, order=2)
for i in range(len(data)):
    data[i]=signal.lfilter(b,a,data[i])
    data[i] = signal.lfilter(d, c, data[i])
    datad.append(data[i])
datad = np.array(datad)


# Data normalization
normaliseData = fc.normalize(datad)
# remove outlier
removedOutlier = fc.outliers(normaliseData)



train_data, test_data, train_labels,test_labels = train_test_split(normaliseData, labels, test_size=0.2,random_state=21)

X = train_data
y = train_labels

digits = 2
examples = y.shape[0]
y = y.reshape(1, examples)
Y = np.eye(digits)[y.astype('int32')]
Y = Y.T.reshape(digits, examples).T


X2 = test_data
y2 = test_labels

examples = y2.shape[0]
y2 = y2.reshape(1, examples)
Y2 = np.eye(digits)[y2.astype('int32')]
Y2 = Y2.T.reshape(digits, examples).T

svmF1,svmPrec,svmRecall,svmAcc,svmTime=fc.svmClassifier(train_data, test_data, train_labels,test_labels)
mlpF1,mlpPrec,mlpRecall,mlpAcc,mlpTime=fc.MLPclassifier(X,Y,X2,Y2)
cnnF1,cnnPrec,cnnRecall,cnnAcc,cnnTime=fc.CNNclassifier(X,Y,X2,Y2)
lstmF1,lstmPrec,lstmRecall,lstmAcc,lstmTime=fc.lstmAE(X,Y,X2,Y2)

fig = go.Figure(data=[go.Table(
    columnwidth = [800],
    header=dict(values=['','SVM','MLP','CNN','LSTM'],align='left'),
    cells = dict(values=[['F1 Score','Precision Score','Recall Score','Accuracy','Time Complexity'],
                        [svmF1,svmPrec,svmRecall,svmAcc,svmTime],
                         [mlpF1,mlpPrec,mlpRecall,mlpAcc,mlpTime],
                         [cnnF1,cnnPrec,cnnRecall,cnnAcc,cnnTime],
                         [lstmF1,lstmPrec,lstmRecall,lstmAcc,lstmTime],
                         ],fill=dict(color=['paleturquoise', 'white']),align='left')
        )
])
fig.write_image("Table.pdf")