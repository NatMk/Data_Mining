import numpy as np
import matplotlib.pyplot as plt

#List to hold the data generated randomly
Training_Data= []

#List to hold the data generated randomly
Testing_Data = []

#number of data to be generated
n = 1000

#number of data to be generated for testing
n_test = 500

mu1 = [1,0]
mu2 = [0, 1.5]
prob = 0

#Standard Deviations
sd1= [[1, 0.75], [0.75, 1]]
sd2 = [[1, 0.75], [0.75, 1]]

#Training label and total iterations
ALPHA = 1
NUMB_OF_ITERATIONS = 1000

#generating 1000 data in two sets
Training_Data.append(np.random.multivariate_normal(mu1, sd1, n))
Training_Data.append(np.random.multivariate_normal(mu2, sd2, n))

#generating 500 data in two sets
Testing_Data.append(np.random.multivariate_normal(mu1, sd1, n_test))
Testing_Data.append(np.random.multivariate_normal(mu2, sd2, n_test))

#Bios for Training and testing data
Training_Data_set1 = np.insert(Training_Data[0],0,1.0, axis = 1)
Training_Data_set2 = np.insert(Training_Data[1],0,1.0, axis = 1)

Testing_Data_set1 = np.insert(Testing_Data[0], 0, 1.0, axis= 1)
Testing_Data_set2 = np.insert(Testing_Data[1], 0, 1.0, axis= 1)

#labels
label_1 = [0.0]; label_2 = [1.0]
Temp_set1 = []; Temp_set2 = []
Temp_test1 = []; Temp_test2 = []

x = Training_Data_set1.tolist(); y = Training_Data_set2.tolist()

x_test = Testing_Data_set1.tolist(); y_test = Testing_Data_set2.tolist()

Temp_set1.append(x); Temp_set2.append(y)
Temp_test1.append(x_test); Temp_test2.append(y_test)

Another_Temp_1 = []; Another_Temp_2 = []
Another_Test_Temp1 = []; Another_Test_Temp_2 = []

#add labels to the data_sets
for i in range(1000):
   Another_Temp_1.append([Temp_set1[0][i],label_1])
   Another_Temp_2.append([Temp_set1[0][i], label_2])

for i in range(500):
    Another_Test_Temp1.append([Temp_test1[0][i], label_1])
    Another_Test_Temp_2.append([Temp_test2[0][i],label_2])

#Concatenate the two data sets
Total_Data = Another_Temp_1 + Another_Temp_2
Total_Test_Data = Another_Test_Temp1 + Another_Test_Temp_2
#--------------------------------------------------------------------------------------------
class LogisticRegression:

    def gradient_descent(self, x, y):
        #inital weight w = [1, 1, 1]
        w = np.ones((np.shape(x)[1], 1))
        for i in range(NUMB_OF_ITERATIONS):
            w = w - ALPHA * x.transpose() * (self.logistic(x * w) - y)
        return w
#-----------------------------------------------------------------------------------------------------------------------
    def classify(self, x, w):
        global prob
        prob = self.logistic(sum(x * w))
        classification = "probability = "+prob.__str__()+" | classified as 0 (will not be considered)"
        if prob >= 0.5:
            classification = "probability =" +prob.__str__()+" | classified as 1 (will be considered)"
        return classification
#-----------------------------------------------------------------------------------------------------------------------
    def logistic(self, ws):
        return 1.0/(1 + np.exp(-ws))
#-----------------------------------------------------------------------------------------------------------------------
    def plot(self, x, y, w):
        considered_Data1 = []; considered_Data2 = []
        not_considered_Data1 = []; not_considered_Data2 = []

        for i in range(0, x.__len__()):
            if y[i] == 0:
                not_considered_Data1.append(x[i][1])
                not_considered_Data2.append(x[i][2])
            else:
                considered_Data1.append(x[i][1])
                considered_Data2.append(x[i][2])

        graph = plt.figure().add_subplot(1, 1, 1)
        graph.scatter(considered_Data1, considered_Data2, s = 25, c = 'blue')
        graph.scatter(not_considered_Data1, not_considered_Data2, s = 25, c = 'red')

        plt.xlabel('Data Set 1 Values'); plt.ylabel('Data Set 2 Values')

        Data1Val = np.arange(0.0, 10.0, 0.01)
        Data2val = (-w[0] - w[1] * Data1Val) / w[2]

        graph.plot(Data1Val, Data2val)
        plt.show()
#-----------------------------------------------------------------------------------------------------------------------
    def _command_line(self):
        flag = True
        while (flag):
            entry = input("> Enter data type 1 & 2 separated by space to classify data "
                          "or type exit to end program:\n")
            if(entry == ''):
                continue
            if (entry != "exit"):
                User_Data = entry.split()
                print(logisticRegression.classify([1.0, float(User_Data[0]), float(User_Data[1])], wArray))
            else:
                flag = False
#-----------------------------------------------------------------------------------------------------------------------
    def accuracy(self, prob, actual_labels):
        global val
        for i in range(0, Total_Test_Data.__len__()):
            val = prob - actual_labels[i] * 100
        return val
#-----------------------------------------------------------------------------------------------------------------------
logisticRegression = LogisticRegression()
xArray = []; yArray = []
for i in range(0, Total_Test_Data.__len__()):
      xArray.append([Total_Test_Data [i][0][0],Total_Test_Data [i][0][1], Total_Test_Data [i][0][2]])
      yArray.append(Total_Test_Data [i][1][0])
wArray = logisticRegression.gradient_descent(np.mat(xArray), np.mat(yArray).transpose())
print("close display window to check for other data sets")
logisticRegression.accuracy(prob, yArray)
logisticRegression.plot(xArray, yArray, wArray.getA())
logisticRegression._command_line()
#-----------------------------------------------------------------------------------------------------------------------