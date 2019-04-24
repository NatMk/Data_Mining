import numpy as np
import matplotlib.pyplot as plt

Training_Data = [[["", "score for interview #1", "score for interview # 2"], ["result"]],
                 [[1.0, 1.00, 2.00], [0.0]],
                 [[1.0, 2.45, 2.53], [0.0]],
                 [[1.0, 1.42, 1.00], [0.0]],
                 [[1.0, 3.00, 2.08], [0.0]],
                 [[1.0, 3.71, 3.05], [0.0]],
                 [[1.0, 3.71, 5.55], [0.0]],
                 [[1.0, 2.00, 1.50], [0.0]],
                 [[1.0, 4.71, 5.55], [0.0]],
                 [[1.0, 1.00, 8.00], [0.0]],
                 [[1.0, 2.00, 4.00], [0.0]],
                 [[1.0, 1.50, 6.00], [0.0]],
                 [[1.0, 3.00, 6.50], [0.0]],
                 [[1.0, 6.00, 1.00], [0.0]],
                 [[1.0, 5.00, 2.00], [0.0]],

                 [[1.0, 4.03, 5.06], [1.0]],
                 [[1.0, 6.02, 7.01], [1.0]],
                 [[1.0, 7.04, 8.04], [1.0]],
                 [[1.0, 9.05, 7.04], [1.0]],
                 [[1.0, 8.00, 8.08], [1.0]],
                 [[1.0, 9.08, 4.08], [1.0]],
                 [[1.0, 7.00, 4.00], [1.0]],
                 [[1.0, 6.00, 9.00], [1.0]],
                 [[1.0, 9.00, 9.00], [1.0]],
                 [[1.0, 8.00, 1.00], [1.0]]]
ALPHA = 0.05
NUMB_OF_ITERATIONS = 750
#-----------------------------------------------------------------------------------------------------------------------
class LogisticRegression:

    def gradient_descent(self, x, y):
        w = np.ones((np.shape(x)[1], 1))
        for i in range(NUMB_OF_ITERATIONS):
            w = w - ALPHA * x.transpose() * (self.logistic(x * w) - y)
        return w
#-----------------------------------------------------------------------------------------------------------------------
    def classify(self, x, w):
        prob = self.logistic(sum(x * w))
        classification = "prob = "+prob.__str__()+" | classified as 0 (will not be hired)"
        if prob > 0.5:
            classification = "prob =" +prob.__str__()+" | classified as 1 (will be hired)"
        return classification
#-----------------------------------------------------------------------------------------------------------------------
    def logistic(self, ws):
        return 1.0/(1 + np.exp(-ws))
#-----------------------------------------------------------------------------------------------------------------------
    def plot(self, x, y, w):
        hiredInterview1Score = []; hiredInterview2Score = []
        notHiredInterview1Score = []; notHiredInterview2Score = []

        for i in range(0, x.__len__()):
            if y[i] == 0:
                notHiredInterview1Score.append(x[i][1])
                notHiredInterview2Score.append(x[i][2])
            else:
                hiredInterview1Score.append(x[i][1])
                hiredInterview2Score.append(x[i][2])

        ax = plt.figure().add_subplot(1, 1, 1)
        ax.scatter(hiredInterview1Score, hiredInterview2Score, s = 25, c = 'blue')
        ax.scatter(notHiredInterview1Score, notHiredInterview2Score, s = 25, c = 'grey')

        plt.xlabel(Training_Data[0][0][1]); plt.ylabel(Training_Data[0][0][2])

        interview1Score = np.arange(0.0, 10.0, 0.01)
        interview2Score = (-w[0] - w[1] * interview1Score) / w[2]

        ax.plot(interview1Score, interview2Score)
        plt.show()
#-----------------------------------------------------------------------------------------------------------------------
    def handle_command_line(self):
        flag = True
        while (flag):
            entry = input("> to classify new candidates enter scores "
                          "for interviews 1 & 2 separated by space (or exit):\n")
            if(entry == ''):
                continue
            if (entry != "exit"):
                score = entry.split()
                print(logisticRegression.classify([1.0, float(score[0]), float(score[1])], wArray))
            else:
                flag = False
#-----------------------------------------------------------------------------------------------------------------------
logisticRegression = LogisticRegression()
xArray = []; yArray = []
for i in range(1, Training_Data.__len__()):
        xArray.append([Training_Data[i][0][0], Training_Data[i][0][1], Training_Data[i][0][2]])
        yArray.append(Training_Data[i][1][0])
wArray = logisticRegression.gradient_descent(np.mat(xArray), np.mat(yArray).transpose())
print("close display window to proceed")
logisticRegression.plot(xArray, yArray, wArray.getA())
print(Training_Data[1][0][0])

print(Training_Data[1][0][1])

print(Training_Data[1][0][2])

print(Training_Data[1][1][0])

print(Training_Data.__len__())
#logisticRegression.handle_command_line()

