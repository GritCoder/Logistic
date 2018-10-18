import math
from numpy import *
def loadSet():
    dataMat = []
    labelMat = []
    fr = open("testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])#为什么要多增加一列1 书上解释是方便计算 我不是很理解
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
def sigmoid(inX):
    try:
        ans = 1.0/(1 + math.exp(-inX))
    except OverflowError:
        ans = float('inf')
    return ans
def gradAscent(dataMatIn,classLabels):#递归求解回归系数 它用于确定不同数据的分界线
    dataMatrix = mat(dataMatIn)#将数组转换为矩阵
    labelMat = mat(classLabels).transpose()#标签矩阵，
    m, n = shape(dataMatrix)#获取矩阵的行数和列数
    alpha = 0.001#迭代的步长
    maxCycle = 500
    weights = ones((n,1))#初始回归系数设置为1 注意回归系数是个向量
    for k in range(maxCycle):#设置的是迭代500次结束
        h = sigmoid(sum(dataMatrix * weights))#sigmoid函数的参数是标量还是矢量，现在可以确认是标量
        error = labelMat - h#整个矩阵相乘需要消耗很多计算力
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights
def improve_gradAscent(dataMatIn,classLabels,numIter=150):#上面的迭代算法效率太低，每次迭代都要遍历所有的数据集计算
    m,n = shape(dataMatIn)
    Weights = ones(n)  # 细节，产生n个1的列表
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01#每次迭代时就调整，可以缓解数据波动
            randomIndex = int(random.uniform(0,len(dataIndex)))#均匀分布中随机选一个值
            h = sigmoid(sum(dataMatIn[randomIndex] * Weights))
            #error = classLabels[randomIndex] - h#此处不再是矩阵相乘  (注，如果进行病马预测，不要加error，因为不满足矩阵相乘的规则，index也越界了)
            Weights = Weights + alpha * h * dataMatIn[randomIndex]
            del(dataIndex[randomIndex])#从随机列表中删除已经选过的值，避免重复迭代
    return Weights#随机梯度上升 求最大值 该算法收敛快 效率高
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
def colicTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split()
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = improve_gradAscent(array(trainingSet),trainingLabels,500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split()
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is : %f" % (errorRate))
    return errorRate
def muliTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error is : %f " % (numTests,errorSum/float(numTests)))
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMatIn, classLabels = loadSet()
    dataArr = array(dataMatIn)
    n = shape(dataArr)[0]#获得数据集的行数
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):#对于每一个数据实例
        if int(classLabels[i]) == 1:#分类1
            xcord1.append(dataArr[i,1])#X1 不能从0开始 因为第0列是额外添加的全1列
            ycord1.append(dataArr[i,2])#X2
        else: #分类0
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()#创建一个画板
    ax = fig.add_subplot(111)#子画板 # 111 表示1x1网格的第一个子图 234表示2x3网格的第4个子图
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')#绘制散点图
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1] * x)/weights[2]
    ax.plot(x,y)#上面是设置 x y轴坐标的刻度
    plt.xlabel('X1')#设置x y 的标签
    plt.ylabel('X2')
    plt.show()
if __name__ == "__main__":
    dataMatIn,classLabels = loadSet()
    #print(gradAscent(dataMatIn,classLabels))#第一行不是，从第二行 第三行开始算
    #plotBestFit(improve_gradAscent(array(dataMatIn),classLabels,500))#getA()函数把矩阵转化为数组
    #print(improve_gradAscent(array(dataMatIn),classLabels,500))
    muliTest()#若输出的错误率不变，则说明回归系数可能已经收敛