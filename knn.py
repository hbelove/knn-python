import csv
import random
import math

def loadDataset(filename, split_ratio=0.7):
	"""
	载入数据集并划分为训练集和测试集
	:param filename: 数据集文件
	:param split_ratio: 训练集所占比例，默认0.7
	:return: trainingSet,testSet

	NOTICE: 数据集大小m×n代表m个实例，n-1个特征，第n列应为类别标签
	"""
	trainingSet=[]
	testSet=[]
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    m,n=len(dataset),len(dataset[0]) 	#行，列
	    for x in range(m):     	
	        for y in range(n-1):		
	            dataset[x][y] = float(dataset[x][y])
	    random.shuffle(dataset)
	    trainingSet=dataset[:int(m*split_ratio)]
	    testSet=dataset[int(m*split_ratio):]
	return trainingSet,testSet


def euclideanDistance(instance1, instance2, feature_num):
	#计算两个实例间的欧式距离
	distance = 0
	for x in range(feature_num):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	#在训练集中找出距离测试实例最近的k个邻居
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=lambda d:d[1])
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	#找出距离最近的一个邻居
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=lambda d:d[1] , reverse=True)
	return sortedVotes[0][0]

def predict(k,trainingSet,testSet):
	#预测测试集的类别
	predictions=[]
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	#计算准确度
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return correct/float(len(testSet))
	
	
#测试	
if __name__=='__main__':
	# prepare data
	split_ratio = 0.7
	trainingSet,testSet=loadDataset('iris.csv', split_ratio) 
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))

	# generate predictions
	k = 3
	predictions=predict(k,trainingSet,testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy))
