"""
 Name: Your Name
 Assignment: Lab 4 - Decision Tree
 Course: CS 330
 Semester: Fall 2021
 Instructor: Dr. Cao
 Date: the current date
 Sources consulted: any books, individuals, etc consulted

 Known Bugs: description of known bugs and other program imperfections

 Creativity: anything extra that you added to the lab

 Instructions: After a lot of practice in Python, in this lab, you are going to design the program for decision tree and implement it from scrath! Don't be panic, you still have some reference, actually you are going to translate the JAVA code to Python! The format should be similar to Lab 2!

"""
import sys
import argparse
import math
import os

# You may need to define the Tree node and add extra helper functions here


def DTtrain(data, model):
    """
    This is the function for training a decision tree model
    """

    class DTTrain:
        def __init__(self):
            self.datamap = {}
            self.attvalues = {}
            self.atts = []
            self.numAtts = 0
            self.numClasses = 0
            self.root = None
    
        def readFile(self, inFile, percent):
            try:
                self.datamap = {}

                with open(inFile, 'r') as theFile:
                    attline = inFile.readline()[1:]
                    self.atts = attline.split("|")
                    self.numAtts = len(self.atts) - 1
                    self.attvalues = {a: [] for a in self.atts}

                    index = 0
                    for line in theFile:
                        dataclass, *values = line.split()
                        if dataclass not in self.attvalues[self.atts[0]]:
                            self.numAtts[self.atts[0]].append(dataclass)
                    
                        if dataclass not in self.datamap:
                            self.datamap[dataclass] = []
                            a = self.datamap[dataclass]
                            datapoint = []
                            for i in range(self.numAtts):
                                val = values[i]
                                datapoint.append(val)
                                arr = self.attvalues[self.atts[i+1]]
                                if val not in arr:
                                    arr.append(val)
                            
                            
                            if index % 100 < percent:
                                a.append(datapoint)
                        
                            index += 1
                    
                        self.numClasses = len(self.datamap.keys())
            except IOError as ioe:
                print("Error reading:", ioe)
                exit(0)

        def buildTree(self):
            self.root = TreeNode(None)
            curratts = [att for att in self.atts[1:]]
            self.root = self.buildTreeNode(None, curratts, self.datamap)
        
        def buildTreeNode(self, par, curratts, nodeD):
            curr = TreeNode(par)
            min = 1
            minatt = None

            for i in range(self.numAtts):
                att = curratts[i]
                if att is not None:
                    vals = self.attvalues[att]

                    part = [[0] * self.numClasses for _ in range(len(vals))]
                    for j in range(self.numClasses):
                        outcome = self.attvalues[self.atts[0][j]]
                        l = nodeD[outcome]
                        for l2 in 1:
                            part[vals.index(12[i])[j]]+= 1

                    entropy = self.partE(part)
                    if (entropy < min):
                        min = entropy
                        minatt = att
                
                if minatt is None:
                    maxCount = 0
                    maxClass = "Unknown"
                    for j in range(self.numClasses):
                        outcome = self.attV[self.atts[0][j]]
                        if len(nodeD[outcome]) >= maxCount:
                            maxCount = len(nodeD[outcome])
                            maxClass = outcome
                    
                    curr.returnV = maxClass
                    return curr
                
                curr.att = minatt
                attIndex = curratts.index(minatt)
                curratts[attIndex] = None

                for i in self.attV[minatt]:
                    temp = {}
                    for j in range(self.numClasses):
                        outcome = self.attV[self.atts[0][j]]
                        trimList = [l2 for l2 in nodeD[outcome] if l2[attIndex] == v]
                        temp[outcome] = trimList
                    
                    print(v + "---> ")
                    curr.child[v] = self.buildTreeNode(curr, curratts, temp)

                
                curratts[attIndex] = minatt
                return curr
            
            def partEnt(self, partition):
                totalE = 0
                total = 0

                for i in range(len(part)):
                    n = sum(part[i])
                    total += n
                    totalE += n * self.ent(part[i])
                
                return totalE / total
            
            def ent(self, class_count):
                total = sum(class_count)

                if total == 0:
                    return 0
                
                ent_sum = 0
                for i in range(len(class_count)):
                    ent_sum -= (class_count[i] / total) * self.log2(class_count[i] / total)

                return ent_sum
            
        @staticmethod
        def log2(x):
            if x == 0:
                return 0
            return math.log(x) / math.log(2)
        
        def solveModel(self, theFile1):
            try:
                with open(theFile1, 'w') as outFile:
                    for i in range(self.numAtts):
                        outFile.write(self.atts[i+1] + " ")
                    outFile.write("\n")

                    self.writeNode(outFile, self.root)

            except IOError as ioe:
                print("Error writing file:", ioe)
                exit(1)
        
        def writeNode(self, outFile, curr):
            if curr.returnV is not None:
                outFile.write("[" + curr.returnV + "]")
                return
            
            outFile.write(curr.attr + " ( ")
            for k, value in curr.children.items():
                outFile.write(k + " ")
                self.writeN(outFile, value)

            outFile.write(" ) ")

        def main(self, args):
            if len(args) >= 2:
                inFile = args[0]
                theFile = args[0]
                if len(args) == 3:
                    self.readFile(inFile, int(args[2]))
                else:
                    self.readFile(inFile, 100)
                
                self.buildTree()

                self.solveModel(theFile)
            
            else:
                print("Please format input DTTrain 'trainingdata' 'modelfile' [percentage of data]")

    class TreeNode:
        def __init__(self, par):
            self.parent = par
            self.child = {}
            self.attr = "None"
            self.returnV = None

def DTpredict(data, model, prediction):
    """
    This is the main function to make predictions on the test dataset. It will load saved model file,
    and also load testing data TestDataNoLabel.txt, and apply the trained model to make predictions.
    You should save your predictions in prediction file, each line would be a label, such as:
    1
    0
    0
    1
    ...
    """
    # implement your code here
    class DTPredict:
        class TreeNode:
            def __init__(self):
                self.root = None
                self.attributes = []
                self.predict = []
    
    def readFile(self, theFile):
        try : 
            with open(theFile, 'r') as infile:
                attr = infile.readline().split()
                self.attribute = attr
                self.root = self.readTree(infile)
        except IOError as ioe:
            print("Error, please select another file")
            os._exit(1)

def readTree(self, infile):
    val = infile.readline().strip()
    if val[0] == '[':
        return TreeNode(None, None, val[1:-1])
    node = TreeNode(val, {}, None)
    infile.read(1)
    val = infile.readline().strip()
    while val != ')':
        node.child[val] = self.readTree(infile)
        i = infile.readline().strip()
    return node

def predictModel(self, testFile):
    try:
        self.predict = None
        with open(testFile, 'r') as theFile:
            for line in theFIle:
                d = line.strip().split()[1:]
                p = self.theTree(self.root, d)
                self.predict.append(p)
    except IOError as ioe:
        print("Error reading test file: {e}")
        ioe._exit(1)

def theTree(self, node, data):
    if node.value is not None:
        return node.value
    attr = node.attribute
    val = data[self.attribute.index(attr)]
    x = node.child[val]
    return self.theTree(x, data)

def thePredict(self, outFile):
    try:
        with open(outFile, 'w') as o:
            for p in self.predict:
                p.write(p + '\n')
    except IOError as ioe:
        print("Error writing to file: {e}")

def runProgram(self, args):
    if len(args) == 3:
        self.readFile(args[1])
        print("Success")
        self.predictModel(args[0])
        print("Predict complete")
        self.thePredict(args[2])
        print("Predictions saved")
    else:
        print("Please format: ")

class TreeNode:
    def __init__(self, atrr, ch, returnV):
        self.attribute = atrr
        self.child = ch
        self.value = returnV


def EvaDT(predictionLabel, realLabel, output):
    """
    This is the main function. You should compare line by line,
     and calculate how many predictions are correct, how many predictions are not correct. The output could be:

    In total, there are ??? predictions. ??? are correct, and ??? are not correct.

    """
    correct,incorrect, length = 0,0,0
    with open(predictionLabel,'r') as file1, open(realLabel, 'r') as file2:
        pred = [line for line in file1]
        real = [line for line in file2]
        length = len(pred)
        for i in range(length):
            if pred.pop(0) == real.pop(0):
                correct += 1
            else:
                incorrect += 1
    Rate = correct/length

    result = "In total, there are "+str(length)+" predictions. "+str(correct)+" are correct and "+ str(incorrect) + " are incorrect. The percentage is "+str(Rate)
    with open(output, "w") as fh:
        fh.write(result)

def main():
    options = parser.parse_args()
    mode = options.mode       # first get the mode
    print("mode is " + mode)
    if mode == "T":
        """
        The training mode
        """
        inputFile = options.input
        outModel = options.output
        if inputFile == '' or outModel == '':
            showHelper()
        DTtrain(inputFile, outModel)
    elif mode == "P":
        """
        The prediction mode
        """
        inputFile = options.input
        modelPath = options.modelPath
        outPrediction = options.output
        if inputFile == '' or modelPath == '' or outPrediction == '':
            showHelper()
        DTpredict(inputFile,modelPath,outPrediction)
    elif mode == "E":
        """
        The evaluating mode
        """
        predictionLabel = options.input
        trueLabel = options.trueLabel
        outPerf = options.output
        if predictionLabel == '' or trueLabel == '' or outPerf == '':
            showHelper()
        EvaNB(predictionLabel,trueLabel, outPerf)
    pass

def showHelper():
    parser.print_help(sys.stderr)
    print("Please provide input augument. Here are examples:")
    print("python " + sys.argv[0] + " --mode T --input TrainingData.txt --output DTModel.txt")
    print("python " + sys.argv[0] + " --mode P --input TestDataNoLabel.txt --modelPath DTModel.txt --output TestDataLabelPrediction.txt")
    print("python " + sys.argv[0] + " --mode E --input TestDataLabelPrediction.txt --trueLabel LabelForTest.txt --output Performance.txt")
    sys.exit(0)


if __name__ == "__main__":
    #------------------------arguments------------------------------#
    #Shows help to the users                                        #
    #---------------------------------------------------------------#
    parser = argparse.ArgumentParser()
    parser._optionals.title = "Arguments"
    parser.add_argument('--mode', dest='mode',
    default = '',    # default empty!
    help = 'Mode: T for training, and P for making predictions, and E for evaluating the machine learning model')
    parser.add_argument('--input', dest='input',
    default = '',    # default empty!
    help = 'The input file. For T mode, this is the training data, for P mode, this is the test data without label, for E mode, this is the predicted labels')
    parser.add_argument('--output', dest='output',
    default = '',    # default empty!
    help = 'The output file. For T mode, this is the model path, for P mode, this is the prediction result, for E mode, this is the final result of evaluation')
    parser.add_argument('--modelPath', dest='modelPath',
    default = '',    # default empty!
    help = 'The path of the machine learning model ')
    parser.add_argument('--trueLabel', dest='trueLabel',
    default = '',    # default empty!
    help = 'The path of the correct label ')
    if len(sys.argv)<3:
        showHelper()
    main()
