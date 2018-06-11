import numpy
import math


def relu(x):
    return max(0,x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def loss_abs_diff(out,act):
    s=0
    for i in range(len(out)):
        s+=abs(act[i]-out[i])
    return s

def softmax(inputs):

    return numpy.exp(inputs) / float(sum(numpy.exp(inputs)))

class NeuroES(object):
    def __init__(self,input,output,out_fun):
        self.weights=[]
        self.input=input
        self.output=output
        self.completed=0
        self.node_list=[]
        self.layer_count=0
        self.weight_count=0
        self.out_f=out_fun
    def add_layer(self,nodes,act_f):
        if nodes<=0:
            print("Nodes should be atleast one or more than one!")
            return
        if self.completed==1:
            print("ERROR, Neural Network has been completed and cannot be further modified")
            return
        self.layer_count+=1
        self.node_list.append([nodes,act_f])
    def completed_network(self):
        ini=self.input
        for i in range(len(self.node_list)):
            k=self.node_list[i][0]
            self.weight_count+=k*ini+k
            ini=k
        self.weight_count+=self.output*ini+self.output
        self.completed=1
    def set_weights(self,wg):
        if len(wg)==self.weight_count:
            self.weights=wg
        else:
            print("ERROR: Given weights are not enough or too much!")
            return
    def get_weights(self):
        return self.weights
    def get_weight_count(self):
        return self.weight_count
    def clear_weights(self):
        self.weights=[]

    def init_rand_weights(self):
        for i in range(self.weight_count):
            self.weights.append(float(numpy.random.uniform(size=1)*2-1))
        return self.weights
    def evaluate(self,inputs):
        if self.input!=len(inputs):
            print("Input size is not correct")
            return
        out=[]
        prev=inputs
        w_i=0
        for i in range(self.layer_count):
            current=[]
            f=self.node_list[i][1]
            l=self.node_list[i][0]
            for j in range(l):
                w_array=self.weights[w_i:w_i+len(prev)]
                arg2=numpy.array(w_array)
                arg1=numpy.array(prev)
                w_i+=len(prev)
                arg3=numpy.matmul(arg1,arg2)+self.weights[w_i]
                w_i+=1
                arg3=f(arg3)
                current.append(arg3)
            prev=current
        for i in range(self.output):
            w_array=self.weights[w_i:w_i+len(prev)]
            arg2 = numpy.array(w_array)
            arg1 = numpy.array(prev)
            w_i += len(prev)
            arg3 = numpy.matmul(arg1, arg2)+self.weights[w_i]
            w_i+=1

            out.append(arg3)

        out=self.out_f(out)
        return out
    #def train(self,inp_array,cr_rate,mut_rate,pop_size):








