import math
import numpy
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
import io
import random

__errors__= [];
__avg__=0;
__maxVal__=0;


def hypothesis(params, samples):
  acum = 0
  for i in range(len(params)):
    acum = acum + params[i] * samples[i]
  acum = acum * (-1)
  acum = 1 / (1 + math.exp(acum))
  return acum

def show_errors(params, samples, y):
    global __errors__
    error_acum =0
    error = 0
    for i in range(len(samples)):
    	hyp = hypothesis(params,samples[i])

    	if(y[i] == 1):
    		if(hyp ==0):
    			hyp = .0001;
    		error = (-1)*math.log(hyp);
    	if(y[i] == 0):
    		if(hyp ==1):
    			hyp = .9999;
    		error = (-1)*math.log(1-hyp);
    	print( "error %f  hyp  %f  y %f " % (error, hyp,  y[i]))
    	error_acum = error_acum +error
    mean_error_param=error_acum/len(samples);
    __errors__.append(mean_error_param)
    return mean_error_param;

def gradDesc(params, samples, y, alfa):
    temp = list(params)
    general_error = 0
    for j in range(len(params)):
        acum = 0
        error_acum = 0
        for i in range(len(samples)):
            error = hypothesis(params,samples[i]) - y[i]
            acum = acum + error*samples[i][j]
            temp[j] = params[j] - alfa*(1/len(samples))*acum
    return temp

def scaling(samples):
  global __avg__
  global __maxVal__
  acum = 0
  samples = numpy.asarray(samples).T.tolist()
  for i in range(1,len(samples)):
    for j in range(len(samples[i])):
      acum=+ samples[i][j]
    avg = acum/(len(samples[i]))
    max_val = max(samples[i])
    print("To scale feature %i use (Value -  avg[%f])/ maxval[%f]" % (i, avg, max_val))
    for j in range(len(samples[i])):
      #print(samples[i][j])
      samples[i][j] = (samples[i][j] - avg)/max_val  #Mean scaling
  __avg__ = avg
  __maxVal__ = max_val
  return numpy.asarray(samples).T.tolist()

def scalingTest(samples, avg, max_val):
    samples = numpy.asarray(samples).T.tolist()
    for i in range(1,len(samples)):
        for j in range(len(samples[i])):
          #print(samples[i][j])
          samples[i][j] = (samples[i][j] - avg)/max_val  #Mean scaling
    return numpy.asarray(samples).T.tolist()

def yPredict(params, samples):
    acum = 0
    for i in range(len(params)):
      acum = acum + params[i] * samples[i]
    acum = acum * (-1)
    acum = 1 / (1 + math.exp(acum))
    return acum
#the file has no names for columns.
columns = ["sample_code_number","clump_thickness","uniformity_of_cell_size","uniformity_of_cell_shape","marginal_adhession","single_epithelial_size","bare_nuclei","bland_chromatin","normal_nucleoli","mitoses","label"]
df = pd.read_csv('breast-cancer-wisconsin.csv',names = columns)
#df = df.sample(frac=1)
#df.head()
#df.columns

df_x = df[["sample_code_number","clump_thickness","uniformity_of_cell_size","uniformity_of_cell_shape","marginal_adhession","single_epithelial_size","bare_nuclei","bland_chromatin","normal_nucleoli","mitoses"]]
df_y = df[["label"]]

alfa = 0.03 #learning rate

p1 = random.randint(-10, 10)
p2 = random.randint(-10, 10)
p3 = random.randint(-10, 10)
p4 = random.randint(-10, 10)

params = [p1,p2,p3,p4]


dfl = int(len(df.index) * 0.20)
#print(len(df.index))
#print(dfl)

df_tr = df_x.iloc[:-dfl]
df_t = df_x.iloc[-dfl:]
y_tr = df_y.iloc[:-dfl]
y_t = df_y.iloc[-dfl:]


x1 = list(df_tr.bland_chromatin.values)
x2 = list(df_tr.normal_nucleoli.values)
x3 = list(df_tr.mitoses.values)

x1_t = list(df_tr.bland_chromatin.values)
x2_t = list(df_tr.normal_nucleoli.values)
x3_t = list(df_tr.mitoses.values)

y = list(y_tr.label.values)
y_test = list(y_t.label.values)

bias = []
t_bias = []

for i in range(len(x1)):
  bias.append(1)
bias = list(bias)

for i in range(len(x1_t)):
  t_bias.append(1)
t_bias = list(t_bias)

samples = numpy.array([bias,x1,x2,x3]).transpose()
test_samples = numpy.array([t_bias,x1_t,x2_t,x3_t]).transpose()

"""
print ("original samples:")
print (samples)
samples = scaling(samples)
print ("scaled samples:")
print (samples)
"""

#print("global avg:")
#print(__avg__)
#print("global max val:")
#print(__maxVal__)

avg = __avg__
max_val = __maxVal__

#test_samples = scalingTest(test_samples, avg, max_val)

epoch = 0

while True:
  oldparams = list(params)
  print (params)
  params=gradDesc(params, samples,y,alfa)
  error = show_errors(params, samples, y)
  epoch = epoch + 1
  if(oldparams == params or error < 0.01 or epoch > 20000  ):
    print ("samples:")
    print (samples)
    print ("final params:")
    print (params)
    break


plt.plot(__errors__)
plt.show()

predYArr = []
ids = []
res = 0
for i in range(len(y_test)):
    res = yPredict(params,test_samples[i])
    print("y prediction: ",res)
    print("y real: ",y_test[i])
    print("\n")
    predYArr.append(res)
    ids.append(i)
print("=======================================================\n")
print("Query Section\n")
print("=======================================================\n")

while True:
    bc = input("Introduce the bland chromatin value:")
    nn = input("Introduce the normal nucleoli value:")
    m = input("Introduce the mitoses value:")
    bc = int(bc)
    nn = int(nn)
    m = int(m)
    samp = [1,bc,nn,m]
    res = yPredict(params,samp)
    print("y prediction: ",res)
    print("\n")





"""
plt.plot(predYArr)
plt.show()
"""
