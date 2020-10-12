import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
import io

__error_arr__ = [];
__avg__ = 0;
__max_val__ = 0;
__param_checkpoint__ =[]

def hypothesis(params, sample):
    acum = 0
    for i in range(len(params)):
        acum = acum + params[i] * sample[i]
    return acum

def cost(params, samples, y):
    acum = 0
    m = len(samples)
    i = 0
    while i < m:
        acum = acum + (hypothesis(params[i], samples[i]) - y[i])**2
        i = i + 1
    acum = acum / (2*m)
    return acum

def gradientDescent(params, samples, y, alpha ):
	temp = list(params)
	general_error=0
	for j in range(len(params)):
		acum =0; error_acum=0
		for i in range(len(samples)):
			error = hypothesis(params,samples[i]) - y[i]
			acum = acum + error*samples[i][j]
		temp[j] = params[j] - alpha*(1/len(samples))*acum
	return temp


def scaling(samples):
    global __avg__, __max_val__, __param_checkpoint__

    acum = 0
    samples = np.asarray(samples).T.tolist()
    for i in range(1, len(samples)):
        for j in range(len(samples[i])):
            acum =+ samples[i][j]
        average = acum / len(samples[i])
        max_val = max(samples[i])
        for j in range(len(samples[i])):
            samples[i][j] = (samples[i][j]  - average)/ max_val
        __param_checkpoint__= samples
    __avg__ = average
    __max_val__ = max_val
    return np.asarray(samples).T.tolist()

def scale_q(query):
    global __avg__, __max_val__
    average = __avg__,
    max_val = __max_val__
    if isinstance(query, list):
        query=  [1]+query
    else:
        query =  [1,query]
    for i in range(len(query)):
        query[i] = (query[i] - average[0]) / max_val
    return query



def show_errors(params, samples, y):
    global __error_arr__
    acum_err = 0
    sample_len = len(samples)
    for i in range (sample_len):
        h = hypothesis(params, samples[i])
        #print("Hipothesis: "+ str(h) + " y: "+str(y[i]))
        error = h -  y[i]
        acum_err = acum_err  + (error ** 2)
    sme = acum_err / sample_len
    __error_arr__.append(sme)

def sortXY(x, y):
    final_arr = []
    for i in range(len(x)):
        final_arr.append([x[i], y[i]])
    final_arr= sorted(final_arr)


    final_x = []
    final_y = []
    for i in range(len(x)):
        final_x.append(final_arr[i][0])
        final_y.append(final_arr[i][1])
    #print(final_x)
    #print(final_y)
    return final_x, final_y


def main():

    file= pd.read_csv('house_dataset.csv',encoding = "ISO-8859-1")
    #file= pd.read_csv('kc_house_data.csv',encoding = "ISO-8859-1")
    #file = file.sample(frac=1)


    x1 = file.sqft_above.values
    x2 = file.sqft_basement.values
    x3 = file.sqft_living.values
    x4 = file.sqft_lot.values
    x5 = file.grade.values

    y = file.price.values

    x1 = list(x1)
    x2 = list(x2)
    x3 = list(x3)
    x4 = list(x4)
    x5 = list(x5)
    y = list(y)
    """
    g_x1,g_y1= sortXY( x1, y)
    g_x2,g_y2 = sortXY( x2, y)
    g_x3,g_y3 = sortXY( x3, y)
    g_x4,g_y4 = sortXY( x4, y)
    g_x5,g_y5 = sortXY( x5, y)


    sx1 = sorted(x1)
    sx2 = sorted(x2)
    sx3 = sorted(x3)
    sx4 = sorted(x4)
    sx5 = sorted(x5)

    ids = []
    for i in range(len(x1)):
        ids.append(i)

    plt.plot(sx1,ids, color='blue', label='x1')
    plt.plot(sx2,ids, color='black', label='x2')
    plt.plot(sx3,ids, color='green', label='x3')
    #plt.plot(sx4,ids, color='yellow', label='x3')
    #plt.plot(sx5,ids, color='red', label='x3')
    plt.show()

    plt.scatter(g_x1,g_y1, color='blue', label='x1')
    plt.legend()
    plt.show()
    plt.scatter(g_x2,g_y2, color='black', label='x2')
    plt.legend()
    plt.show()
    plt.scatter(g_x3,g_y3, color='green', label='x3')
    plt.legend()
    plt.show()
    plt.scatter(g_x4,g_y4, color='yellow', label='x3')
    plt.legend()
    plt.show()
    plt.scatter(g_x5,g_y5, color='red', label='x3')
    plt.legend()
    plt.show()

    plt.scatter(x1,y, color='blue', label='x1: sqft above')
    plt.legend()
    plt.show()
    plt.scatter(x2,y, color='black', label='x2: sqft basement')
    plt.legend()
    plt.show()
    plt.scatter(x3,y, color='green', label='x3: sqft living')
    plt.legend()
    plt.show()
    plt.scatter(x4,y, color='orange', label='x4: sqft lot')
    plt.legend()
    plt.show()
    plt.scatter(x5,y, color='red', label='x5: grade')
    plt.legend()
    plt.show()
    """


    bias = []
    for i in range(len(x1)):
        bias.append(1)
    bias = list(bias)

    samples = np.array([bias,x1,x2,x3,x4,x5]).transpose()
    #samples = np.array([bias,x1,x2,x3]).transpose()


    #plt.plot(x1,y, color='blue', label='x1')
    #plt.plot(x2,y, color='green', label='x2')
    #plt.plot(x3,y, color='red', label='x3')
    #plt.plot(x4,y, color='black', label='x4')
    #plt.plot(x5,y, color='yellow', label='x5')
    #plt.show()


    params = [0,0,0,0,0,0]
    alfa =0.01
    epochs = 0

    #print ("original samples:")
    #print (samples)
    samples = scaling(samples)
    #print ("scaled samples:")
    #print (samples)
    #samples = np.array(samples)

    t_len = (len(x1))*0.2
    t_len  = int(t_len )
    samples_train = samples[:-t_len]
    samples_test = samples[-t_len:]
    y_train = y[:-t_len]
    y_test = y[-t_len:]


    #print(len(y_train))
    #print("samples_train: ")
    #print(samples_train)
    #print("samples_test: ")
    #print(samples_test)

    x1_train = []
    x2_train = []
    x3_train = []
    x4_train = []
    x5_train = []


    for i in range(len(samples_train)):
        x1_train.append(samples_train[i][1])
        x2_train.append(samples_train[i][2])
        x3_train.append(samples_train[i][3])
        x4_train.append(samples_train[i][4])
        x5_train.append(samples_train[i][5])

    while True:
        oldparams = list(params)
        #print (params)
        params=gradientDescent(params, samples_train,y,alfa)
        epochs = epochs + 1
        show_errors(params, samples, y)
        if(oldparams == params or epochs == 60000):
        	#print ("samples:")
        	#print(samples)
        	print ("\nfinal params:")
        	print (params)
        	break


    id = []
    y_predict = []
    for i in range(len(samples_train)):
        id.append(i)
        acum = 0
        for j in range(len(samples_train[i])):
            acum = acum + (samples_train[i][j] * params[j])
        y_predict.append(acum)


    id2 = []
    y_predict_test= []
    for i in range(len(samples_test)):
        id2.append(i)
        acum = 0
        for j in range(len(samples_test[i])):
            acum = acum + (samples_test[i][j] * params[j])
        y_predict_test.append(acum)

    print("Mean Squared Error Train = ", round(sm.mean_squared_error(y_train,y_predict),2))
    print("R2 Score Train= ", round(sm.r2_score(y_train,y_predict),2))


    print("Mean Squared Error Test = ", round(sm.mean_squared_error(y_test,y_predict_test),2))
    print("R2 Score Test= ", round(sm.r2_score(y_test,y_predict_test),2))


    print("\nQueries: \n")
    for i in range(20):
        print("\ny_predict: "+str(y_predict_test[i]))
        print("y: "+ str(y_test[i])+"\n")


    plt.plot(__error_arr__)
    plt.title("Error")
    plt.show()

    plt.plot(id,y_train, color='black',label = "training y's")
    plt.plot(id,y_predict, color="red",label="predicted training y's")
    plt.title("Real  vs Predicted: Training")
    plt.legend()
    plt.show()

    plt.plot(id2,y_test, color='black', label="test y's")
    plt.plot(id2,y_predict_test, color="blue",label="predicted test y's")
    plt.title("Real  vs Predicted: Test")
    plt.legend()
    plt.show()


    yts = sorted(y_train)
    yts2 = sorted(y_test)
    yps = sorted(y_predict)
    yps2 = sorted(y_predict_test)

    plt.plot(id,yts, color='black', label = "training y's")
    plt.plot(id,yps, color="red", label="predicted training y's")
    plt.title("Real  vs Predicted: Training(Sorted)")
    plt.legend()
    plt.show()

    plt.plot(id2,yts2, color='black', label="test y's")
    plt.plot(id2,yps2, color="blue", label="predicted test y's")
    plt.title("Real  vs Predicted: Test (Sorted)")
    plt.legend()
    plt.show()


main()
