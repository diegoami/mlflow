import h2o

h2o.init()
model = h2o.load_model("mlruns/0/models/m-935ff251a94d41899ecbee172e66c5b8/artifacts/model.h2o/DRF_model_python_1751837466102_2")


my_wine =[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8]
wine_frame = h2o.H2OFrame([my_wine])
# Set proper column names to match training data
wine_frame.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                      'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                      'density', 'pH', 'sulphates', 'alcohol']
predictions = model.predict(wine_frame)
print(predictions)