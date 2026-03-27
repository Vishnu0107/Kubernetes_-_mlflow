import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# connect to your MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Vishnu_Narayanan_2022BCS0001")

# dataset
X,y=load_diabetes(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

with mlflow.start_run():

    model=RandomForestRegressor(n_estimators=100,max_depth=5)
    model.fit(X_train,y_train)

    preds=model.predict(X_test)
    mse=mean_squared_error(y_test,preds)

    # log parameters
    mlflow.log_param("model","RandomForest")
    mlflow.log_param("n_estimators",100)
    mlflow.log_param("max_depth",5)

    # log metric
    mlflow.log_metric("mse",mse)

    # save artifact
    plt.scatter(y_test,preds)
    plt.savefig("plot.png")
    mlflow.log_artifact("plot.png")

    # log model
    mlflow.sklearn.log_model(model,"model")

    print("Run completed")