### Running MLflow 

```bash
conda activate exp-tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Forward port 5000 in VSCode and go to http://127.0.0.1:5000

If problems with starting MLflow server:
```bash
ps ax | grep guni
kill -9 <PID>
```



To run preprocessing:
```bash
cd ~/mlops-zoomcamp/cohorts/2024/02-experiment-tracking/homework
python preprocess_data.py --raw_data_path ~/notebooks/data --dest_path ~/notebooks/output/
ls ~/notebooks/output
# dv.pkl  test.pkl  train.pkl  val.pkl # 4 files
```

To run modified train.py:
(check you are in the right directory)
```bash
cd ~/mlops-zc/02-exp_tracking
python train.py --data_path ~/notebooks/output/
```


Check MLflow UI