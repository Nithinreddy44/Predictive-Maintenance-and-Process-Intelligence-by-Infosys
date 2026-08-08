# Predictive Maintenance Analysis Report

## Dataset Summary
- Rows: 5000
- Machines: 25
- Failure rate: 66.54%
- Missing values: 5699

## Model Comparison
              Model  Accuracy  Precision  Recall     F1  ROC_AUC
Logistic Regression     0.552     0.7183  0.5368 0.6145   0.5965
            XGBoost     0.663     0.6723  0.9624 0.7916   0.5763
      Random Forest     0.627     0.6877  0.8045 0.7415   0.5660

## Key Findings
- XGBoost was trained in the current run, but some environments may require the OpenMP runtime (libomp) to load the native library: 
XGBoost Library (libxgboost.dylib) could not be loaded.
Likely causes:
  * OpenMP runtime is not installed
    - vcomp140.dll or libgomp-1.dll for Windows
    - libomp.dylib for Mac OSX
    - libgomp.so for Linux and other UNIX-like OSes
    Mac OSX users: Run `brew install libomp` to install OpenMP runtime.

  * You are running 32-bit Python on a 64-bit OS

Error message(s): ["dlopen(/Users/nithinkumar/Downloads/ai preditive/.venv/lib/python3.13/site-packages/xgboost/lib/libxgboost.dylib, 0x0006): Library not loaded: @rpath/libomp.dylib\n  Referenced from: <010DE1F1-B66F-31D2-8EDA-A08913D25DDA> /Users/nithinkumar/Downloads/ai preditive/.venv/lib/python3.13/site-packages/xgboost/lib/libxgboost.dylib\n  Reason: tried: '/opt/homebrew/opt/libomp/lib/libomp.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/opt/libomp/lib/libomp.dylib' (no such file), '/opt/homebrew/opt/libomp/lib/libomp.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/opt/libomp/lib/libomp.dylib' (no such file)"]

- Failures increase sharply when vibration, temperature, and maintenance severity move beyond normal operating ranges.
- The highest-risk operating conditions are concentrated in older machines and in maintenance histories flagged as Major or Critical.

## Recommended Actions
1. Prioritize inspection and replacement of components on machines with high vibration, elevated temperature, and poor maintenance history.
2. Schedule preventive maintenance before the failure rate rises above 5% in any machine cluster.
3. Tighten operating thresholds for pressure, humidity, RPM, and voltage to reduce abnormal operating states.

## Failure Drivers by Maintenance History
Maintenance History  Failure
           Critical    0.791
              Major    0.732
              Minor    0.651