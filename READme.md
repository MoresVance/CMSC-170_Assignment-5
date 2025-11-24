# CMSC 170 - Introduction to Artificial Intelligence
## Deploying a Trained Deep Learning Classifier as a Web Application

This is a project for compliance with CMSC 170.

### Prerequisite:
- Python 3.12
- Anaconda (Conda 25)

### How to compile, using streamlit:
- ensure you are in the root directory of the project before running any commnads
- create a new conda environment with this command: 
```bash 
    conda create -n cmsc170 python=3.12
```
- activate your new conda environment: 
```bash 
    conda activate cmsc170
```
- install dependencies: 
```bash 
    pip install streamlit numpy tensorflow
```

- Now you are able to compile the program through streamlit
```bash 
    streamlit run streamlit_app.py
```
