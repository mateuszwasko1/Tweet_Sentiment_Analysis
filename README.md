# INSTRUCTIONS ON HOW TO RUN THE API AND INSTALL THE DEPENDENCIES

# STARTING THE STREAMLIT
Make sure that you run the FastAPI file beforehand on one terminal: 
```bash
    uvicorn project_name.deployment.deploy_model:app --reload
```
Then, on a separate terminal, run the streamlit app:
```bash
    streamlit run project_name/demo.py
```

## RUNNING THE API
Run the following code in the terminal to start the API
```bash
    uvicorn main:app --reload
```
Then go to http://127.0.0.1:8000

Click at the POST /predict box and then “Try it out”. This will give you a box where you can input a string. The prediction can be found in the following box:

Response Body
```bash
    {
"input":"you are stupid",
"prediction":"anger"
}
```

## INSTALL THE DEPENDENCIES
Make sure that Pipenv is installed.
```bash
    pip install pipenv
```
```bash
# to activate the python environment
    pipenv shell 
# to install the dependencies from the Pipfile
    pipenv install 
```

Note that the code below can also install the exact versions from Pipfile.lock
```bash
# to activate the python environment
   pipenv sync
```

