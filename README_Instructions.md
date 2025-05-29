# INSTRUCTIONS ON HOW TO RUN THE API AND INSTALL THE DEPENDENCIES
## RUNNING THE API
Run the following code in the terminal to start the API
```bash
    uvicorn main:app --reload
```
Use the \predic endpoint to enter your tweet as a string to get the prediction for the emotion

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

