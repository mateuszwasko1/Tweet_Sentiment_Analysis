# INSTRUCTIONS ON HOW TO OPERATE THIS PROJECT

## INSTALL THE DEPENDENCIES
Before running anything you have to make sure you are in the correct env:
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
## TRAIN A MODEL OR CHECK FOR BIAS
In order to train a model or check for bias you have to run the following command and answer the questions in the terminal
```bash
python training_or_bias.py
```

## RUNNING THE DEMO
In order to run the demo (the streamlit) you can run the following command:
```bash
python main.py
```
It should take you to a the streamlit, but if it does not go to this link:
[http://localhost:8501/](http://localhost:8501/)

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

