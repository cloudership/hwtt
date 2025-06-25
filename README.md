# HWTT - A collection of LLM experiments (Was: "HandWriting To Text")

This started as a place for tools that would help convert handwriting or printed text on a page to a digital document.

Now it's a place for all my LLM experiments.

## Requirements

Python 3.12 (NoGIL in 3.13 causes problems)

## USAGE

Copy .env.example to .env, and fill in any placeholders. GEMINI_API_KEY can be ignored, but OPEN_API_KEY is required for
now.

Install packages: `pip install -r requirements.txt`

(In separate terminal) Start MLflow server: `./run-mlflow.sh`

Run the Jupyter notebook `notebooks/question-answering-evaluation.ipynb` and it should work! If everything is working,
you will see the MLflow UI in the notebook itself.
