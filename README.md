# NLP Meeting Analyzer: Automatic Action Item Extraction

## Project Overview
This project uses Natural Language Processing (NLP) and machine learning to identify and extract action items from meeting transcripts.

## Technologies Used
- Python
- Pandas
- spaCy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Streamlit

## Key Features
- Meeting transcript preprocessing
- Linguistic rule-based action candidate detection
- TF-IDF and Logistic Regression baseline
- DistilBERT-based action classification
- Action item extraction
- Speaker, assignee, and deadline identification
- Interactive Streamlit dashboard
- CSV export

## Model Evaluation

| Model | Test Accuracy |
|---|---:|
| TF-IDF + Logistic Regression | 70.00% |
| DistilBERT | 81.67% |

These results reflect the evaluation performed on the project's labelled test dataset.

## Dataset
The project uses the AMI Meeting Corpus. Obtain the dataset through its official distribution process and follow its licence and usage conditions.

## Running the Application

Install dependencies:

    pip install -r requirements.txt

Run the dashboard:

    streamlit run src/dashboard.py

A compatible trained model must be available at the path expected by the application.

## Limitations
- The labelled dataset is relatively small.
- Some action items may be incorrectly classified.
- Assignees and deadlines may require manual verification.
- Further evaluation on independent meeting transcripts is needed.

## Future Improvements
- Expand the labelled dataset.
- Improve action item extraction.
- Evaluate on additional meeting transcripts.
- Enhance the dashboard.

## Author
**Aisha Rasifa**
