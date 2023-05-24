import streamlit as st
import pandas as pd
import nltk
from nltk import word_tokenize
import numpy as np
import json
import os
import altair as alt
import seaborn as sb
from tqdm import tqdm
import string
from scipy.stats import ttest_ind
from scipy.stats import shapiro
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from utils import load_sessions, read_session
from main import generate_buffer
from events import generate_event_seq
from summary import stats


def xxx (events,file_name):
    text = []
    sentence_metrics_list = []
    api_metrics_list = []
    err = []

    for event in events:
        text_buffer = generate_buffer(events)
        text.append(text_buffer[-1])
        event_seq_dict = generate_event_seq(buffer=text_buffer,
                                            events=events)
        sentence_metrics, api_metrics = stats(event_seq_dict)
        sentence_metrics_list.append(sentence_metrics)
        api_metrics_list.append(api_metrics)

        for e in err:
            print(e)

        df = pd.DataFrame()

        df["file_name"] = file_name
        df["text"] = text

        for col in sentence_metrics_list[0]:
            df[str(col)] = [x[col] for x in sentence_metrics_list]

        for col in api_metrics_list[0]:
            df[str(col)] = [x[col] for x in api_metrics_list]
            return df


def read_jsonl_to_df(file):
    if file is not None:
        # Read the contents of the file into a string
        file_contents = file.read().decode("utf-8")
        # Split the string into a list of JSON objects
        events = file_contents.strip().split("\n")
        # Convert the list of JSON objects into a pandas DataFrame
        df = pd.DataFrame([json.loads(event) for event in events])

        # Add a new column with the name of the file
        file_name = os.path.splitext(file.name)[0]
        df["file_name"] = file_name

        # Show the DataFrame
        st.write(df.file_name[15])
        st.write(events)
        st.write(df)
        return events, file_name



def main():
    st.title('Writing analysis report and Dashboard')
    st.write('Upload the jsonl file resulting from your writing exercise')
    file = st.file_uploader("Upload file", type=['jsonl'])
    if not file:
        st.write("Please upload a jsonl file to generate report")
        return
    read_jsonl_to_df(file)



main()
