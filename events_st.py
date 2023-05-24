import pandas as pd
import nltk
import numpy as np
import math
import streamlit as st


def init_df(buffer, events):
    df = pd.DataFrame()
    df["text_buffer"] = buffer
    df["events"] = events
    return df


def extract_sent(df):
    sentence_buffer = []
    num_sentences = []

    for text in df["text_buffer"]:
        sentences = nltk.tokenize.sent_tokenize(text)
        sentence_buffer.append(sentences)
        num_sentences.append(len(sentences))

    df["sentences"] = sentence_buffer
    df["num_sentences"] = num_sentences

    return df


def extract_event_name(df):
    df["event_name"] = df["events"].apply(lambda x: x["eventName"])
    df["eventSource"] = df["events"].apply(lambda x: x["eventSource"])
    df["eventTimestamp"] = df["events"].apply(lambda x: x["eventTimestamp"])
    df["textDelta"] = df["events"].apply(lambda x: x["textDelta"])
    df["cursorRange"] = df["events"].apply(lambda x: x["cursorRange"])
    df["currentDoc"] = df["events"].apply(lambda x: x["currentDoc"])
    df["currentCursor"] = df["events"].apply(lambda x: x["currentCursor"])
    df["currentSuggestions"] = df["events"].apply(lambda x: x["currentSuggestions"])
    df["currentSuggestionIndex"] = df["events"].apply(lambda x: x["currentSuggestionIndex"])
    df["currentHoverIndex"] = df["events"].apply(lambda x: x["currentHoverIndex"])
    df["currentN"] = df["events"].apply(lambda x: x["currentN"])
    df["currentMaxToken"] = df["events"].apply(lambda x: x["currentMaxToken"])
    df["currentTemperature"] = df["events"].apply(lambda x: x["currentTemperature"])
    df["currentTopP"] = df["events"].apply(lambda x: x["currentTopP"])
    df["currentPresencePenalty"] = df["events"].apply(lambda x: x["currentPresencePenalty"])
    df["currentFrequencyPenalty"] = df["events"].apply(lambda x: x["currentFrequencyPenalty"])
    df["eventNum"] = df["events"].apply(lambda x: x["eventNum"])
    # count_sentences = lambda x: len(nltk.sent_tokenize(' '.join(x)))
    # df['current_sentences'] = df['sentences'].apply(count_sentences)

    return df


def correct_sent_num(df):
    df = df.groupby("num_sentences", group_keys=True).apply(lambda x: x)
    df = df.sort_index()

    num_sentences = np.array(df["num_sentences"])
    event_names = np.array(df["event_name"])

    start_idx = 0
    select_flag = False

    for idx, event in enumerate(event_names):
        if event == "suggestion-get":
            start_idx = idx
        if event == "suggestion-select":
            select_flag = True
        if select_flag and event == "text-insert":
            if num_sentences[start_idx] == num_sentences[idx]:
                end_idx = idx + 1
            elif num_sentences[start_idx] < num_sentences[idx]:
                end_idx = idx
            for i in range(start_idx, end_idx):
                num_sentences[i] += 1
            select_flag = False

    df["num_sentences"] = num_sentences

    return df


def compute_seq(events):
    # Remove suggestion-open, suggestion-hover, suggestion-down suggestion-up
    events = np.delete(events, np.where(events == "suggestion-open"))
    events = np.delete(events, np.where(events == "suggestion-hover"))
    events = np.delete(events, np.where(events == "suggestion-down"))
    events = np.delete(events, np.where(events == "suggestion-up"))

    # Remove suggestion-reopen (for now; unsure of its impact)
    events = np.delete(events, np.where(events == "suggestion-reopen"))

    # Remove text-insert after suggestion-select
    select_flag = False
    new_events = []
    for idx, event in enumerate(events):
        if event == "suggestion-select":
            select_flag = True
        if event == "text-insert" and select_flag:
            select_flag = False
            continue
        new_events.append(event)
    events = np.array(new_events)

    # Identify GPT-3 modifications
    select_flag = False
    new_events = []
    for idx, event in enumerate(events):
        if event == "suggestion-select":
            select_flag = True
        if event == "text-insert":
            select_flag = False
        if (event == "cursor-backward" or event == "cursor-select" or event == "text-delete") and select_flag:
            select_flag = False
            event = "gpt3-modify"
        new_events.append(event)
    events = np.array(new_events)

    # Remove cursor-forward, cursor-backward, cursor-select
    events = np.delete(events, np.where(events == "cursor-forward"))
    events = np.delete(events, np.where(events == "cursor-backward"))
    events = np.delete(events, np.where(events == "cursor-select"))

    # Remove text-delete
    events = np.delete(events, np.where(events == "text-delete"))

    # Remove suggestion-close
    events = np.delete(events, np.where(events == "suggestion-close"))

    # Identify GTP-3 calls
    events = events.tolist()
    start_idx = 0
    api_flag = False
    pop_idx = []
    for idx, event in enumerate(events):
        if event == "suggestion-get":
            start_idx = idx
            api_flag = True
        if event == "suggestion-select" and api_flag:
            api_flag = False
            for i in range(start_idx, idx):
                pop_idx.append(i)
    events = np.array(events)
    events = np.delete(events, pop_idx)

    # Group together text-inserts
    new_events = []
    temp = []
    for event in events:
        if event == "text-insert":
            temp.append(event)
        else:
            if len(temp) != 0:
                new_events.append("text-insert")
            new_events.append(event)
            temp = []
    if len(temp) != 0:
        new_events.append("text-insert")
    events = np.array(new_events)

    # Rename sequences
    seq_name_dict = {
        "system-initialize": "prompt",
        "text-insert": "user",
        "suggestion-get": "empty-call",
        "suggestion-select": "gpt3-call",
        "gpt3-modify": "modify-gpt3",
    }
    new_events = [seq_name_dict[event] for event in events]
    events = np.array(new_events)

    return events


def get_sent_num_and_event_seq(df):
    temp_dict = {
        "num_sent": [],
        "sequence": [],
    }

    for num in np.unique(df["num_sentences"]):
        # sent = np.array(df[df["num_sentences"] == num]["text_buffer"])[-1]
        event_seq = np.array(df[df["num_sentences"] == num]["event_name"])
        temp_dict["num_sent"].append(num)
        temp_dict["sequence"].append(compute_seq(event_seq))

    # Bug fix for prompt deletion
    if temp_dict["num_sent"][0] == 0:
        for idx in range(len(temp_dict["sequence"])):
            if "prompt" in temp_dict["sequence"][idx]:
                temp_arr = temp_dict["sequence"][idx]
                temp_dict["sequence"][idx] = np.delete(temp_arr, np.where(temp_arr == "prompt"))

    return temp_dict


def generate_event_df(buffer, events):
    df = init_df(buffer, events)
    df = extract_sent(df)
    df = extract_event_name(df)
    df = correct_sent_num(df)

    df['maxCursor'] = df['currentCursor'].cummax()
    df['time_dif_since_last_event'] = (df['eventTimestamp'] - df['eventTimestamp'].shift(1)) / 1000
    df['time_dif_since_last_event'] = df['time_dif_since_last_event'].fillna(0)
    df['pause'] = df['time_dif_since_last_event'] >= 5
    df['pause_num'] = df['pause'].cumsum()
    df['pause_num'] = np.where(df['pause'] == False, np.nan, df['pause_num'])
    df['pause_num'] = df['pause_num'].fillna(0)
    df['delete'] = np.where(df['event_name'] == 'text-delete', 1, 0)
    df['delete'] = df['delete'].fillna(0)
    # df['num_sent'] = int(df['num_sent'])
    min_value = df['num_sentences'].min()
    max_value = df['num_sentences'].max()
    #stage_values = ((max_value - min_value) + 1) / 3
    #df.loc[df['num_sentences'] <= min_value + stage_values, 'stage'] = 'stage1'
    #df.loc[df['num_sentences'] >= min_value + stage_values + stage_values, 'stage'] = 'stage3'
    #df.loc[~((df['num_sentences'] <= min_value + stage_values) | (
    #        df['num_sentences'] >= min_value + stage_values + stage_values)), 'stage'] = 'stage2'

    stage_values = ((max_value - min_value) + 1) / 2
    df.loc[df['num_sentences'] <= min_value + stage_values, 'stage'] = 'stage1'
    df.loc[df['num_sentences'] > min_value + stage_values, 'stage'] = 'stage2'


    max_pause_num = int(df['pause_num'].max())
    for i in range(max_pause_num + 1):
        if i == 0:
            df.loc[df['pause_num'] == i, 'pause_outburst'] = np.nan
        elif i == max_pause_num:
            df.loc[df['pause_num'] == i, 'pause_outburst'] = df['maxCursor'].max() - df.loc[
                df['pause_num'] == i, 'maxCursor']
        else:
            df.loc[df['pause_num'] == i, 'pause_outburst'] = \
                df.loc[df['pause_num'] == i + 1, 'maxCursor'].values[0] - \
                df.loc[df['pause_num'] == i, 'maxCursor'].values[0]

    df['pause_outburst2'] = df['pause_outburst'] / 10  # change?

    return df


def generate_event_seq(df):
    readable_sent_seq = get_sent_num_and_event_seq(df)
    # st.write('readable_sent_seq')
    # st.write(readable_sent_seq)

    return readable_sent_seq
