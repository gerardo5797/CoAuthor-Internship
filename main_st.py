import os
import time

import nltk
import pandas as pd
import altair as alt
from utils_st import read_file
from operations_st import build_text
from events_st import generate_event_seq, generate_event_df
from summary_st import get_summary_stats, get_summary_stats2
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def generate_buffer(events):
    text_buffer = []
    for event in events:
        buffer = build_text(text_buffer, event)
        text_buffer.append(buffer)
    return text_buffer


# Function to play the text buffer
def play(buffer, speed='fast'):
    speed_dict = {
        "fast": 0.001,
        "medium": 0.01,
        "slow": 0.1,
        "instant": True,
    }
    if speed_dict[speed] == True:
        print(buffer[-1])
    else:
        for text in buffer:
            os.system('clear')
            print(text)
            time.sleep(speed_dict[speed])


def main():
    # App file selection
    st.title('Writing analysis report and Dashboard')
    st.subheader('Select the writing exercise you want to analyze')
    # List all jsonl files in the folder
    jsonl_files = [file for file in os.listdir() if file.endswith('.jsonl')]
    # Display dropdown menu with file names
    selected_file = st.selectbox('Select a file', jsonl_files)
    # Read events from the selected file
    events = read_file(selected_file)

    # st.write(events)

    # Generate text buffer
    text_buffer = generate_buffer(events)

    # Generate events df
    event_seq_df = generate_event_df(buffer=text_buffer, events=events)
    event_seq_df['file_name'] = selected_file

    event_seq_df['last_sentence'] = event_seq_df['sentences'].apply(lambda x: x[-1])


    # Reset the indexes
    event_seq_sug = event_seq_df.copy()
    event_seq_sug.reset_index(drop=True, inplace=True)

    # Identify the indices of "suggestion-open" and "suggestion-close"
    open_indices = event_seq_sug[event_seq_sug['event_name'] == 'suggestion-open'].index
    close_indices = event_seq_sug[event_seq_sug['event_name'] == 'suggestion-close'].index
    system_indices = event_seq_sug[event_seq_sug['event_name'] == 'system-initialize'].index

    # Initialize the new column with empty values
    event_seq_sug['suggestion_'] = ''

    # Set "suggestion" for rows between "suggestion-open" and "suggestion-close"
    for start, end in zip(open_indices, close_indices):
        event_seq_sug.loc[start:end, 'suggestion_'] = 'suggestion'
        # Set "pre-suggestion" value to the row before "suggestion-open"
        if start > 0:
            event_seq_sug.loc[start - 2:start - 1, 'suggestion_'] = 'pre-suggestion'
        # Set "post-suggestion" value to the two rows after "suggestion-close"
        if end < len(event_seq_sug) - 1:
            event_seq_sug.loc[end + 1:end + 2, 'suggestion_'] = 'post-suggestion'

    # Set "post-suggestion" value for rows where event_name is "system-initialize"
    for system_index in system_indices:
        event_seq_sug.loc[system_index:system_index + 2, 'suggestion_'] = 'post-prompt'

    event_seq_sug['suggestion_'] = event_seq_sug['suggestion_'].replace('', 'user-input')

    # Initialize the revision column with empty strings
    event_seq_sug['revision'] = ''
    event_seq_sug['revision_number'] = ''

    # Flag revisions based on conditions
    in_revision = False
    revision_count = 1
    for index, row in event_seq_sug.iterrows():
        if row['event_name'] == 'cursor-backward':
            in_revision = True

        if in_revision:
            event_seq_sug.at[index, 'revision'] = 'revision'
            event_seq_sug.at[index, 'revision_number'] = revision_count

        if in_revision and row['currentCursor'] >= row['maxCursor']:
            in_revision = False
            revision_count += 1

    # Create column cursor sentence (useful for when the user goes back and fort).

    # Function to count the number of sentences in a text
    def count_sentences(text):
        sentences = nltk.sent_tokenize(text)
        return len(sentences)

    # Apply the count_sentences function to create the 'cursor_sentence' column
    event_seq_sug['cursor_sentence'] = event_seq_sug.apply(
        lambda row: count_sentences(row['text_buffer'][:row['currentCursor']]), axis=1)

    # st.write(event_seq_sug)

    # Generate event sequence dictionary
    event_seq_dict = generate_event_seq(event_seq_df)

    author_sent_df = pd.DataFrame.from_dict(event_seq_dict)

    def author_source(seq):
        if ("gpt3-call" not in seq) and ("prompt" not in seq) and ("user" in seq):
            return 'original sentence'
        if ("gpt3-call" in seq) and ("user" not in seq):
            return 'GPT-3 sentence'
        else:
            return 'Co-Authored sentence'

    author_sent_df['authorship'] = author_sent_df['sequence'].apply(lambda x: author_source(x))

    dfe = pd.DataFrame(event_seq_dict)
    dfe['sequence'] = dfe['sequence'].apply(tuple)

    # Define the values to count
    values_to_count = ['user', 'gpt3-call', 'modify-gpt3', 'prompt',
                       'empty-call', 'not-modified', 'stage']

    # Create a series with the values to count
    s = pd.Series(values_to_count, name='sequence')

    # Explode the tuples into separate rows
    dfe_exploded = dfe.explode('sequence')

    # Count the occurrences of each value with crosstab
    counts = pd.crosstab(dfe_exploded['num_sent'],
                         dfe_exploded['sequence']).reindex(s, axis=1, fill_value=0)
    counts['not-modified'] = counts['gpt3-call'] - counts['modify-gpt3']

    # Reorder the columns to match the order of the values to count
    # rename the index values
    # counts = counts.rename(index={1: 'stage1', 2: 'stage2', 3: 'stage3'})
    min_value = counts.index.min()
    max_value = counts.index.max()
    stage_values = ((max_value - min_value)+1) / 2
    counts.loc[counts.index <= min_value + stage_values, 'stage'] = 'stage1'
    counts.loc[counts.index > min_value + stage_values, 'stage'] = 'stage2'

    counts = counts[s]

    # Call the function and assign the output to a tuple
    output = get_summary_stats(event_seq_dict)

    # Access the returned objects using indexing
    sentence_metrics = output[0]
    api_metrics = output[1]
    summary_df1 = output[2]
    summary_df1['file_name'] = selected_file
    #st.write(summary_df1)

    summary_df2 = get_summary_stats2(event_seq_df)
    #st.write(summary_df2)

    summary_df_merged = pd.merge(summary_df1, summary_df2, how='inner', left_on='file_name', right_on='file_name')
    #st.write(summary_df_merged)
    ###########
    # Update DF

    # Group the DataFrame by num_sentences and apply aggregation functions
    # Reset the index of the DataFrame
    event_seq_df = event_seq_df.drop("num_sentences", axis=1)
    event_seq_df = event_seq_df.reset_index()

    event_seq_df = event_seq_df.set_index('num_sentences').rename_axis('index_num_sentences')
    event_seq_df = event_seq_df.reset_index()

    event_seq_df = pd.concat(
        [event_seq_df, event_seq_sug[['suggestion_', 'revision', 'revision_number', 'cursor_sentence']]], axis=1)

    #st.write(event_seq_df)
    sentences_df = event_seq_df.groupby('index_num_sentences').agg(
        pauses_over_10_sec=('pause_num', lambda x: (x > 0).sum()),
        pause_burst=('pause_outburst', 'sum'),
        sentence_current=('last_sentence', 'last'),
        time_spent=('time_dif_since_last_event', 'sum'),
        revisions=('revision_number', 'nunique')
    )

    #st.write(sentences_df)

    sentences_df = pd.merge(sentences_df, author_sent_df, left_on='index_num_sentences', right_on='num_sent')

    sentences_df['length_char'] = sentences_df['sentence_current'].str.len()

    # Find similarities suggestions and sentences

    sug_list = []

    # Iterate over the rows of the DataFrame
    for index, row in event_seq_df.iterrows():
        suggestions = row['currentSuggestions']
        if isinstance(suggestions, list):
            sug_list.extend(suggestions)


    # Extract only the 'original' values from sug_list
    suggestions_get = [item['original'] for item in sug_list]
    # Create a new column 'suggestions_get' with the suggestions as lists
    sentences_df['suggestions_get'] = [sug_list] * len(sentences_df)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Convert the dictionaries in 'suggestions_get' column to strings
    sentences_df['suggestions_get'] = sentences_df['suggestions_get'].apply(lambda x: [item['original'] for item in x])

    # Vectorize the sentence_current and suggestions_get columns
    sentence_vectors = vectorizer.fit_transform(sentences_df['sentence_current'])
    suggestion_vectors = [vectorizer.transform(suggestions) for suggestions in sentences_df['suggestions_get']]

    # Compute the similarity scores for each suggestion
    best_suggestions = []

    # Iterate over each row
    for idx, suggestions in enumerate(sentences_df['suggestions_get']):
        max_similarity = 0.0
        best_suggestion = None

        # Iterate over each suggestion
        for suggestion in suggestions:
            similarity = cosine_similarity(sentence_vectors[idx], vectorizer.transform([suggestion]))[0][0]

            # Update the best suggestion if the similarity score is higher
            if similarity > max_similarity:
                max_similarity = similarity
                best_suggestion = suggestion

        # Append the best suggestion to the list
        best_suggestions.append(best_suggestion)

    # Assign the best suggestions to the 'best_suggestion' column
    sentences_df['best_suggestion'] = best_suggestions


    ##

    # DO sentiment analysis (Do cleaning/text processing in a future stage for better results)
    analyzer = SentimentIntensityAnalyzer()
    sentences_df['sentiment'] = sentences_df['sentence_current'].apply(
        lambda x: analyzer.polarity_scores(x)['compound'])
    sentences_df['sentiment_label'] = sentences_df['sentiment'].apply(
        lambda score: 'Positive' if score > 0 else 'Neutral' if score == 0 else 'Negative')

    # Use textblob with weights
    custom_weights = {'positive': 2.0, 'negative': -2.0}

    # Apply sentiment analysis with custom weights
    sentences_df['sentiment2'] = sentences_df['sentence_current'].apply(
        lambda x: TextBlob(x).sentiment.polarity * custom_weights.get(TextBlob(x).sentiment.polarity >= 0, 1.0))
    sentences_df['sentiment_label2'] = sentences_df['sentiment2'].apply(
        lambda score: 'Positive' if score > 0 else 'Neutral' if score == 0 else 'Negative')

    #st.write(sentences_df)

    ##############################
    # Time to plot
    ##############################

    # Pie chart 1

    # create new DataFrame with only 4 columns

    df_selected = summary_df_merged[['Sentences initial prompt', 'Original sentences',
                                     'GPT-3 sentences', 'Co-authored sentences']]

    # melt DataFrame to create 'variable' and 'value' columns
    melted = df_selected.melt(var_name='variable', value_name='value')

    # Filter the dataframe to consider only the relevant variables
    filtered_df_pie = melted[
        melted['variable'].isin(['Original sentences', 'GPT-3 sentences', 'Co-authored sentences'])]

    # Get the values for each variable
    original_sentences_value = filtered_df_pie[filtered_df_pie['variable'] == 'Original sentences']['value'].iloc[0]
    gpt3_sentences_value = filtered_df_pie[filtered_df_pie['variable'] == 'GPT-3 sentences']['value'].iloc[0]
    coauthored_sentences_value = filtered_df_pie[filtered_df_pie['variable'] == 'Co-authored sentences']['value'].iloc[
        0]

    # Initialize the sentence variable
    sentence_pie = ""

    # Compare the values based on the conditions
    # if original_sentences_value > gpt3_sentences_value and original_sentences_value > coauthored_sentences_value:
    #   sentence_pie += "It is great to see that most of your sentences were originally produced. "
    if original_sentences_value > gpt3_sentences_value + coauthored_sentences_value:
        sentence_pie += "It is great to see that most of your sentences were originally produced. "
    if original_sentences_value < gpt3_sentences_value + coauthored_sentences_value:
        sentence_pie += "The amount of sentences with GPT intervension are greater than your original" \
                        "sentences, you are getting the most of the tool and collaborating a lot. "
    if gpt3_sentences_value > original_sentences_value and gpt3_sentences_value > coauthored_sentences_value:
        sentence_pie += "We can see that most of your sentences were authored by GPT-3 and not modified," \
                        "go back to the full text and analyzed if the purple sentences can be improved. "
    if coauthored_sentences_value > original_sentences_value and coauthored_sentences_value > gpt3_sentences_value:
        sentence_pie += "We can see that most of your sentences were Co-authored with GPT-3," \
                        "Great job collaborating with GPT. "
    if original_sentences_value == gpt3_sentences_value and original_sentences_value == coauthored_sentences_value:
        sentence_pie += "Your writing exercise was completely balanced, the amount of sentences produced" \
                        "originally by you, by GPT and Co-authored were exactly the same, this also means" \
                        "GPT was involved in 66% of your sentences, any idea how to make this even more original? "
    if original_sentences_value < gpt3_sentences_value and original_sentences_value < coauthored_sentences_value:
        sentence_pie += "Your original sentences were considerably lower than those where GPT intervened," \
                        "take a look at your text and see if you can make it more original while adding value," \
                        "if not possible, make sure to reference GPT's contributions. "

    if gpt3_sentences_value > 0 and coauthored_sentences_value > 0:
        if gpt3_sentences_value > coauthored_sentences_value:
            sentence_pie += "When asking for help from GPT, most of the times you decided to stick with its" \
                            "suggestion, check the purple sentences and see if collaboration could improve" \
                            "the quality of the sentence. "
        elif gpt3_sentences_value < coauthored_sentences_value:
            sentence_pie += "When asking for help from GPT, most of the times you edited their suggestion," \
                            "that is what we call collaboration, great job at mixing original ideas with " \
                            "suggestions. "

    if gpt3_sentences_value > 0 and coauthored_sentences_value == 0:
        sentence_pie += "Every time you received a GPT suggestion, you used it as a complete sentence " \
                        "and did not modify it. "
    if gpt3_sentences_value == 0 and coauthored_sentences_value > 0:
        sentence_pie += "All the sentence where you received GPT suggestions were modified, " \
                        "either by modifying the suggestion or adding/removing text from that sentence " \
                        "Great job collaborating with this tool. "
    if gpt3_sentences_value == 0 and coauthored_sentences_value == 0:
        sentence_pie += "You did not generate any sentence with the help of GPT, do you think collaborating" \
                        "with the tool could help you achieve better results? was this intentional, reflect. "

    # st.write(melted)

    # create pie chart using melted DataFrame
    pie = alt.Chart(melted).mark_arc().encode(
        theta='value',
        color='variable'
    ).properties(
        width=300,
        height=300,
        title='Sentence source'
    )

    # Bar charts 1 and 2

    # Aggregate the data by stage and calculate the sum of modify-gpt3 and not-modified columns

    df_agg = counts.groupby('stage').agg({'gpt3-call': 'sum', 'empty-call': 'sum',
                                          'modify-gpt3': 'sum', 'not-modified': 'sum'}).reset_index()
    #st.write(df_agg)
    # Melt the dataframe to make it tidy for Altair
    df_melt = df_agg.melt(id_vars='stage', var_name='status', value_name='count')
    counts['num_sent'] = counts.index
    #st.write('counts')
    #st.write(counts)
    df_agg2 = counts[['num_sent', 'gpt3-call', 'empty-call', 'modify-gpt3', 'not-modified']]
    #st.write(df_agg2)  #
    df_melt2 = df_agg2.melt(id_vars='num_sent', var_name='status', value_name='count')

    # Drop rows with count = 0 from df_melt2
    df_melt2 = df_melt2.loc[df_melt2['count'] != 0]

    # Drop rows with count = 0 from df_melt
    df_melt = df_melt.loc[df_melt['count'] != 0]

    #st.write(df_melt)
    #st.write(df_melt2)
    # Create the stacked bar chart using Altair
    chart1 = alt.Chart(
        df_melt2[(df_melt2['status'] == 'gpt3-call') | (df_melt2['status'] == 'empty-call')]).mark_bar().encode(
        x=alt.X('num_sent:N', title='Sentence'),
        y=alt.Y('count:Q', title='GPT-3 calls'),
        color=alt.Color('status:N', legend=alt.Legend(title='Status'),
                        scale=alt.Scale(domain=['gpt3-call', 'empty-call'],
                                        range=['#1f77b4', '#ff7f0e']))
    ).properties(
        width=200,
        height=200,
        title='Counts of GPT-3 calls per Sentence (Accepted-Dismissed)'
    )

    chart2 = alt.Chart(
        df_melt2[(df_melt2['status'] == 'modify-gpt3') | (df_melt2['status'] == 'not-modified')]).mark_bar().encode(
        x=alt.X('num_sent:N', title='Sentence'),
        y=alt.Y('count:Q', title='Count'),
        color=alt.Color('status:N', legend=alt.Legend(title='Status'),
                        scale=alt.Scale(domain=['modify-gpt3', 'not-modified'],
                                        range=['blue', 'yellow']))
    ).properties(
        width=200,
        height=200,
        title='Counts of Accepted GPT-3 call per stage (modified and not-modified)'
    )

    # Calculate the sums for each stage
    sum_stage1 = df_agg.loc[df_agg['stage'] == 'stage1', ['gpt3-call', 'empty-call']].sum().sum()
    sum_stage2 = df_agg.loc[df_agg['stage'] == 'stage2', ['gpt3-call', 'empty-call']].sum().sum()

    # Generate the text based on GPT calls made per stage
    if sum_stage1 > sum_stage2 and sum_stage1 != 0:
        sentence_call = "Seems like you needed more help from GPT at the beginning of the exercise, " \
                        "then you started writing more freely on your own. Do you usually have more " \
                        "issues getting started when writing?"

    elif sum_stage1 > sum_stage2 and sum_stage2 == 0:
        sentence_call = "Seems like you needed more help from GPT at the beginning of the exercise, " \
                        "then you started writing freely on your own. You did not make any call at the " \
                        "end. Do you usually have more issues getting started when writing?"

    elif sum_stage2 > sum_stage1 and sum_stage2 != 0:
        sentence_call = "Seems like you relied less on GPT at the beginning of the exercise, while you " \
                        "used it more towards the end. Do you usually have more trouble finishing your " \
                        "texts? Which factors do you think influence this?"

    elif sum_stage2 > sum_stage1 and sum_stage1 == 0:
        sentence_call = "Seems like you did not need any help from GPT at the beginning of your exercise, " \
                        "you only used it towards the second half. Do you usually have more trouble " \
                        "finishing your texts? Which factors do you think influence this?"

    elif sum_stage2 == sum_stage1 and sum_stage2 != 0:
        sentence_call = "Your call for suggestions was equally the same at the beginning and end of " \
                        "the exercise. Seems like you balanced well when to ask for help."

    elif sum_stage2 == sum_stage1 == 0:
        sentence_call = "You did not request any help from GPT. Seems like you prefer to write completely " \
                        "on your own. Are there any particular reasons you prefer not to use the help " \
                        "from this available tool?"

####

    # Calculate the sums for each column
    sum_modify_gpt3 = df_agg['modify-gpt3'].sum()
    sum_not_modified = df_agg['not-modified'].sum()

    # Generate the text based on modified suggestions
    if sum_modify_gpt3 > sum_not_modified and sum_not_modified != 0:
        sentence_modify = "From the GPT suggestions you used, you modified most of them. Great job " \
                          "collaborating with the tool."

    elif sum_not_modified > sum_modify_gpt3 != 0:
        sentence_modify = "From the GPT suggestions you got, you used most of them as they were " \
                          "suggested. Check the sentences with orange bars on the right-plot and " \
                          "analyse if you can make them more personal or if they are completely fine " \
                          "and served its purpose. take into account that even if the suggestion was " \
                          "not modified, the sentence could still be co-authored if you added further " \
                          "text within the sentence, without modifying the actual suggestion."

    elif sum_modify_gpt3 > sum_not_modified == 0:
        sentence_modify = "From the GPT suggestions you used, you modified all of them. Great job " \
                          "collaborating with the tool."

    elif sum_not_modified > sum_modify_gpt3 == 0:
        sentence_modify = "From the GPT suggestions you used, all of them were used as suggested. " \
                          "Go back to the sentences on the right plot and analyze if potential " \
                          "changes can make the text more personalised, take into account that even " \
                          "if the suggestion was not modified, the sentence could still be co-authored " \
                          "if you added further text within the sentence, without modifying the actual " \
                          "suggestion."

    elif sum_modify_gpt3 == sum_not_modified and sum_modify_gpt3 != 0:
        sentence_modify = "From the GPT suggestions you used, you modified half of them. " \
                          "That is some good collaboration."

    elif sum_modify_gpt3 == sum_not_modified == 0:
        sentence_modify = "You did not use any GPT suggestion in your text. Is there any reason you " \
                          "did not use the available help? is it just because you did not feel you " \
                          "needed it? or you just do not like to collaborate with language models like " \
                          "GPT"

    # Scatterplot 1

    # create the scatter plot with regression line
    scatter = alt.Chart(event_seq_df).mark_circle().encode(
        x='time_dif_since_last_event',
        y='pause_outburst'
    )

    reg_line = scatter.transform_regression(
        'time_dif_since_last_event', 'pause_outburst'
    ).mark_line(color='red').properties(
        width=400,
        height=400,
        title='Pause length / Burst'
    )

    scatter_chart = (scatter + reg_line)

    # Pauses per stage

    # Compute the counts and sums
    df_counts = event_seq_df[event_seq_df['pause_num'] != 0].groupby('stage')['pause_num'].count().reset_index()
    df_sums = event_seq_df.groupby('stage')['pause_outburst2'].sum().reset_index()

    # Create the bar chart
    bar_chart = alt.Chart(df_counts).mark_bar().encode(
        x=alt.X('stage:N', title='Stage'),
        y=alt.Y('pause_num:Q', title='Pause Count')
    ).properties(
        width=400,
        height=400,
        title='Pause Count / Burst by Stage'
    )

    # Create bar chart pauses according to suggestion_

    # Filter the DataFrame to include only rows where pause is True
    filtered_df_sug_ = event_seq_df[event_seq_df['pause'] == True]
    #st.write(filtered_df_sug_)

    # Group the filtered DataFrame by cursor_sentence and suggestion_, and count the number of pause_num
    grouped_df = filtered_df_sug_.groupby(['cursor_sentence', 'suggestion_'])['pause_num'].count().reset_index()
    #st.write(grouped_df)
    grouped_df = grouped_df.rename(columns={'cursor_sentence': 'Sentence Number',
                                            'pause_num': 'Number of Pauses',
                                            'suggestion_': 'Pause Stage'})
    #st.write(grouped_df)

    # Create the Altair chart
    chart_pause1 = alt.Chart(grouped_df).mark_bar().encode(
        x='Sentence Number:O',
        y='Number of Pauses:Q',
        color='Pause Stage:N',
        tooltip=['Sentence Number', 'Pause Stage', 'Number of Pauses']
    )

    # Adjust the chart properties
    chart_pause1 = chart_pause1.properties(
        width=alt.Step(40),
        height=400
    )

    mapping = {
        'post-prompt': 'after the initial prompt',
        'post-suggestion': 'right after a suggestion',
        'pre-suggestion': 'right before a suggestion',
        'suggestion': 'during a suggestion',
        'user-input': 'while writing'
    }

    grouped_df['Pause Stage'] = grouped_df['Pause Stage'].map(mapping)

    # Calculate the total number of pauses
    total_pauses = grouped_df['Number of Pauses'].sum()

    # Generate the first sentence
    sentence_pauses1 = f"During your writing exercise, you took {total_pauses} pauses."

    # Calculate the total number of pauses per category (suggestion_)
    category_pauses = grouped_df.groupby('Pause Stage')['Number of Pauses'].sum()

    # Generate the second sentence for each category
    for suggestion, pauses in category_pauses.items():
        sentence_pauses1 += f" The total number of pauses {suggestion} was {pauses}."

    sentence_pauses1 += " Do you spot any sentence where you took multiple pauses?, go to the full text " \
                        "and analyze if there was something particularly challenging about the sentence."

    ##########
    filtered_df_pause3 = event_seq_df[(event_seq_df['pause'] == True) &
                                      (event_seq_df['suggestion_'] == 'user-input')]
    #st.write(filtered_df_pause3)
    filtered_df_pause3 = filtered_df_pause3.rename(columns={'cursor_sentence': 'Sentence Number',
                                            'time_dif_since_last_event': 'Pauses Length',
                                            'pause_outburst': 'Total Pause burst'})
    #st.write(filtered_df_pause3)
    # Calculate the sum of time_dif_since_last_event for each cursor_sentence
    bar_data_pause3 = filtered_df_pause3.groupby('Sentence Number')['Pauses Length'].sum().reset_index()
    #st.write(bar_data_pause3)
    # Calculate the sum of pause_outburst
    line_data_pause3 = \
        filtered_df_pause3.groupby('Sentence Number')['Total Pause burst'].sum().reset_index()
    #st.write(line_data_pause3)

    # Create the bar chart
    bars_pause3 = alt.Chart(bar_data_pause3).mark_bar().encode(
        x='Sentence Number:O',
        y='Pauses Length:Q'
    )

    # Create the line chart
    line_pause3 = alt.Chart(line_data_pause3).mark_line(color='red').encode(
        x='Sentence Number:O',
        y='Total Pause burst:Q'
    )

    # Combine the bar chart and line chart
    chart_pause3 = alt.layer(bars_pause3, line_pause3).resolve_scale(y='independent')

    # Adjust the chart properties
    chart_pause3 = chart_pause3.properties(
        width=alt.Step(40),
        height=400
    )

    # Calculate the total pauses duration and sentences with top 3 pause lengths
    total_pauses_duration = bar_data_pause3['Pauses Length'].sum()
    top_sentences_pauses = bar_data_pause3.nlargest(3, 'Pauses Length')['Sentence Number'].tolist()
    # Generate the first sentence
    sentence_pauses2 = f"While you were writing (not during suggestions), you took pauses that lasted in total " \
                       f"for {total_pauses_duration} seconds."

    # Append the sentences with top 3 pause lengths
    sentence_pauses2 += f" The three sentences where you spent more time pausing were: " \
                        f"{', '.join(map(str, top_sentences_pauses))}."

    # Calculate the total text inputs and sentences with top 3 pause outbursts
    total_text_inputs = line_data_pause3['Total Pause burst'].sum()
    top_sentences_outbursts = line_data_pause3.nlargest(3, 'Total Pause burst')['Sentence Number'].tolist()

    # Generate the second sentence
    sentence_pauses2 += f" All the pauses resulted in {total_text_inputs} text inputs."

    sentence_pauses2 += f" The three sentences where pauses generated more text were: " \
                        f"{', '.join(map(str, top_sentences_outbursts))}. "

    sentence_pauses2 += "Can you go back to your text and reflect if there is any particular reason " \
                        "you took those pauses? do you remember what you did when you took longer pauses, " \
                        "pay special attention to those who generated more text, what helped you get" \
                        "inspiration, was it the pause or something else?"



    ##############
    # Revision Analysis
    ##############

    # Create the line chart for revisions plot 1
    line_chart = alt.Chart(df_sums).mark_line(color='red').encode(
        x=alt.X('stage:N', title='Stage'),
        y=alt.Y('pause_outburst2:Q', title='Pause Outburst Sum')
    )

    # Create an Altair chart
    chart_rev = alt.Chart(event_seq_df).mark_line().encode(
        x='eventTimestamp',
        y='currentCursor',
        color=alt.value('blue'),
        tooltip=['eventTimestamp', 'currentCursor']
    ).properties(
        width=800,
        height=400,
        title='Revisions over time'
    ).interactive()

    # Add the maxCursor line to the chart
    max_cursor_line = alt.Chart(event_seq_df).mark_line().encode(
        x='eventTimestamp',
        y='maxCursor',
        color=alt.value('red'),
        tooltip=['eventTimestamp', 'maxCursor']
    )

    chart_revision = chart_rev + max_cursor_line

    # Create revisions per sentence plot 1

    rev_df = event_seq_df[event_seq_df['revision'] == 'revision']
    # st.write(rev_df)

    grouped_df = rev_df.groupby('cursor_sentence')['revision_number'].nunique().reset_index()
    # st.write(grouped_df)
    chart_rev1 = alt.Chart(grouped_df).mark_bar().encode(
        x='cursor_sentence:O',
        y='revision_number:Q',
        tooltip=['cursor_sentence', 'revision_number']
    ).properties(
        width=350,
        height=350,
        title='Number of revisions per sentence'
    )

    # Create revisions per sentence plot 2
    rev_df2 = event_seq_df[event_seq_df['revision'] == 'revision']
    # st.write(rev_df2)

    grouped_df2 = rev_df2.groupby('cursor_sentence')['revision_number'].count().reset_index()
    # st.write(grouped_df2)
    chart_rev2 = alt.Chart(grouped_df2).mark_circle().encode(
        x='cursor_sentence:O',
        y='revision_number:Q',
        size='revision_number:Q',
        tooltip=['cursor_sentence', 'revision_number']
    ).properties(
        width=350,
        height=350,
        title='Length of revisions per sentence'
    )

    # Display the plots side by side
    combined_chart = alt.hconcat(chart_rev1, chart_rev2)

    ##############################
    # TEST APP
    ##############################

    st.write(
        f'Successfully analysed {len(events)} events from your writing session. '
        f'Events refer to every action you took when writing this piece, '
        f'insertions, deletions, moving your cursor to a different section of '
        f'the text, opening a suggestion and selecting it or dismissing it, are all counted '
        f'as events. As you can see, a lot of actions were taken to produce your text '
        f'( {len(events)} ).')
    # Reproduce text
    st.write('Take a look at some of the summary statistics from your text, this will give you an '
             'initial idea of your writing process.')

    # Summary statistics

    st.header('Summary statistics')
    # Create two columns with equal width
    col1, col2 = st.columns(2)
    # Display sentence_metrics in the first column
    with col1:
        st.subheader("Sentence Level Metrics:")
        for ele in sentence_metrics:
            st.write(ele, ":", sentence_metrics[ele])

    # Display api_metrics in the second column
    with col2:
        st.subheader("GPT-3 Suggestion Metrics:")
        for ele in api_metrics:
            st.write(ele, ":", api_metrics[ele])

    st.write("\n\n")
    st.write(
        'Before getting into further analysis, take some time to read your text from start to finish, and come with your own'
        ' conclusions about it, hover the sentences to get additional insights.')

    st.header('Final text')
    # play(buffer=text_buffer, speed="instant")
    #
    # Filter the DataFrame to include only rows where event_name is 'system-initialize'
    filtered_df_prompt = event_seq_df[event_seq_df['event_name'] == 'system-initialize']

    # Check if there is any row matching the condition
    if not filtered_df_prompt.empty:
        # Get the value of the currentDoc column from the first row
        current_doc_value = filtered_df_prompt.iloc[0]['currentDoc']

        # Print the value in Streamlit
        st.write(current_doc_value)

    # Reproduce text plus annotations.
    # Define CSS styles for different authorships
    authorship_styles = {
        'Co-Authored sentence': 'background-color: yellow;',
        'original sentence': 'background-color: cyan;',
        'GPT-3 sentence': 'background-color: magenta;',
    }

    for i, row in sentences_df.iterrows():
        sentence = row['num_sent']
        pauses = row['pauses_over_10_sec']
        pause_burst = row['pause_burst']
        authorship = row['authorship']
        time_spent = row['time_spent']
        revisions = row['revisions']
        sentiment = row['sentiment_label2']
        last_sentence = row['sentence_current'] #Check this.

        # Get the CSS style for the current authorship
        authorship_style = authorship_styles.get(authorship, '')

        # Use st.markdown to add interactive tags to the sentence with the CSS style
        st.markdown(
            f"<span title='Sentence #: {sentence}. Authorship: {authorship}. "
            f"Time_spent = {time_spent}. Sentiment = {sentiment}. "
            f"Pauses: {pauses}. pause_burst: {pause_burst}. revisions: {revisions}' "
            f"style='{authorship_style}'>{last_sentence}</span>",
            unsafe_allow_html=True
        )

    st.header('Dashboard and Analysis')
    st.write('Now that you have read your text and have seen some insights about it, have you discovered '
             'anything interesting? Up next we will see different visualisations along with feedback '
             'to further explore your writing process.')

    st.subheader('Sentence Authorship')
    st.altair_chart(pie, use_container_width=True)
    st.write(sentence_pie)

    ###############################
    ########GPT SUGGESTION ANALYSIS
    ###############################

    st.subheader('GPT-3 Suggestions Analysis')
    # Display the chart in Streamlit
    st.altair_chart(alt.hconcat(chart1, chart2))
    st.write(sentence_call)
    st.write(sentence_modify)

    st.subheader('Pauses Analysis')
    # Display charts side-by-side
    #col1, col2 = st.columns(2)  #

    #with col1:
     #   st.altair_chart(scatter_chart, use_container_width=True)

    #with col2:
     #   st.altair_chart(bar_chart + line_chart, use_container_width=True)

    # Display the charts side by side using Streamlit's columns
    #col1, col2 = st.columns(2)
    #with col1:
    #    st.altair_chart(chart_pause1)

    #with col2:
     #   st.altair_chart(chart_pause3)

    st.write(chart_pause1)
    st.write(sentence_pauses1)
    st.write(chart_pause3)
    st.write(sentence_pauses2)


    # Line chart revision
    st.subheader('Revision Analysis')
    # Use Streamlit to show chart_rev
    st.altair_chart(chart_revision)
    # Use Streamlit to show chart_rev2
    st.altair_chart(combined_chart)

    st.write('#####DATA TO TEXT FEEDBACK/INSIGHTS')

    # Delete
    st.subheader('event_seq_df')
    st.write(event_seq_df)
    st.write(event_seq_df.index.nlevels)
    st.write(event_seq_df.index.names)
    st.subheader('author_sent_df')
    st.write(author_sent_df)
    st.subheader('sentences_df')
    st.write(sentences_df)
    st.subheader('event_seq_dict')
    st.write(event_seq_dict)
    st.subheader('dfe')
    st.write(dfe)
    st.subheader('summary_df1')
    st.write(summary_df1)
    st.subheader('summary_df2')
    st.write(summary_df2)
    st.subheader('summary_df_merged')
    st.write(summary_df_merged)
    st.subheader('dfe_exploded')
    st.write(dfe_exploded)
    st.subheader('counts')
    st.write(counts)
    st.subheader('output')
    st.write(output)


if __name__ == "__main__":
    main()
