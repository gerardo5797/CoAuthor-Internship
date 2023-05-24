import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import math
import seaborn as sns
import os
import json
import glob
from glob import glob

metrics_df = pd.read_csv('allmetrics.csv')
metrics_df.columns = [c.lower().replace(' ', '_') for c in metrics_df.columns]
metrics_df.columns = [c.lower().replace('-', '_') for c in metrics_df.columns]
print(metrics_df.head(5))
events_df = pd.read_csv('output.csv')
print(events_df)

sent_num = metrics_df.total_number_of_sentences[0]
sent_num = str(sent_num)

init_prompt_sent_num = metrics_df.number_of_sentences_of_initial_prompt[0]
comp_auth_sent_num = metrics_df.number_of_sentences_completely_authored_by_the_user[0]
comp_gpt_sent_num = metrics_df.number_of_sentences_completely_authored_by_gpt_3[0]
col_auth_gpt_sent_num = metrics_df.number_of_sentences_authored_by_gpt_3_and_user[0]

# Pie chart 1

pie_chart_data = pd.DataFrame({
    'Source': ['Initial Prompt', 'Authored by User', 'Authored by GPT-3',
               'Collaboratively Authored'],
    'Sentences': [init_prompt_sent_num, comp_auth_sent_num, comp_gpt_sent_num, col_auth_gpt_sent_num]
})

pie_chart1 = alt.Chart(pie_chart_data).mark_arc().encode(
    theta="Sentences",
    color="Source"
)

# Pie chart 2
gpt_used = metrics_df.number_of_times_gpt_3_suggestion_is_used[0]
gpt_rejected = metrics_df.number_of_times_user_rejected_gpt_3_suggestion[0]
gpt_calls = metrics_df.total_number_of_gpt_3_calls_made[0]


pie_chart_data2 = pd.DataFrame({
    'Used': ['GPT-3 suggestion used', 'GPT-3 suggestion rejected'],
    'Times': [gpt_used, gpt_rejected]
})

pie_chart2 = alt.Chart(pie_chart_data2).mark_arc().encode(
    theta="Times",
    color="Used"
)
# Pie Chart 3

gpt_modified = metrics_df.number_of_times_gpt_3_suggestion_is_modified[0]
gpt_as_is = metrics_df.number_of_times_gpt_3_suggestion_is_used_as_is[0]


pie_chart_data3 = pd.DataFrame({
    'Used': ['GPT-3 suggestion modified', 'GPT-3 suggestion used as is'],
    'Times': [gpt_modified, gpt_as_is]
})

pie_chart3 = alt.Chart(pie_chart_data3).mark_arc().encode(
    theta="Times",
    color="Used"
)

# Bar chart 1
suggest_1 = metrics_df.suggest_select_stage1[0]
suggest_2 = metrics_df.suggest_select_stage2[0]
suggest_3 = metrics_df.suggest_select_stage3[0]

bar_chart_data1 = pd.DataFrame({
    'Used': ['Stage 1', 'Stage 2', 'Stage 3'],
    'Times': [suggest_1, suggest_2, suggest_3]
})

bar_chart1 = alt.Chart(bar_chart_data1, width=500).mark_bar(size=50).encode(
    y="Times",
    x="Used"
)



# Bar CHart 2

pivot = events_df.pivot_table(index='stage', values='pause', aggfunc='sum')

bar_chart2 = alt.Chart(pivot.reset_index(), width = 500).mark_bar(size=50).encode(
    x='stage:N',
    y='pause:Q'
)

# Scatterplot 1

# filter rows where pause_num is greater than 0
events_df_sub1 = events_df[events_df['pause_num'] > 0]

# create scatterplot

scatterplot1 = alt.Chart(events_df_sub1).mark_circle(size=60).encode(
    x='time_dif_since_last_event',
    y='pause_outburst',
    tooltip=['pause_num','time_dif_since_last_event', 'pause_outburst']
)

# End
mins_spent = metrics_df.time_spent_mins[0]


rev_rate = metrics_df.revision_rate[0]
rev_rate = str(rev_rate)

st.title('Writing analysis Dashboard')

st.header('Find your analysis below')

st.markdown("Your text contained: " + str(sent_num) + ' sentences ' +
            'it was produced in ' + str(mins_spent) + ' minutes')

st.altair_chart(pie_chart1)

st.markdown('From those ' + f"**{sent_num}**" + ' sentences, ' +
            f"**{init_prompt_sent_num}**" + ' were provided on the initial prompt, ' +
            f"**{comp_auth_sent_num}**" + ' were completely authored by you, '
            f"**{comp_gpt_sent_num}**" + ' were completely authored by GPT,' +
            'and 'f"**{col_auth_gpt_sent_num}**" + ' were authored in collaboration between you and GPT')

st.altair_chart(pie_chart2)

st.markdown('From the ' + f"**{gpt_calls}**" + ' GPT calls you made , ' +
            f"**{gpt_used}**" + ' were used, while ' +
            f"**{gpt_rejected}**" + ' were rejected. ')

st.altair_chart(pie_chart3)

st.markdown('From the ' + f"**{gpt_used}**" + ' GPT suggestions were used , ' +
            f"**{gpt_modified}**" + ' were modified, while ' +
            f"**{gpt_as_is}**" + ' were used exactly as suggested. ')

st.altair_chart(bar_chart1)

st.markdown('From the ' + f"**{gpt_used}**" + ' GPT suggestions were used , ' +
            f"**{suggest_1}**" + ' were used during the first stage of your writing ' +
            f"**{suggest_2}**" + ' were used during the second stage and ' +
            f"**{suggest_3}**" + ' during the last stage. ')

st.altair_chart(bar_chart2)

st.altair_chart(scatterplot1)



#st.markdown("Your revision rate is: " + str(rev_rate))





