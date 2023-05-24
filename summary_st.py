import pandas as pd
import streamlit as st


def stats(event_seq_dict):
    total_api_calls = 0
    num_gpt3_used = 0
    num_gpt3_not_satsfd = 0
    num_gpt3_mods = 0
    num_gpt3_non_mods = 0

    total_num_sent = event_seq_dict["num_sent"][-1]
    num_user_authored = 0
    num_gpt3_authored = 0
    num_prompt = event_seq_dict["num_sent"][0]

    for seq in event_seq_dict["sequence"]:

        # if ("gpt3-call" not in seq) and ("empty-call" not in seq):
        if ("gpt3-call" not in seq) and ("prompt" not in seq) and ("user" in seq):
            num_user_authored += 1
        if ("gpt3-call" in seq) and ("user" not in seq):
            if ("prompt" not in seq) and ("modify-gpt3" not in seq):
                num_gpt3_authored += 1

        for event in seq:
            if event == "gpt3-call":
                num_gpt3_used += 1
            if event == "empty-call":
                num_gpt3_not_satsfd += 1
            if event == "modify-gpt3":
                num_gpt3_mods += 1

    total_api_calls = num_gpt3_not_satsfd + num_gpt3_used
    num_gpt3_non_mods = num_gpt3_used - num_gpt3_mods
    num_gpt3_auth_user_mod = total_num_sent - \
                             num_gpt3_authored - num_user_authored - num_prompt

    sentence_metrics = {
        "Sentences": total_num_sent,
        "Sentences initial prompt": num_prompt,
        "Original sentences": num_user_authored,
        "GPT-3 sentences": num_gpt3_authored,
        "Co-authored sentences": num_gpt3_auth_user_mod,
    }

    api_metrics = {
        "GPT-3 calls made": total_api_calls,
        "Times GPT-3 suggestion was used": num_gpt3_used,
        "Times GPT-3 suggestion was rejected": num_gpt3_not_satsfd,
        "Times GPT-3 suggestion was modified": num_gpt3_mods,
        "Times GPT-3 suggestion was used as suggested": num_gpt3_non_mods,
    }

    return sentence_metrics, api_metrics


def get_summary_stats(event_seq_dict):
    sentence_metrics, api_metrics = stats(event_seq_dict)

    sentence_metrics2 = pd.DataFrame.from_dict(sentence_metrics, orient='index', columns=['value'])

    sentence_metrics2 = sentence_metrics2.transpose()

    api_metrics2 = pd.DataFrame.from_dict(api_metrics, orient='index', columns=['value'])
    api_metrics2 = api_metrics2.transpose()

    merged_metrics = pd.merge(sentence_metrics2, api_metrics2, left_index=True, right_index=True)

    return sentence_metrics, api_metrics, merged_metrics


def get_summary_stats2(df):
    summary_df2 = df.groupby(df.file_name).agg(
        pauses_over_10_sec=('pause_num', lambda x: (x > 0).sum()),
        suggest_accepted_stage1=('event_name', lambda x: (
                (x == 'suggestion-select') & (
                df['stage'] == 1)).sum()),
        suggest_accepted_stage2=('event_name', lambda x: (
                (x == 'suggestion-select') & (
                df['stage'] == 2)).sum()),
        suggest_accepted_stage3=('event_name', lambda x: (
                (x == 'suggestion-select') & (
                df['stage'] == 3)).sum()),
        suggest_opened_stage1=('event_name', lambda x: (
                (x == 'suggestion-open') & (
                df['stage'] == 1)).sum()),
        suggest_opened_stage2=('event_name', lambda x: (
                (x == 'suggestion-open') & (
                df['stage'] == 2)).sum()),
        suggest_opened_stage3=('event_name', lambda x: (
                (x == 'suggestion-open') & (
                df['stage'] == 3)).sum()),
        time_spent_mins=(
            'eventTimestamp', lambda x: ((x.max() - x.min()) / 60000)),
        max_text=('currentCursor', lambda x: (max(x))),
        # Change, sometime the max num is not the final num,
        # since there might be deletions at the end.
        delete_count=('delete', lambda x: (x > 0).sum())
    )
    # summary_df2['suggest_select_stage1_perc'] = summary_df2['suggest_select_stage1'] / (
    #       summary_df2['suggest_select_stage1'] + summary_df2['suggest_select_stage2'] + summary_df2[
    #  'suggest_select_stage3'])
    # summary_df2['suggest_select_stage1_perc'] = summary_df2['suggest_select_stage1_perc'].fillna(0)

    # summary_df2['suggest_select_stage2_perc'] = summary_df2['suggest_select_stage2'] / (
    #       summary_df2['suggest_select_stage1'] + summary_df2['suggest_select_stage2'] + summary_df2[
    #  'suggest_select_stage3'])
    # summary_df2['suggest_select_stage2_perc'] = summary_df2['suggest_select_stage2_perc'].fillna(0)

    # summary_df2['suggest_select_stage3_perc'] = summary_df2['suggest_select_stage3'] / (
    #       summary_df2['suggest_select_stage1'] + summary_df2['suggest_select_stage2'] + summary_df2[
    #  'suggest_select_stage3'])
    # summary_df2['suggest_select_stage3_perc'] = summary_df2['suggest_select_stage3_perc'].fillna(0)

    summary_df2['revision_rate'] = \
        summary_df2['delete_count'] / (summary_df2['delete_count'] + summary_df2['max_text'])
    summary_df2['revision_rate'] = \
        summary_df2['revision_rate'].fillna(0)

    summary_df2['suggest_dismissed_stage1'] = \
        summary_df2['suggest_opened_stage1'] - summary_df2['suggest_accepted_stage1']
    summary_df2['suggest_dismissed_stage2'] = \
        summary_df2['suggest_opened_stage2'] - summary_df2['suggest_accepted_stage2']
    summary_df2['suggest_dismissed_stage3'] = \
        summary_df2['suggest_opened_stage3'] - summary_df2['suggest_accepted_stage3']

    return summary_df2
