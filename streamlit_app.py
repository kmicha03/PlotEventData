#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:10:03 2023

@author: konstantinosmichail
"""

from highlight_text.htext import AnnotationBbox
from matplotlib.offsetbox import OffsetImage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
from PIL import Image


import streamlit as st
import types
import psycopg2
from mplsoccer import Pitch, VerticalPitch, FontManager
from matplotlib import rcParams

import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from matplotlib import cm
from highlight_text import fig_text, ax_text
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties


from mplsoccer import VerticalPitch

supabase_password = "Gamwtoapoel99!"
project_url = "https://xkzfeabisrfkyotvpozu.supabase.co"

conn = psycopg2.connect(
    user="postgres",
    password=supabase_password,
    host="db.xkzfeabisrfkyotvpozu.supabase.co",
    port=5432,
    database="postgres"
)


def get_available_leagues():
    cursor = conn.cursor()

    cursor.execute("""
            Select
              name,id
            from
              "leagues"
            
        """)

    result = cursor.fetchall()
    
    if result:
        leagues = [leagues for leagues in result]
        return leagues
    else:
        return None
    
def get_league_teams(selected_league):
    league_id = league_mapping[selected_league]
    cursor = conn.cursor()

    cursor.execute("""
            Select
              name,id
            from
              "teams"
            where
              teams.league_id=%s
            
        """, (league_id,))

    result = cursor.fetchall()
    
    if result:
        teams = [teams for teams in result]
        return teams
    else:
        return None
    
def get_players(selected_team):
    team_id = teams_mapping[selected_team]

    cursor = conn.cursor()

    cursor.execute("""
            Select
              player_name,player_id,game_id,minutes_played,starting_position
            from
              "Players"
            where
              "Players"."team_id"=%s
            
        """, (team_id,))

    result = cursor.fetchall()
    
    if result:
        df = pd.DataFrame(result, columns=['player_name','player_id', 'game_id', 'minutes_played','starting_position'])
        return df
    else:
        return None

available_leagues=get_available_leagues()

# Create a dictionary to map displayed names to actual values
league_mapping = {name: league_id for name, league_id in available_leagues}

# Use the displayed names for the dropdown
available_leagues = list(league_mapping.keys())

# Function to query data based on the selected league
@st.cache_data(hash_funcs={types.FunctionType: id})
def load_matches(selected_player, selected_positions):
    # Use the actual value (league_id) instead of the displayed name
    player_id = str(players[players['player_name'] == selected_player]["player_id"].iloc[0])
    positions_tuple = tuple(selected_positions) if selected_positions else tuple([''])  # Fallback to a dummy value if positions is empty

    positions_array_str = "{" + ",".join(positions_tuple) + "}"

    cursor = conn.cursor()

    cursor.execute("""
      select
        m.date,
        m.id,
        t.name as home_team_name,
        t2.name as away_team_name
      from
        "Matches" m
        inner join "Players" p on m.id = p.game_id
        inner join "teams" t on m.home_team_id = t.id
        inner join "teams" t2 on m.away_team_id = t2.id
      where
        p.player_id = %s AND minutes_played>0 AND p.starting_position = ANY(%s);
      """, (player_id,positions_array_str,))

    result = cursor.fetchall()

    # Convert the result to a DataFrame
    df = pd.DataFrame(result, columns=['date','match_id', 'home_team_name', 'away_team_name'])

    return df

# App title
st.title("Football Analytics - Event Data")

st.sidebar.header('Filters')
selected_league = st.sidebar.selectbox("Select a League", available_leagues)

# Create a dictionary to map displayed names to actual values
available_teams = get_league_teams(selected_league)

teams_mapping = {name: team_id for name, team_id in available_teams}

# Use the displayed names for the dropdown
available_teams = list(teams_mapping.keys())

selected_team = st.sidebar.selectbox("Select a team", available_teams)

players = get_players(selected_team)

# Group by player name and calculate the sum of minutes played
player_minutes_df = players.groupby('player_name')['minutes_played'].sum()

# Resetting the index to get a clean DataFrame
player_minutes_df = player_minutes_df.reset_index()

# Use the displayed names for the dropdown
available_players = players['player_name'].unique().tolist()

selected_player = st.sidebar.selectbox("Select a player", available_players)

positions = players[players['player_name']==selected_player]['starting_position'].unique().tolist()

selected_positions = st.sidebar.multiselect("Select player positions", positions, positions)

# Load and display data based on the selected league
matches = load_matches(selected_player,selected_positions)

# Create a new column 'Match' with the desired format
matches['Match'] = matches['home_team_name'] + ' vs ' + matches['away_team_name'] + ' - ' + matches['date'].astype(str)

# Extract the 'Match' column from the DataFrame as the options for multiselect
match_options = matches['Match'].tolist()

selected_matches = st.sidebar.multiselect('Matches', match_options, match_options)

# Create a mapping from 'Match' string to 'match_id'
match_id_mapping = {row['Match']: row['match_id'] for index, row in matches.iterrows()}

# Get selected match IDs from selected matches
selected_match_ids = [match_id_mapping[match] for match in selected_matches]

def get_player_events(selected_player, match_ids):
    player_id = str(players[players['player_name'] == selected_player]["player_id"].iloc[0])
    match_ids_tuple = tuple(match_ids)  # Convert list to tuple for SQL query

    cursor = conn.cursor()

    query = """
        SELECT period_id, time_seconds, start_x, end_x, start_y, end_y,type_name,result_name,bodypart_name,"EPV","xT_value"
        FROM "Events"
        WHERE "player_id" = %s AND "game_id" IN %s;
    """

    cursor.execute(query, (player_id, match_ids_tuple))

    result = cursor.fetchall()

    # Convert the result to a DataFrame
    df = pd.DataFrame(result, columns=['period_id', 'time_seconds', 'start_x', 'end_x', 'start_y', 'end_y','type_name','result_name','bodypart_name','EPV','xT_value'])  # Add your columns here

    return df

# Call the function with the selected player and match IDs

if len(selected_match_ids)>0:
  events_df = get_player_events(selected_player, selected_match_ids)

  unique_type_names = events_df['type_name'].unique().tolist()

  if 'GK' not in selected_positions:
    # Filter out event types starting with "keeper"
    unique_type_names = [event for event in unique_type_names if not event.startswith('keeper')]

  selected_type_name = st.sidebar.selectbox("Select an Event Type", unique_type_names)

event_results = events_df[events_df["type_name"] == selected_type_name]['result_name'].unique().tolist()

event_result = st.sidebar.multiselect("Select Result Type", event_results, event_results)

create_plot_button = st.sidebar.button("Create Plot")

if create_plot_button:
    
    minutes_played = player_minutes_df[player_minutes_df["player_name"] == selected_player]["minutes_played"].iloc[0]
    event_type_correct_name = selected_type_name.replace('_', ' ').title()
    selected_positions_correct_name = ','.join(selected_positions)
    # Create a dynamic title
    plot_title = f"{selected_player} ({selected_positions_correct_name}) - {selected_team}"
    plot_title2 = f"{minutes_played} Minutes Played - 2023/24"
    #st.title(plot_title)

    # Create and customize the plot
    pitch = VerticalPitch(
        pitch_type='custom',
        goal_type='box',
        linewidth=1.25,
        line_color='white',
        pitch_color='#12130e',
        pitch_length=105, 
        pitch_width=68 
    )

    # Create subplots
    fig, ax = plt.subplots(figsize=(14, 12))
    ax1 = plt.subplot2grid(shape=(3, 3), loc=(1, 0), rowspan=3, colspan=1) 
    
    robotto_regular = FontManager()

    fig.set_facecolor('#12130e')

    fig_text(
        x=0.13, y=.60,
        s=plot_title,
        va="bottom", ha="left",
        fontsize=12, color="white", font="DM Sans", weight="bold"
    )
    fig_text(
        x=0.13, y=.585,
        s=plot_title2,
        va="bottom", ha="left",
        fontsize=9, color="#B2BEB5", font="Karla"
    )
    map_title = f"{event_type_correct_name} Map"

    fig_text(
        x=0.24, y=.56,
        s=map_title,
        va="bottom", ha="center",
        fontsize=12, color="white", font="DM Sans", weight="bold"
    )

    fig_text(
        x=0.225, y=.149,
        s="Created by @kmicha03",
        va="bottom", ha="left",
        fontsize=10, color="white", font="Karla"
    )


    pitch_ax1 = VerticalPitch(pitch_type='custom', pitch_color='#12130e', line_color='#B2BEB5', half=False,pitch_length=105, pitch_width=68 )
    pitch_ax1.draw(ax=ax1)

    # Load the image
    image_path = f"/Users/konstantinosmichail/Club Logos/{selected_team}_logo.png"
    image = plt.imread(image_path)

    imagebox = OffsetImage(image, zoom=0.15, resample=True, alpha=0.6)
    ab = AnnotationBbox(imagebox, (0.5, 0.7), frameon=False, boxcoords="axes fraction", pad=0.0)
    ax1.add_artist(ab)

    filtered_events = events_df[(events_df['type_name'] == selected_type_name) & (events_df['result_name'].isin(event_result))]
    mask_complete = filtered_events.result_name.isin(["success"])

    colour_success = '#0BDA51'
    colour_fail = '#BA4F45'

         # Filter the DataFrame based on the selected event type
    if ((selected_type_name == 'throw_in') | (selected_type_name == 'cross') | (selected_type_name == 'pass') | (selected_type_name == 'shot') 
        | (selected_type_name == 'freekick_short') | (selected_type_name == 'corner_crossed') | (selected_type_name == 'freekick_crossed') 
        | (selected_type_name == 'corner_short') | (selected_type_name == 'shot_freekick') | (selected_type_name == 'shot_corner') | (selected_type_name == 'goalkick')):
            
      # Plot the completed passes
      pitch_ax1.lines(filtered_events[mask_complete].start_x, filtered_events[mask_complete].start_y,
                          filtered_events[mask_complete].end_x, filtered_events[mask_complete].end_y,
                          lw=3, transparent=True, comet=True, label=f'Successful {event_type_correct_name}',
                          color=colour_success, ax=ax1)

        # Plot the other passes
      pitch_ax1.lines(filtered_events[~mask_complete].start_x, filtered_events[~mask_complete].start_y,
                          filtered_events[~mask_complete].end_x, filtered_events[~mask_complete].end_y,
                          lw=3, transparent=True, comet=True, label=f'Unsuccessful {event_type_correct_name}',
                          color=colour_fail, ax=ax1,alpha=0.7)
      
      pitch_ax1.scatter(filtered_events[mask_complete].end_x, filtered_events[mask_complete].end_y,
                      ax=ax1, color=colour_success, s=15)
      
      pitch_ax1.scatter(filtered_events[~mask_complete].end_x, filtered_events[~mask_complete].end_y,
                      ax=ax1, color=colour_fail,s=15)
        
          
    elif ((selected_type_name == 'dribble')):

      pitch_ax1.lines(filtered_events[mask_complete].start_x, filtered_events[mask_complete].start_y,
                          filtered_events[mask_complete].end_x, filtered_events[mask_complete].end_y,
                          lw=3, transparent=True, comet=True, label=f'Successful {event_type_correct_name}',
                          color=colour_success, ax=ax1)
      pitch_ax1.lines(filtered_events[~mask_complete].start_x, filtered_events[~mask_complete].start_y,
                          filtered_events[~mask_complete].end_x, filtered_events[~mask_complete].end_y,
                          lw=3, transparent=True, comet=True, label=f'Unsuccessful {event_type_correct_name}',
                          color=colour_fail, ax=ax1,alpha=0.7)
      
      pitch_ax1.scatter(filtered_events[mask_complete].end_x, filtered_events[mask_complete].end_y,
                      ax=ax1, color=colour_success, s=15)
      
      pitch_ax1.scatter(filtered_events[~mask_complete].end_x, filtered_events[~mask_complete].end_y,
                      ax=ax1, color=colour_fail,s=15)
          
    elif ((selected_type_name == 'take_on') | (selected_type_name == 'keeper_claim')):

      pitch_ax1.scatter(filtered_events[mask_complete].start_x, filtered_events[mask_complete].start_y,
                      ax=ax1, color=colour_success,s=15, label=f"Successful {event_type_correct_name}")
      pitch_ax1.scatter(filtered_events[~mask_complete].start_x, filtered_events[~mask_complete].start_y,
                      ax=ax1, color=colour_fail,s=15, label = f"Unsuccessful {event_type_correct_name}")
          
    elif ((selected_type_name == 'interception') | (selected_type_name == 'clearance') | (selected_type_name == 'tackle') 
          | (selected_type_name == 'keeper_pick_up') | (selected_type_name == 'keeper_save') | (selected_type_name == 'keeper_punch')):

      pitch_ax1.scatter(filtered_events[mask_complete].start_x, filtered_events[mask_complete].start_y,
                      ax=ax1, color=colour_success, s=15, label = f"Successful {event_type_correct_name}")
      
    elif ((selected_type_name == 'bad_touch') | (selected_type_name == 'foul')):

      pitch_ax1.scatter(filtered_events[mask_complete].start_x, filtered_events[mask_complete].start_y,
                      ax=ax1, color=colour_fail, s=15, label = f"{event_type_correct_name}")
    # Display the plot in Streamlit
    legend = ax1.legend(facecolor='#B2BEB5', edgecolor='None', fontsize=7, loc='upper left', handlelength=1)
    for text in legend.get_texts():
      text.set_color('#FFFFFF')

    st.pyplot(fig)