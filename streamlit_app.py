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
from mplsoccer import Pitch, VerticalPitch, FontManager, add_image
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

supabase_password = st.secrets["db_password"]
project_url = st.secrets["db_project_url"]

conn = psycopg2.connect(
    user=st.secrets["db_user"],
    password=supabase_password,
    host= st.secrets["db_host"],
    port=st.secrets["db_port"],
    database=st.secrets["db_name"]
)


def get_available_leagues():
    cursor = conn.cursor()

    cursor.execute("""
            Select
              name,id
            from
              "leagues"
            where
               active=1
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
st.subheader('Created by Konstantinos Michail (@kmicha03)\nData from Opta')


with st.expander('Instructions'):
    st.write('''
      1) Choose Your League: Begin by selecting from our list of available leagues.
      2) Pick Your Team: Next, identify and select the team you are interested in.
      3) Select a Player: Choose a player from your chosen team to focus on.
      4) Filter by Position: Refine your search by selecting the player's position. This will tailor the events to that specific role.
      5) Match Selection: By default, all matches are selected. You can customize this by adding or removing matches from your analysis.
      6) Event Type Selection: Choose the type of event you're interested in exploring further.
      7) Result Filtering: Lastly, narrow down your results by filtering through the different outcomes of the selected event.
    ''')

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
        SELECT period_id, time_seconds, start_x, end_x, start_y, end_y,type_name,result_name,bodypart_name,"xT_value",open_play_assist,set_piece_assist,goal_creating_action,shot_creating_action
        FROM "Events"
        WHERE "player_id" = %s AND "game_id" IN %s;
    """

    cursor.execute(query, (player_id, match_ids_tuple))

    result = cursor.fetchall()

    # Convert the result to a DataFrame
    df = pd.DataFrame(result, columns=['period_id', 'time_seconds', 'start_x', 'end_x', 'start_y', 'end_y','type_name','result_name','bodypart_name','xT_value','open_play_assist','set_piece_assist','goal_creating_action','shot_creating_action'])  # Add your columns here

    return df

# Call the function with the selected player and match IDs

if len(selected_match_ids)>0:
  events_df = get_player_events(selected_player, selected_match_ids)
    
  unique_type_names = events_df['type_name'].unique().tolist()

  custom_metrics = ["Goal","Open Play Assist","Set-Piece Assist","Most Dangerous Passes"
                           ,"Attacking Third Passes", "Attacking Third Carries", "Progressive Passes","Progressive Carries"]
    #"Goal Creating Actions","Shot Creating Actions",
  unique_type_names.extend(custom_metrics)
  # Generate human-readable names for the event types
  unique_type_names_readable = [name.replace('_', ' ').title() for name in unique_type_names]

  # Create a mapping from the readable names back to the original technical names
  type_name_mapping = dict(zip(unique_type_names_readable, unique_type_names))

  if 'GK' not in selected_positions:
    # Filter out event types starting with "keeper"
    unique_type_names = [event for event in unique_type_names if not event.startswith('keeper')]

  selected_type_name_readable = st.sidebar.selectbox("Select an Event Type", unique_type_names_readable)
  selected_type_name = type_name_mapping[selected_type_name_readable]

  if selected_type_name not in custom_metrics:
    event_results = events_df[events_df["type_name"] == selected_type_name]['result_name'].unique().tolist()
    event_result = st.sidebar.multiselect("Select Result Type", event_results, event_results)
      
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
    
    #robotto_regular = FontManager()
    
  fig, axs = pitch.grid(endnote_height=0.03, endnote_space=0,
                      title_height=0.08, title_space=0,
                      # Turn off the endnote/title axis. I usually do this after
                      # I am happy with the chart layout and text placement
                      axis=False,
                      grid_height=0.84)
    
    # endnote and title
  axs['endnote'].text(1, 0.5, '@kmicha03', va='center', ha='right', fontsize=15,color='#dee6ea')
    
  # Assuming 'axs' is a dictionary of axes, if 'title' is an axis dedicated for the title, use it like this:
  axs['title'].text(0, 0.9, plot_title, color='#dee6ea', va='center', ha='left', fontsize=20, weight='bold')
  axs['title'].text(0, 0.55, plot_title2, color='#dee6ea', va='center', ha='left', fontsize=17)
    
  axs['pitch'].set_title(f"{event_type_correct_name} Map", color='white', va='center', ha='center', fontsize=18, weight='bold')
    
    #robotto_regular = FontManager()
    
  fig.set_facecolor('#12130e')
    
    # Load the image
  image_path = f"Club Logos/{selected_team}_logo.png"
  image = plt.imread(image_path)
    
  ax_image = add_image(image, fig, left=0.91, bottom=0.88, width=0.12,
                interpolation='hanning')
    
  colour_success = '#0BDA51'
  colour_fail = '#BA4F45'
    
          # Filter the DataFrame based on the selected event type
  if ((selected_type_name == 'throw_in') | (selected_type_name == 'cross') | (selected_type_name == 'pass') | (selected_type_name == 'shot') 
      | (selected_type_name == 'freekick_short') | (selected_type_name == 'corner_crossed') | (selected_type_name == 'freekick_crossed') 
      | (selected_type_name == 'corner_short') | (selected_type_name == 'shot_freekick') | (selected_type_name == 'shot_corner') | (selected_type_name == 'goalkick')):
          
    filtered_events = events_df[(events_df['type_name'] == selected_type_name) & (events_df['result_name'].isin(event_result))]
    mask_complete = filtered_events.result_name.isin(["success"])
    # Plot the completed passes
    pitch.lines(filtered_events[mask_complete].start_x, filtered_events[mask_complete].start_y,
                        filtered_events[mask_complete].end_x, filtered_events[mask_complete].end_y,
                        lw=2, transparent=True, comet=True, label=f'Successful {event_type_correct_name}',
                        color=colour_success, ax=axs['pitch'])

      # Plot the other passes
    pitch.lines(filtered_events[~mask_complete].start_x, filtered_events[~mask_complete].start_y,
                        filtered_events[~mask_complete].end_x, filtered_events[~mask_complete].end_y,
                        lw=2, transparent=True, comet=True, label=f'Unsuccessful {event_type_correct_name}',
                        color=colour_fail, ax=axs['pitch'],alpha=0.7)
    
    pitch.scatter(filtered_events[mask_complete].end_x, filtered_events[mask_complete].end_y,
                    ax=axs['pitch'], color=colour_success, s=15)
    
    pitch.scatter(filtered_events[~mask_complete].end_x, filtered_events[~mask_complete].end_y,
                    ax=axs['pitch'], color=colour_fail,s=15)
      
        
  elif ((selected_type_name == 'dribble')):

    filtered_events = events_df[(events_df['type_name'] == selected_type_name) & (events_df['result_name'].isin(event_result))]
    mask_complete = filtered_events.result_name.isin(["success"])

    pitch.lines(filtered_events[mask_complete].start_x, filtered_events[mask_complete].start_y,
                        filtered_events[mask_complete].end_x, filtered_events[mask_complete].end_y,
                        lw=2, transparent=True, comet=True, label=f'Successful {event_type_correct_name}',
                        color=colour_success, ax=axs['pitch'])
    pitch.lines(filtered_events[~mask_complete].start_x, filtered_events[~mask_complete].start_y,
                        filtered_events[~mask_complete].end_x, filtered_events[~mask_complete].end_y,
                        lw=2, transparent=True, comet=True, label=f'Unsuccessful {event_type_correct_name}',
                        color=colour_fail, ax=axs['pitch'],alpha=0.7)
    
    pitch.scatter(filtered_events[mask_complete].end_x, filtered_events[mask_complete].end_y,
                    ax=axs['pitch'], color=colour_success, s=15)
    
    pitch.scatter(filtered_events[~mask_complete].end_x, filtered_events[~mask_complete].end_y,
                    ax=axs['pitch'], color=colour_fail,s=15)
        
  elif ((selected_type_name == 'take_on') | (selected_type_name == 'keeper_claim')):
    filtered_events = events_df[(events_df['type_name'] == selected_type_name) & (events_df['result_name'].isin(event_result))]
    mask_complete = filtered_events.result_name.isin(["success"])

    pitch.scatter(filtered_events[mask_complete].start_x, filtered_events[mask_complete].start_y,
                    ax=axs['pitch'], color=colour_success,s=15, label=f"Successful {event_type_correct_name}")
    pitch.scatter(filtered_events[~mask_complete].start_x, filtered_events[~mask_complete].start_y,
                    ax=axs['pitch'], color=colour_fail,s=15, label = f"Unsuccessful {event_type_correct_name}")
        
  elif ((selected_type_name == 'interception') | (selected_type_name == 'clearance') | (selected_type_name == 'tackle') 
        | (selected_type_name == 'keeper_pick_up') | (selected_type_name == 'keeper_save') | (selected_type_name == 'keeper_punch')):
    filtered_events = events_df[(events_df['type_name'] == selected_type_name) & (events_df['result_name'].isin(event_result))]
    mask_complete = filtered_events.result_name.isin(["success"])

    pitch.scatter(filtered_events[mask_complete].start_x, filtered_events[mask_complete].start_y,
                    ax=axs['pitch'], color=colour_success, s=15, label = f"Successful {event_type_correct_name}")
    
  elif ((selected_type_name == 'bad_touch') | (selected_type_name == 'foul')):
    filtered_events = events_df[(events_df['type_name'] == selected_type_name) & (events_df['result_name'].isin(event_result))]
    mask_complete = filtered_events.result_name.isin(["success"])

    pitch.scatter(filtered_events[mask_complete].start_x, filtered_events[mask_complete].start_y,
                    ax=axs['pitch'], color=colour_fail, s=15, label = f"{event_type_correct_name}")

  elif ((selected_type_name == 'Goal')):
    filtered_events = events_df[(events_df['type_name'] == 'shot') & (events_df['result_name']=='success')]

    pitch.lines(filtered_events.start_x, filtered_events.start_y,
                        filtered_events.end_x, filtered_events.end_y,
                        lw=2, transparent=True, comet=True, label=f'Goals',
                        color=colour_success, ax=axs['pitch'])
    
    pitch.scatter(filtered_events.end_x, filtered_events.end_y,
                    ax=axs['pitch'], color=colour_success, s=15)
  
  elif ((selected_type_name == 'Open Play Assist')):
    filtered_events = events_df[(events_df['open_play_assist'] == 1)]

    pitch.lines(filtered_events.start_x, filtered_events.start_y,
                        filtered_events.end_x, filtered_events.end_y,
                        lw=2, transparent=True, comet=True, label=f'Open Play Assist',
                        color=colour_success, ax=axs['pitch'])
    
    pitch.scatter(filtered_events.end_x, filtered_events.end_y,
                    ax=axs['pitch'], color=colour_success, s=15)
    
  elif ((selected_type_name == 'Set-Piece Assist')):
    filtered_events = events_df[(events_df['set_piece_assist'] == 1)]

    pitch.lines(filtered_events.start_x, filtered_events.start_y,
                        filtered_events.end_x, filtered_events.end_y,
                        lw=2, transparent=True, comet=True, label=f'Open Play Assist',
                        color=colour_success, ax=axs['pitch'])
    
    pitch.scatter(filtered_events.end_x, filtered_events.end_y,
                    ax=axs['pitch'], color=colour_success, s=15)
    
  elif ((selected_type_name == 'Most Dangerous Passes')):
    filtered_events = events_df[(events_df['type_name'] == 'pass') & (events_df['result_name'] == 'success')]
    filtered_events=filtered_events.sort_values(by='xT_value', ascending=False).head(10)

    pitch.lines(filtered_events.start_x, filtered_events.start_y,
                        filtered_events.end_x, filtered_events.end_y,
                        lw=2, transparent=True, comet=True, label=f'Most Dangerous Passes (10)',
                        color=colour_success, ax=axs['pitch'])
    
    pitch.scatter(filtered_events.end_x, filtered_events.end_y,
                    ax=axs['pitch'], color=colour_success, s=15)
  
  elif ((selected_type_name == 'Most Dangerous Carries')):
    filtered_events = events_df[(events_df['type_name'] == 'dribble') & (events_df['result_name'] == 'success')]
    filtered_events=filtered_events.sort_values(by='xT_value', ascending=False).head(10)

    pitch.lines(filtered_events.start_x, filtered_events.start_y,
                        filtered_events.end_x, filtered_events.end_y,
                        lw=2, transparent=True, comet=True, label=f'Most Dangerous Carries (10)',
                        color=colour_success, ax=axs['pitch'])
    
    pitch.scatter(filtered_events.end_x, filtered_events.end_y,
                    ax=axs['pitch'], color=colour_success, s=15)
    
  elif ((selected_type_name == 'Attacking Third Passes')):
    filtered_events = events_df[(events_df['type_name'] == 'pass') & (events_df['result_name'] == 'success') 
                                & (events_df['end_x'] >= (105/3)*2)]

    pitch.lines(filtered_events.start_x, filtered_events.start_y,
                        filtered_events.end_x, filtered_events.end_y,
                        lw=2, transparent=True, comet=True, label=f'Attacking Third Passes',
                        color=colour_success, ax=axs['pitch'])
    
    pitch.scatter(filtered_events.end_x, filtered_events.end_y,
                    ax=axs['pitch'], color=colour_success, s=15)
  
  elif ((selected_type_name == 'Attacking Third Carries')):
    filtered_events = events_df[(events_df['type_name'] == 'dribble') & (events_df['result_name'] == 'success') 
                                & (events_df['end_x'] >= (105/3)*2)]

    pitch.lines(filtered_events.start_x, filtered_events.start_y,
                        filtered_events.end_x, filtered_events.end_y,
                        lw=2, transparent=True, comet=True, label=f'Attacking Third Carries',
                        color=colour_success, ax=axs['pitch'])
    
    pitch.scatter(filtered_events.end_x, filtered_events.end_y,
                    ax=axs['pitch'], color=colour_success, s=15)
    
  elif ((selected_type_name == 'Progressive Passes')):
    # Constants for pitch halves
    own_half = 52.5
    opponents_half = 52.5
    
    # Define the conditions for a progressive pass
    conditions = (
        # If the pass starts and ends in the team's own half
        ((events_df['start_x'] <= own_half) & (events_df['end_x'] <= own_half) & ((events_df['end_x'] - events_df['start_x']) >= 30)) |
        # If the pass starts in the team's half and ends in the different half
        ((events_df['start_x'] <= own_half) & (events_df['end_x'] > opponents_half) & ((events_df['end_x'] - events_df['start_x']) >= 15)) |
        # If the pass starts and ends in the opponent's half
        ((events_df['start_x'] > opponents_half) & (events_df['end_x'] > opponents_half) & ((events_df['end_x'] - events_df['start_x']) >= 10))
    )
    
    # Apply the filter for 'pass' type and 'success' result along with the progressive pass conditions
    filtered_events = events_df[(events_df['type_name'] == 'pass') & (events_df['result_name'] == 'success') & conditions]
    
    # Your plotting code remains the same
    pitch.lines(filtered_events.start_x, filtered_events.start_y,
                filtered_events.end_x, filtered_events.end_y,
                lw=2, transparent=True, comet=True, label='Progressive Passes',
                color=colour_success, ax=axs['pitch'])
    
    pitch.scatter(filtered_events.end_x, filtered_events.end_y,
                  ax=axs['pitch'], color=colour_success, s=15)
    
  elif ((selected_type_name == 'Progressive Carries')):
        # Constants for pitch halves
    own_half = 52.5
    opponents_half = 52.5
    
    # Define the conditions for a progressive pass
    conditions = (
        # If the pass starts and ends in the team's own half
        ((events_df['start_x'] <= own_half) & (events_df['end_x'] <= own_half) & ((events_df['end_x'] - events_df['start_x']) >= 30)) |
        # If the pass starts in the team's half and ends in the different half
        ((events_df['start_x'] <= own_half) & (events_df['end_x'] > opponents_half) & ((events_df['end_x'] - events_df['start_x']) >= 15)) |
        # If the pass starts and ends in the opponent's half
        ((events_df['start_x'] > opponents_half) & (events_df['end_x'] > opponents_half) & ((events_df['end_x'] - events_df['start_x']) >= 10))
    )
    
    # Apply the filter for 'pass' type and 'success' result along with the progressive pass conditions
    filtered_events = events_df[(events_df['type_name'] == 'dribble') & (events_df['result_name'] == 'success') & conditions]
    
    # Your plotting code remains the same
    pitch.lines(filtered_events.start_x, filtered_events.start_y,
                filtered_events.end_x, filtered_events.end_y,
                lw=2, transparent=True, comet=True, label='Progressive Passes',
                color=colour_success, ax=axs['pitch'])
    
    pitch.scatter(filtered_events.end_x, filtered_events.end_y,
                  ax=axs['pitch'], color=colour_success, s=15)
    
  #"Goal Creating Actions","Shot Creating Actions", "Progressive Passes","Progressive Carries",Most common pass clusters
    
  # Display the plot in Streamlit
  legend = axs['pitch'].legend(facecolor='#B2BEB5', edgecolor='None', fontsize=7, loc='upper left', handlelength=1)
  for text in legend.get_texts():
    text.set_color('#FFFFFF')

  st.pyplot(fig)

with st.expander('Events Manual'):
    st.write('''
      1. Pass: A normal pass in open play
      2. Dribble: Player carries the ball at least 3 meters
      3. Clearance: Action by a defending player that temporarily removes the attacking threat on their goal/that effectively alleviates pressure on their goal
      4. Interception: Preventing an opponent's pass from reaching their teammates
      5. Bad Touch: When a player mis-controls the ball with a poor touch and loses the ball
      6. Take On: Attempt to dribble past opponent
      7. Foul: Any infringement penalised as foul play by a referee that results in a free-kick or penalty event
      8. Tackle: Tackle on the ball
      9. Cross: A cross into the box
      10. Shot: Shot attempt not from penalty or free-kick
      11. Throw-in: A throw-in from a player
      12. Freekick Short: Short free-kick
      13. Freekick Crossed: Free kick crossed into the box
      14. Shot Freekick: Direct free-kick on goal
      15. Goalkick: Goal kick
      16. Corner Crossed: Corner crossed into the box
      17. Corner Short: Short corner 
      18. Penalty Shot: Penalty shot
      19. Keeper Pick Up: Keeper picks up the ball
      20. Keeper Save: Keeper saves a shot on goal
      21. Keeper Punch: Keeper punches the ball clear
      22. Keeper Claim: Keeper catches a cross
      23. Goal:
      24. Open Play Assist: 
      25. Set-Piece Assist:
      26. Most Dangerous Passes(10):
      27. Most Dangerous Carries(10):
      28. Attacking Third Passes:
      29. Attacking Third Carries:
      30. Progressive Passes:
      31: Progressive Carries:
    ''')

