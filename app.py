import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="NBA BPM Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
    }
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .player-card {
        background-color: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Title with emojis and styling
st.markdown('<h1 class="main-header">🏀 NBA Advanced Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6B7280; font-size: 1.1rem;">Player Impact Analysis with BPM Metrics • Real-time Matchup Predictions</p>', unsafe_allow_html=True)

# Constants
TOTAL_TEAM_MINUTES = 240
IMPACT_MULTIPLIER = 2.083

# Function to load data with progress bar
@st.cache_data(ttl=86400)
def load_nba_data():
    try:
        with st.spinner('🔄 Loading NBA data from Basketball-Reference...'):
            progress_bar = st.progress(0)
            
            url = "https://www.basketball-reference.com/leagues/NBA_2026_advanced.html"
            tables = pd.read_html(url)
            progress_bar.progress(30)
            
            df = tables[0]
            df = df[df['Rk'] != 'Rk'].copy()
            df.columns = df.columns.str.strip()
            progress_bar.progress(50)
            
            # Find BPM column
            bpm_column = None
            for col in df.columns:
                if 'BPM' in col:
                    bpm_column = col
                    break
            
            if bpm_column is None:
                st.error("❌ Could not find BPM column!")
                return None
            
            df = df[['Player', 'Team', 'G', 'MP', bpm_column]].copy()
            df = df.rename(columns={bpm_column: 'BPM'})
            
            df['G'] = pd.to_numeric(df['G'], errors='coerce')
            df['MP'] = pd.to_numeric(df['MP'], errors='coerce')
            df['BPM'] = pd.to_numeric(df['BPM'], errors='coerce')
            df = df.dropna()
            progress_bar.progress(70)
            
            # Calculate actual MPG from data
            df['Actual_MPG'] = df['MP'] / df['G']
            
            # For each team, normalize MPG to percentage of 240 minutes
            normalized_data = []
            
            for team in df['Team'].unique():
                team_df = df[df['Team'] == team].copy()
                total_team_minutes = team_df['Actual_MPG'].sum()
                
                if total_team_minutes > 0:
                    team_df['Minutes_Percent'] = team_df['Actual_MPG'] / total_team_minutes
                    team_df['Normalized_MPG'] = team_df['Minutes_Percent'] * TOTAL_TEAM_MINUTES
                else:
                    team_df['Minutes_Percent'] = 0
                    team_df['Normalized_MPG'] = 0
                
                normalized_data.append(team_df)
            
            # Combine all teams
            df = pd.concat(normalized_data, ignore_index=True)
            
            # Use normalized MPG for impact calculation
            df['MPG'] = df['Normalized_MPG'].round(1)
            df['Impact'] = (df['BPM'] / 100) * df['MPG'] * IMPACT_MULTIPLIER
            df['Impact'] = df['Impact'].round(3)
            
            # Clean up columns
            df = df[['Player', 'Team', 'G', 'Actual_MPG', 'MPG', 'BPM', 'Impact']]
            progress_bar.progress(100)
            
        st.success(f"✅ Successfully loaded data for {len(df['Team'].unique())} teams")
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Function to redistribute minutes when players are removed
def redistribute_minutes_after_removal(data, removed_players, team):
    team_data = data[data['Team'] == team].copy()
    
    if len(team_data) == 0:
        return data
    
    healthy_players = team_data[~team_data['Player'].isin(removed_players)].copy()
    
    if len(healthy_players) == 0 or len(removed_players) == 0:
        return data
    
    removed_minutes = team_data[team_data['Player'].isin(removed_players)]['MPG'].sum()
    
    if removed_minutes == 0:
        for player in removed_players:
            data = data[~((data['Player'] == player) & (data['Team'] == team))]
        return data
    
    total_healthy_mpg = healthy_players['MPG'].sum()
    
    if total_healthy_mpg > 0:
        healthy_percentages = healthy_players['MPG'] / total_healthy_mpg
        new_mpg_values = healthy_percentages * TOTAL_TEAM_MINUTES
        
        for idx, player in healthy_players.iterrows():
            player_name = player['Player']
            new_mpg = new_mpg_values.loc[idx]
            
            data.loc[(data['Player'] == player_name) & (data['Team'] == team), 'MPG'] = new_mpg
            data.loc[(data['Player'] == player_name) & (data['Team'] == team), 'Impact'] = (
                (data.loc[(data['Player'] == player_name) & (data['Team'] == team), 'BPM'].values[0] / 100) 
                * new_mpg 
                * IMPACT_MULTIPLIER
            )
    
    for player in removed_players:
        data = data[~((data['Player'] == player) & (data['Team'] == team))]
    
    return data

# Load data with visual feedback
nba_data = load_nba_data()

if nba_data is None:
    st.stop()

# Sidebar with better styling
with st.sidebar:
    st.markdown('<div class="team-card"><h3>📊 Dashboard Controls</h3></div>', unsafe_allow_html=True)
    
    # Team Selection
    st.markdown('<p style="color: #4B5563; font-weight: 600;">🏀 Select Matchup</p>', unsafe_allow_html=True)
    all_teams = sorted(nba_data['Team'].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Home Team", all_teams, index=0 if 'LAL' in all_teams else 0, key='team1')
    with col2:
        team2_default = 'BOS' if 'BOS' in all_teams else all_teams[1] if len(all_teams) > 1 else all_teams[0]
        team2 = st.selectbox("Away Team", all_teams, index=all_teams.index(team2_default) if team2_default in all_teams else 0, key='team2')
    
    # Injury Management
    st.markdown('<p style="color: #4B5563; font-weight: 600; margin-top: 20px;">🩹 Injury Management</p>', unsafe_allow_html=True)
    
    team1_players = nba_data[nba_data['Team'] == team1]['Player'].tolist()
    team2_players = nba_data[nba_data['Team'] == team2]['Player'].tolist()
    all_matchup_players = team1_players + team2_players
    
    removed_players = st.multiselect(
        "Select injured players to remove:",
        options=all_matchup_players,
        help="Selected players will be removed and their minutes redistributed",
        placeholder="Click to select players..."
    )
    
    # Initialize variables
    team1_removed = []
    team2_removed = []
    
    if removed_players:
        team1_removed = [p for p in removed_players if p in team1_players]
        team2_removed = [p for p in removed_players if p in team2_players]
        
        # Display injury info
        if team1_removed:
            minutes_removed = nba_data[(nba_data['Player'].isin(team1_removed)) & (nba_data['Team'] == team1)]['MPG'].sum()
            st.info(f"**{team1}:** {len(team1_removed)} player(s) injured ({minutes_removed:.1f} MPG)")
        
        if team2_removed:
            minutes_removed = nba_data[(nba_data['Player'].isin(team2_removed)) & (nba_data['Team'] == team2)]['MPG'].sum()
            st.info(f"**{team2}:** {len(team2_removed)} player(s) injured ({minutes_removed:.1f} MPG)")
    
    # Filters
    st.markdown('<p style="color: #4B5563; font-weight: 600; margin-top: 20px;">⚙️ Advanced Filters</p>', unsafe_allow_html=True)
    min_games = st.slider("Minimum games played:", 1, 82, 20, help="Filter players by minimum games played")

# Process data
filtered_data = nba_data[nba_data['G'] >= min_games].copy()
working_data = filtered_data.copy()

if removed_players:
    if team1_removed:
        working_data = redistribute_minutes_after_removal(working_data, team1_removed, team1)
    if team2_removed:
        working_data = redistribute_minutes_after_removal(working_data, team2_removed, team2)

# Get team data
team1_data = working_data[working_data['Team'] == team1].copy()
team2_data = working_data[working_data['Team'] == team2].copy()

# Calculate metrics
team1_impact = team1_data['Impact'].sum()
team2_impact = team2_data['Impact'].sum()
advantage = team1_impact - team2_impact
team1_total_mpg = team1_data['MPG'].sum()
team2_total_mpg = team2_data['MPG'].sum()

# Original data for comparison
original_team1_data = filtered_data[filtered_data['Team'] == team1]
original_team2_data = filtered_data[filtered_data['Team'] == team2]
original_team1_impact = original_team1_data['Impact'].sum()
original_team2_impact = original_team2_data['Impact'].sum()

# MAIN DASHBOARD LAYOUT
# Header Row
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f'<div class="team-card"><h2 style="text-align: center;">{team1} vs {team2}</h2></div>', unsafe_allow_html=True)

# Matchup Metrics
st.markdown('<div class="sub-header">📊 Matchup Analytics</div>', unsafe_allow_html=True)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color: #6B7280;">{team1} Impact</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #1E40AF;">{team1_impact:.2f}</div>
        <div style="font-size: 0.8rem; color: #4B5563;">MPG: {team1_total_mpg:.0f}/240</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color: #6B7280;">{team2} Impact</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #1E40AF;">{team2_impact:.2f}</div>
        <div style="font-size: 0.8rem; color: #4B5563;">MPG: {team2_total_mpg:.0f}/240</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col3:
    impact_change = team1_impact - original_team1_impact
    delta_color = "#10B981" if impact_change >= 0 else "#EF4444"
    delta_symbol = "↗️" if impact_change >= 0 else "↘️"
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color: #6B7280;">{team1} Change</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {delta_color};">{delta_symbol} {impact_change:+.2f}</div>
        <div style="font-size: 0.8rem; color: #4B5563;">After injuries</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col4:
    if abs(advantage) > 3:
        prediction_color = "#10B981" if advantage > 0 else "#EF4444"
        winner = team1 if advantage > 0 else team2
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #6B7280;">Prediction</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {prediction_color};">{winner}</div>
            <div style="font-size: 0.8rem; color: #4B5563;">by {abs(advantage):.1f} points</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #6B7280;">Prediction</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #F59E0B;">Toss-up</div>
            <div style="font-size: 0.8rem; color: #4B5563;">Within 3 points</div>
        </div>
        """, unsafe_allow_html=True)

# Visualizations
st.markdown('<div class="sub-header">📈 Matchup Visualization</div>', unsafe_allow_html=True)

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Create gauge chart for advantage
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = advantage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{team1} Advantage"},
        delta = {'reference': 0},
        gauge = {
            'axis': {'range': [-20, 20]},
            'bar': {'color': "#3B82F6"},
            'steps': [
                {'range': [-20, -10], 'color': "#EF4444"},
                {'range': [-10, -5], 'color': "#F59E0B"},
                {'range': [-5, 5], 'color': "#10B981"},
                {'range': [5, 10], 'color': "#F59E0B"},
                {'range': [10, 20], 'color': "#EF4444"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': advantage
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with viz_col2:
    # Create bar chart comparison
    fig2 = go.Figure(data=[
        go.Bar(name='Full Strength', x=[team1, team2], 
               y=[original_team1_impact, original_team2_impact],
               marker_color=['#3B82F6', '#10B981']),
        go.Bar(name='With Injuries', x=[team1, team2], 
               y=[team1_impact, team2_impact],
               marker_color=['#60A5FA', '#34D399'])
    ])
    fig2.update_layout(
        title="Team Impact Comparison",
        barmode='group',
        height=300,
        showlegend=True
    )
    st.plotly_chart(fig2, use_container_width=True)

# Team Rosters with Tabs
st.markdown('<div class="sub-header">👥 Team Rosters</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs([f"🏠 {team1}", f"✈️ {team2}"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if len(team1_data) > 0:
            # Create player cards
            for _, player in team1_data.sort_values('Impact', ascending=False).head(10).iterrows():
                impact_color = "#10B981" if player['Impact'] > 0 else "#EF4444" if player['Impact'] < 0 else "#6B7280"
                
                # Check if minutes increased
                mpg_change = ""
                if team1_removed:
                    original_mpg = original_team1_data[original_team1_data['Player'] == player['Player']]['MPG'].values
                    if len(original_mpg) > 0:
                        change = player['MPG'] - original_mpg[0]
                        if change > 0.1:
                            mpg_change = f"<span style='color: #10B981; font-size: 0.8rem;'>+{change:.1f} MPG</span>"
                
                st.markdown(f"""
                <div class="player-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{player['Player']}</strong><br>
                            <span style="font-size: 0.85rem; color: #6B7280;">
                                {player['MPG']:.1f} MPG • {player['BPM']:.1f} BPM
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 1.2rem; font-weight: 700; color: {impact_color};">
                                {player['Impact']:.2f}
                            </span><br>
                            {mpg_change}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Team summary
        st.metric("Active Players", len(team1_data))
        if team1_removed:
            st.metric("Injured Players", len(team1_removed))
        
        # Create pie chart for minute distribution
        if len(team1_data) > 0:
            top_players = team1_data.nlargest(5, 'Impact')
            fig_pie = px.pie(top_players, values='MPG', names='Player', 
                           title=f"Top 5 Minute Distribution")
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if len(team2_data) > 0:
            for _, player in team2_data.sort_values('Impact', ascending=False).head(10).iterrows():
                impact_color = "#10B981" if player['Impact'] > 0 else "#EF4444" if player['Impact'] < 0 else "#6B7280"
                
                mpg_change = ""
                if team2_removed:
                    original_mpg = original_team2_data[original_team2_data['Player'] == player['Player']]['MPG'].values
                    if len(original_mpg) > 0:
                        change = player['MPG'] - original_mpg[0]
                        if change > 0.1:
                            mpg_change = f"<span style='color: #10B981; font-size: 0.8rem;'>+{change:.1f} MPG</span>"
                
                st.markdown(f"""
                <div class="player-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{player['Player']}</strong><br>
                            <span style="font-size: 0.85rem; color: #6B7280;">
                                {player['MPG']:.1f} MPG • {player['BPM']:.1f} BPM
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 1.2rem; font-weight: 700; color: {impact_color};">
                                {player['Impact']:.2f}
                            </span><br>
                            {mpg_change}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Active Players", len(team2_data))
        if team2_removed:
            st.metric("Injured Players", len(team2_removed))
        
        if len(team2_data) > 0:
            top_players = team2_data.nlargest(5, 'Impact')
            fig_pie = px.pie(top_players, values='MPG', names='Player', 
                           title=f"Top 5 Minute Distribution")
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

# Injury Impact Section
if removed_players:
    st.markdown('<div class="sub-header">🚫 Injury Impact Analysis</div>', unsafe_allow_html=True)
    
    # Create DataFrame of removed players
    injury_data = []
    for player in removed_players:
        player_data = nba_data[nba_data['Player'] == player]
        if not player_data.empty:
            injury_data.append({
                'Player': player,
                'Team': player_data['Team'].values[0],
                'MPG': player_data['MPG'].values[0],
                'BPM': player_data['BPM'].values[0],
                'Impact': player_data['Impact'].values[0]
            })
    
    if injury_data:
        injury_df = pd.DataFrame(injury_data)
        
        # Display removed players in cards
        cols = st.columns(min(3, len(injury_data)))
        for idx, (_, player) in enumerate(injury_df.iterrows()):
            with cols[idx % len(cols)]:
                st.markdown(f"""
                <div style="background-color: #FEE2E2; padding: 1rem; border-radius: 8px; border-left: 4px solid #EF4444;">
                    <div style="color: #B91C1C; font-weight: 600;">{player['Player']}</div>
                    <div style="font-size: 0.9rem; color: #7F1D1D;">
                        {player['Team']} • {player['MPG']:.1f} MPG
                    </div>
                    <div style="font-size: 0.85rem; color: #991B1B; margin-top: 0.5rem;">
                        Impact: {player['Impact']:.2f} • BPM: {player['BPM']:.1f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**📊 Data Source**  \nBasketball-Reference.com")

with footer_col2:
    st.markdown("**📈 Impact Formula**  \n(BPM ÷ 100) × MPG × 2.083")

with footer_col3:
    st.markdown("**⏱️ Minute Logic**  \nNormalized to 240 total team minutes")

# Status indicator
if removed_players:
    st.success(f"✅ Analysis complete: {len(removed_players)} player(s) removed")
else:
    st.info("ℹ️ No injuries selected - showing full strength analysis")
