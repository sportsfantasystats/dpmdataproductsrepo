import streamlit as st
import pandas as pd
import kagglehub
import os
import altair as alt
import plotly.express as px
from streamlit_option_menu import option_menu


st.set_page_config(page_title="F1 Data Explorer", layout="wide")
st.title("Formula 1 Data Explorer")

# Set Kaggle credentials from Streamlit secrets
os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]


# Team Colors
# F1 Team Colors (2024 season)
TEAM_COLORS = {
    'Red Bull': '#3671C6',
    'Ferrari': '#E8002D',
    'Mercedes': '#27F4D2',
    'McLaren': '#FF8000',
    'Aston Martin': '#229971',
    'Alpine': '#FF87BC',
    'Alpine F1 Team': '#FF87BC',
    'Williams': '#64C4FF',
    'AlphaTauri': '#5E8FAA',
    'Alfa Romeo': '#C92D4B',
    'Haas': '#B6BABD',
    'Racing Point': '#F596C8',
    'Renault': '#FFF500',
    'Toro Rosso': '#469BFF',
    'Sauber': '#9B0000',
    'Force India': '#FF80C7',
    'Manor': '#6D1E3C',
    'Lotus': '#FFB800',
    'Caterham': '#0B361F',
    'Marussia': '#6D1E3C',
    'HRT': '#C92D4B'
}


@st.cache_data(ttl=3600)
def load_data():
    path = kagglehub.dataset_download("jtrotman/formula-1-race-data")
    
    drivers = pd.read_csv(f'{path}/drivers.csv')
    races = pd.read_csv(f'{path}/races.csv')
    results = pd.read_csv(f'{path}/results.csv')
    constructors = pd.read_csv(f'{path}/constructors.csv')
    lap_times = pd.read_csv(f'{path}/lap_times.csv')
    
    return drivers, races, results, constructors, lap_times

@st.cache_data
def process_data(drivers, races, results, constructors, lap_times):
    # Process races
    races_df = races[['raceId', 'year', 'round', 'circuitId', 'name', 'date', 'time']].copy()
    races_df.columns = ['raceId', 'race_year', 'round', 'circuitId', 'race_name', 'race_date', 'race_time']
    races_df['race_date'] = pd.to_datetime(races_df['race_date'])

    # Only picking seasons after the new points system
    races_df = races_df[races_df['race_year'] >= 2010]
    results = results[results['raceId'].isin(races_df['raceId'])]
    lap_times = lap_times[lap_times['raceId'].isin(races_df['raceId'])]


    # Process drivers
    drivers_df = drivers[['driverId', 'code', 'forename', 'surname', 'dob', 'nationality']].copy()
    drivers_df['driver_name'] = drivers_df['forename'] + ' ' + drivers_df['surname']
    drivers_df = drivers_df[['driverId', 'code', 'driver_name', 'dob', 'nationality']]
    
    # Process constructors
    constructors_df = constructors[['constructorId', 'name']].copy()
    constructors_df.columns = ['constructorId', 'constructor_name']

  
    ## DRIVE SCORE CALCULATION
    points_dict = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 
                   10:1, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 0:0, 25:0}

    def score_func(grid, pos, posText):
        dscore = 0
        if pd.notna(grid):
            # Robustness check: validate grid and pos are in points_dict
            if grid not in points_dict or pos not in points_dict:
                return 0
            if grid == 0 or posText in ['R','D','W']:
                pass
            elif grid == pos:
                dscore = dscore + points_dict[grid] + points_dict[pos]
            elif grid < pos:
                neg = 0
                for i in list(range(grid+1,pos)):
                    neg = neg + points_dict[i]
                dscore = points_dict[grid] + points_dict[pos] - neg
            elif grid > pos:
                bonus = 0
                for i in list(range(pos+1,grid)):
                    bonus = bonus + points_dict[i]
                dscore = points_dict[grid] + points_dict[pos] + bonus
        else:
            pass
        return dscore

    # Merge race data with results, drivers, constructors
    results_with_grid = results[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder', 'positionText', 'points', 
                                 'fastestLapTime', 'time', 'milliseconds']].copy()
    race_data = pd.merge(races_df, results_with_grid, on='raceId', how='left')
    race_data = pd.merge(race_data, drivers_df, on='driverId', how='left')
    race_data = pd.merge(race_data, constructors_df, on='constructorId', how='left')


    ### **DATA CORRECTIONS **
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Oscar Piastri'), 'grid'] = '1'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Lando Norris'), 'grid'] = '2'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Max Verstappen'), 'grid'] = '3'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'George Russell'), 'grid'] = '4'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Andrea Kimi Antonelli'), 'grid'] = '5'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Isack Hadjar'), 'grid'] = '6'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Carlos Sainz'), 'grid'] = '7'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Fernando Alonso'), 'grid'] = '8'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Pierre Gasly'), 'grid'] = '9'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Charles Leclerc'), 'grid'] = '10'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Nico Hülkenberg'), 'grid'] = '11'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Liam Lawson'), 'grid'] = '12'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Oliver Bearman'), 'grid'] = '13'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Alexander Albon'), 'grid'] = '14'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Yuki Tsunoda'), 'grid'] = '15'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Esteban Ocon'), 'grid'] = '16'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Lewis Hamilton'), 'grid'] = '17'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Lance Stroll'), 'grid'] = '18'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Gabriel Bortoleto'), 'grid'] = '19'
    race_data.loc[(race_data['race_date'] == '2025-11-30') & (race_data['driver_name'] == 'Franco Colapinto'), 'grid'] = '20'


    # Calculate driver score
    race_data['grid'] = race_data['grid'].astype('Int64')
    race_data['positionOrder'] = race_data['positionOrder'].astype('Int64')
    race_data['driver_score'] = race_data[['grid','positionOrder','positionText']].apply(
        lambda row: score_func(row['grid'], row['positionOrder'], row['positionText']), axis=1
    )
    
    # Lap level data
    lap_times.columns = ['raceId', 'driverId', 'l_lap', 'l_position', 'l_time', 'l_milliseconds']
    position_df = results[['raceId','driverId','positionText']]

    ldf_1 = pd.merge(races_df, lap_times, on='raceId', how='left')
    ldf_2 = pd.merge(ldf_1, position_df, on=['raceId', 'driverId'], how='left')
    lap_data = pd.merge(ldf_2, drivers_df, on='driverId', how='left')
    
    return races_df, drivers_df, race_data, lap_data

# Load data
with st.spinner('Loading F1 data...'):
    drivers, races, results, constructors, lap_times = load_data()
    races_df, drivers_df, race_data, lap_data = process_data(drivers, races, results, constructors, lap_times)

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Select View",
        options=["Circuit Insights", "Driver Insights", "Export Raw Data"],
        icons=["geo-alt", "person", "download"],
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "nav-link-selected": {"font-size": "14px"},
            "nav-link": {"font-size": "14px"}
        }
    )

# Season filter
all_seasons = sorted(race_data['race_year'].dropna().unique(), reverse=True)
selected_seasons = st.sidebar.multiselect("Filter Seasons", all_seasons, default=all_seasons[:5])

# Filter data
if selected_seasons:
    race_filtered = race_data[race_data['race_year'].isin(selected_seasons)]
    lap_filtered = lap_data[lap_data['race_year'].isin(selected_seasons)]
else:
    race_filtered = race_data
    lap_filtered = lap_data

# Sidebar metrics (filtered data stats)
st.sidebar.markdown("<h4 style='font-size: 16px;'>Selected Data Stats</h4>", unsafe_allow_html=True)
st.sidebar.metric("Seasons", race_filtered['race_year'].nunique() if not race_filtered.empty else 0)
st.sidebar.metric("Races", len(race_filtered) if not race_filtered.empty else 0)
st.sidebar.metric("Drivers", race_filtered['driver_name'].nunique() if not race_filtered.empty else 0)

# Export Raw Data Tab
if selected == "Export Raw Data":
    st.subheader("Export Raw Data")
    
    # Race Level Data section
    st.markdown("### Race-Level Data")
    st.markdown("Complete race results including grid positions, final positions, points, and driver scores.")
    
    export_data = race_filtered[['race_year','race_name','driver_name','constructor_name','grid',
                                 'positionOrder','positionText','points','fastestLapTime','milliseconds','driver_score']].rename(columns={
        'race_year':'Season',
        'race_name':'Race Name',
        'driver_name':'Driver Name',
        'constructor_name':'Constructor Name',
        'grid':'Grid Position',
        'positionOrder':'Final Position',
        'positionText':'Position Text',
        'points':'Points',
        'fastestLapTime':'Fastest Lap Time',
        'time':'Race Time',
        'milliseconds':'Race Time (ms)',
        'driver_score':'Driver Score'
    })

    st.dataframe(export_data, width='stretch', height=400, hide_index=True)
    st.download_button("Download Race-Level CSV", export_data.to_csv(index=False), "race_level_data.csv", "text/csv")
    
    st.markdown("---")
    
    # Lap Level Data section
    st.markdown("### Lap-Time Data")
    st.markdown("Detailed lap-by-lap timing data for all drivers across all races.")
    
    lap_export = lap_filtered[['race_year','race_name','race_date','round','driver_name','l_lap','l_time','l_milliseconds']].rename(columns={
        'race_year':'Season',
        'race name':'Race Name',
        'race_date':'Race Date',
        'round':'Round',
        'driver_name':'Driver Name',
        'l_lap':'Lap Number',
        'l_time':'Lap Time',
        'l_milliseconds':'Lap Time (ms)'
    })
    lap_export['Race Date'] = pd.to_datetime(lap_export['Race Date'])

    st.dataframe(lap_export, width='stretch', height=400, hide_index=True)
    st.download_button("Download Lap-Time CSV", lap_export.to_csv(index=False), "lap_level_data.csv", "text/csv")


# Circuit Insights Tab
elif selected == "Circuit Insights":
    st.subheader("Circuit Insights")
    
    available_circuits = race_filtered['race_name'].sort_values().unique()
    default_circuit_idx = 0  # Default to first circuit
    circuit = st.selectbox("Select Circuit", available_circuits, index=default_circuit_idx)
    circuit_data = race_filtered[race_filtered['race_name'] == circuit]
    
    if not circuit_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Drivers on This Circuit**")
            
            # Calculate win percentage (podiums/total races) and avg drive score
            driver_stats = circuit_data.groupby('driver_name').agg(
                total_races=('raceId', 'count'),
                podiums=('positionOrder', lambda x: (x <= 3).sum()),
                avg_position=('positionOrder', 'mean'),
                avg_driver_score=('driver_score', 'mean')
            ).reset_index()
            driver_stats['win_pct'] = (driver_stats['podiums'] / driver_stats['total_races'] * 100).round(1)
            driver_stats = driver_stats.sort_values('win_pct', ascending=False).head(10)
            
            # Create display dataframe
            display_stats = driver_stats[['driver_name', 'total_races', 'podiums', 'win_pct', 'avg_position', 'avg_driver_score']].copy()
            display_stats['avg_position'] = display_stats['avg_position'].round(1)
            display_stats['avg_driver_score'] = display_stats['avg_driver_score'].round(1)
            display_stats.columns = ['Driver', 'Races', 'Podiums', 'Win %', 'Avg Position', 'Avg Drive Score']
            
            st.dataframe(display_stats, width='stretch', hide_index=True)
        
        with col2:
            st.markdown("**Podium Distribution**")
            st.markdown("<p style='font-size: 11px; color: gray;'>Click the download button below the chart to export data as CSV</p>", unsafe_allow_html=True)
            if not driver_stats.empty:
                # Get constructor color for each driver
                driver_teams = circuit_data.groupby('driver_name')['constructor_name'].agg(
                    lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
                ).reset_index()
                driver_stats_colored = driver_stats.merge(driver_teams, on='driver_name', how='left')
                driver_stats_colored['color'] = driver_stats_colored['constructor_name'].map(TEAM_COLORS).fillna('#808080')
                
                # Altair chart
                chart = alt.Chart(driver_stats_colored.head(10)).mark_bar().encode(
                    x=alt.X('podiums:Q', axis=alt.Axis(title='Podiums')),
                    y=alt.Y('driver_name:N', sort='-x', axis=alt.Axis(title=None)),
                    color=alt.Color('color:N', scale=None, legend=None),
                    tooltip=[
                        alt.Tooltip('driver_name:N', title='Driver'),
                        alt.Tooltip('constructor_name:N', title='Team'),
                        alt.Tooltip('podiums:Q', title='Podiums')
                    ]
                ).properties(height=400)
                
                text = chart.mark_text(align='left', dx=3, color='white').encode(
                    text=alt.Text('podiums:Q', format='.0f')
                )
                
                st.altair_chart(chart + text, width='stretch')
                
                # Download button for chart data
                chart_data_csv = driver_stats_colored[['driver_name', 'constructor_name', 'podiums']].to_csv(index=False)
                st.download_button("Download Chart Data", chart_data_csv, "podium_distribution.csv", "text/csv", key="podium_dl")
        
        # Constructor performance over time
        st.markdown("**Constructor Performance Over Time**")
        st.markdown("<p style='font-size: 11px; color: gray;'>Click the download button below the chart to export data as CSV</p>", unsafe_allow_html=True)
        
        constructors_array = circuit_data['constructor_name'].sort_values().unique()
        default_constructor_idx = list(constructors_array).index('Red Bull') if 'Red Bull' in constructors_array else 0
        constructor = st.selectbox("Select Constructor", constructors_array, index=default_constructor_idx)
        
        constructor_circuit = circuit_data[circuit_data['constructor_name'] == constructor]
        if not constructor_circuit.empty:
            yearly_points = constructor_circuit.groupby('race_year')['points'].sum().reset_index()
            constructor_color = TEAM_COLORS.get(constructor, '#808080')
            
            # Altair chart
            chart = alt.Chart(yearly_points).mark_line(point=True, color=constructor_color).encode(
                x=alt.X('race_year:O', axis=alt.Axis(title='Year')),
                y=alt.Y('points:Q', axis=alt.Axis(title='Total Points')),
                tooltip=[
                    alt.Tooltip('race_year:O', title='Year'),
                    alt.Tooltip('points:Q', title='Points')
                ]
            ).properties(height=400)
            
            st.altair_chart(chart, width='stretch')
            
            # Download button for chart data
            chart_data_csv = yearly_points.to_csv(index=False)
            st.download_button("Download Chart Data", chart_data_csv, f"{constructor}_performance.csv", "text/csv", key="constructor_perf_dl")
    else:
        st.info("No data available for this circuit in selected seasons.")


# Driver Insights Tab
elif selected == "Driver Insights":
    st.subheader("Driver Insights")
    
    available_drivers = race_filtered['driver_name'].sort_values().unique()
    default_driver_idx = list(available_drivers).index('Max Verstappen') if 'Max Verstappen' in available_drivers else 0
    driver = st.selectbox("Select Driver", available_drivers, index=default_driver_idx)
    driver_data = race_filtered[race_filtered['driver_name'] == driver]
    
    if not driver_data.empty:
        # Get driver's primary constructor color
        driver_constructor = driver_data.groupby('constructor_name').size().idxmax()
        driver_color = TEAM_COLORS.get(driver_constructor, '#808080')
        
        # Points per season
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Performance Per Season**")
            st.markdown("<p style='font-size: 11px; color: gray;'>Click the download button below the chart to export data as CSV</p>", unsafe_allow_html=True)
            
            # Radio button to switch between Points and Driver Score
            metric_choice = st.radio("Select Metric:", ["Total Points", "Total Drive Score"], horizontal=True, key="perf_metric")
            
            if metric_choice == "Total Points":
                perf_data = driver_data.groupby('race_year')['points'].sum().reset_index()
                perf_data.columns = ['race_year', 'value']
                y_label = 'Points'
                file_suffix = 'points'
            else:
                perf_data = driver_data.groupby('race_year')['driver_score'].sum().reset_index()
                perf_data.columns = ['race_year', 'value']
                y_label = 'Drive Score'
                file_suffix = 'drive_score'
            
            # Altair chart
            chart = alt.Chart(perf_data).mark_bar(color=driver_color).encode(
                x=alt.X('race_year:O', axis=alt.Axis(title='Season')),
                y=alt.Y('value:Q', axis=alt.Axis(title=y_label)),
                tooltip=[
                    alt.Tooltip('race_year:O', title='Season'),
                    alt.Tooltip('value:Q', title=y_label, format='.1f')
                ]
            ).properties(height=350)
            
            text = chart.mark_text(dy=-5, color='black').encode(
                text=alt.Text('value:Q', format='.0f')
            )
            
            st.altair_chart(chart + text, width='stretch')
            
            # Download button for chart data
            chart_data_csv = perf_data.to_csv(index=False)
            st.download_button("Download Chart Data", chart_data_csv, f"{driver}_{file_suffix}_per_season.csv", "text/csv", key="perf_season_dl")
        
        with col2:
            st.markdown("**Wins Per Season**")
            st.markdown("<p style='font-size: 11px; color: gray;'>Click the download button below the chart to export data as CSV</p>", unsafe_allow_html=True)
            wins_per_season = driver_data[driver_data['positionOrder'] == 1].groupby('race_year').size().reset_index(name='wins')
            
            # Altair chart
            chart = alt.Chart(wins_per_season).mark_bar(color=driver_color).encode(
                x=alt.X('race_year:O', axis=alt.Axis(title='Season')),
                y=alt.Y('wins:Q', axis=alt.Axis(title='Wins')),
                tooltip=[
                    alt.Tooltip('race_year:O', title='Season'),
                    alt.Tooltip('wins:Q', title='Wins')
                ]
            ).properties(height=350)
            
            text = chart.mark_text(dy=-5, color='black').encode(
                text=alt.Text('wins:Q', format='.0f')
            )
            
            st.altair_chart(chart + text, width='stretch')
            
            # Download button for chart data
            chart_data_csv = wins_per_season.to_csv(index=False)
            st.download_button("Download Chart Data", chart_data_csv, f"{driver}_wins_per_season.csv", "text/csv", key="wins_season_dl")
        
        # Average finishing position per season
        st.markdown("**Average Finishing Position Per Season**")
        st.markdown("<p style='font-size: 11px; color: gray;'>Click the download button below the chart to export data as CSV</p>", unsafe_allow_html=True)
        avg_position = driver_data.groupby('race_year')['positionOrder'].mean().reset_index()
        avg_position['positionOrder'] = avg_position['positionOrder'].round(1)
        
        # Altair chart (reversed y-axis for position)
        chart = alt.Chart(avg_position).mark_line(point=True, color=driver_color).encode(
            x=alt.X('race_year:O', axis=alt.Axis(title='Season')),
            y=alt.Y('positionOrder:Q', axis=alt.Axis(title='Avg Position'), scale=alt.Scale(reverse=True)),
            tooltip=[
                alt.Tooltip('race_year:O', title='Season'),
                alt.Tooltip('positionOrder:Q', title='Avg Position', format='.1f')
            ]
        ).properties(height=350)
        
        st.altair_chart(chart, width='stretch')
        
        # Download button for chart data
        chart_data_csv = avg_position.to_csv(index=False)
        st.download_button("Download Chart Data", chart_data_csv, f"{driver}_avg_position.csv", "text/csv", key="avg_pos_dl")
        
        # Performance by circuit
        st.markdown("**Performance by Circuit (Top 10)**")
        circuit_performance = driver_data.groupby('race_name').agg(
            races=('raceId', 'count'),
            avg_position=('positionOrder', 'mean'),
            total_points=('points', 'sum')
        ).reset_index()
        circuit_performance['avg_position'] = circuit_performance['avg_position'].round(1)
        circuit_performance = circuit_performance.sort_values('total_points', ascending=False).head(10)
        
        display_perf = circuit_performance[['race_name', 'races', 'avg_position', 'total_points']].copy()
        display_perf.columns = ['Circuit', 'Races', 'Avg Position', 'Total Points']
        
        st.dataframe(display_perf, width='stretch', hide_index=True)
    else:
        st.info("No data available for this driver in selected seasons.")
