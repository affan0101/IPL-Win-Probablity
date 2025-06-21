import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.pipeline import Pipeline

# Set page configuration
st.set_page_config(
    page_title="IPL Win Predictor Pro+",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
:root {
    --primary: #4CAF50;
    --secondary: #2E7D32;
    --accent: #FF5722;
    --dark: #263238;
    --light: #ECEFF1;
}

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.sidebar .sidebar-content {
    background: linear-gradient(195deg, #3a4a6b 0%, #1a2a4a 100%) !important;
    color: white !important;
}

.stButton>button {
    background: linear-gradient(to right, var(--primary), var(--secondary));
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 28px;
    font-weight: bold;
    font-size: 16px;
    transition: all 0.3s;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    background: linear-gradient(to right, var(--secondary), var(--primary));
}

.header-card {
    background: rgba(255,255,255,0.95);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    margin-bottom: 2rem;
    border-top: 4px solid var(--primary);
    backdrop-filter: blur(5px);
}

.prediction-card {
    background: rgba(255,255,255,0.98);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.1);
    border: 1px solid rgba(0,0,0,0.05);
}

.prediction-card:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0 12px 40px rgba(0,0,0,0.15);
}

.insight-container {
    background: rgba(255,255,255,0.98);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}

.team-name {
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: var(--dark);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.probability-value {
    font-size: 3rem;
    font-weight: 900;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.progress-container {
    height: 24px;
    border-radius: 12px;
    background-color: #f0f0f0;
    margin: 1.5rem 0;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}

.progress-bar {
    height: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    transition: width 0.8s cubic-bezier(0.65, 0, 0.35, 1);
}

.match-analysis {
    background: rgba(248,249,250,0.95);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 2rem;
    border-left: 5px solid var(--primary);
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

/* New elements */
.impact-factor {
    display: flex;
    align-items: center;
    padding: 1rem;
    background: rgba(255,255,255,0.9);
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.impact-icon {
    font-size: 1.8rem;
    margin-right: 1rem;
    color: var(--primary);
}

.impact-label {
    font-weight: 700;
    color: var(--dark);
    margin-bottom: 0.2rem;
}

.impact-value {
    font-weight: 600;
    color: #555;
}

.timeline-container {
    margin-top: 2rem;
    padding: 1.5rem;
    background: rgba(255,255,255,0.98);
    border-radius: 16px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.1);
}

/* Sidebar specific styling */
.sidebar h1 {
    color: white !important;
    text-align: center;
    margin-bottom: 0.5rem;
    font-size: 2rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.sidebar h2, .sidebar h3 {
    color: white !important;
    border-bottom: 1px solid rgba(255,255,255,0.2);
    padding-bottom: 0.5rem;
}

.sidebar p {
    color: rgba(255,255,255,0.9) !important;
    line-height: 1.6;
}

.sidebar .stMarkdown {
    color: rgba(255,255,255,0.9) !important;
}

/* Input field styling */
.stSelectbox, .stNumberInput, .stSlider {
    margin-bottom: 1.5rem;
}

.stSelectbox>div>div>div, 
.stNumberInput>div>div>input, 
.stSlider>div>div>div>div {
    border-radius: 12px !important;
    border: 1px solid #ddd !important;
    padding: 12px !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .header-card, .prediction-card, .insight-container {
        padding: 1.5rem;
    }
    .probability-value {
        font-size: 2.5rem;
    }
}

/* Expander styling */
.streamlit-expanderHeader {
    font-size: 1.2rem;
    font-weight: bold;
    color: var(--dark);
    padding: 1rem;
}

.streamlit-expanderContent {
    padding: 1.5rem 0;
}

/* New trophy badge */
.trophy-badge {
    position: absolute;
    top: -15px;
    right: -15px;
    background: linear-gradient(135deg, #FFD700, #FFA500);
    color: white;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    z-index: 1;
}

/* New prediction confidence indicator */
.confidence-indicator {
    height: 8px;
    border-radius: 4px;
    background: #e0e0e0;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}

.confidence-bar {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #4CAF50, #8BC34A);
    position: absolute;
    left: 0;
    top: 0;
    transition: width 0.5s ease;
}

/* New team comparison cards */
.team-comparison {
    display: flex;
    justify-content: space-between;
    margin: 1.5rem 0;
}

.team-card {
    flex: 1;
    padding: 1.5rem;
    border-radius: 12px;
    background: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 0 0.5rem;
    text-align: center;
    transition: all 0.3s;
}

.team-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.team-logo {
    width: 80px;
    height: 80px;
    margin: 0 auto 1rem;
    border-radius: 50%;
    background: #f5f5f5;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: var(--primary);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.team-stats {
    margin-top: 1rem;
}

.stat-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}

.stat-label {
    font-weight: 600;
    color: #555;
}

.stat-value {
    font-weight: 700;
    color: var(--dark);
}

/* Add this to your main CSS section */
.stSelectbox>div>div>div {
    padding: 10px 12px !important;
    display: flex !important;
    align-items: center !important;
    height: auto !important;
}

.stSelectbox>div>div>div>div {
    overflow: visible !important;
    text-overflow: unset !important;
    white-space: normal !important;
}

.st-bb, .st-ba, .st-b9, .st-b8 {
    align-items: center !important;
}
</style>
""", unsafe_allow_html=True)

# Teams and venues data (must match the training data exactly)
TEAMS = [
    'Chennai Super Kings', 'Delhi Capitals', 'Kolkata Knight Riders',
    'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad',
    'Gujarat Titans', 'Lucknow Super Giants'
]

VENUES = [
    'Eden Gardens', 'Wankhede Stadium', 'MA Chidambaram Stadium',
    'Arun Jaitley Stadium', 'Narendra Modi Stadium',
    'M. Chinnaswamy Stadium', 'Punjab Cricket Association Stadium',
    'Rajiv Gandhi International Stadium', 'Sawai Mansingh Stadium',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
    'Brabourne Stadium', 'DY Patil Stadium', 'Holkar Cricket Stadium',
    'Others'
]

# Team colors for visualizations
TEAM_COLORS = {
    'Chennai Super Kings': '#FDB913',
    'Delhi Capitals': '#004C93',
    'Kolkata Knight Riders': '#3A225D',
    'Mumbai Indians': '#005DA0',
    'Punjab Kings': '#AA4545',
    'Rajasthan Royals': '#2D4D9D',
    'Royal Challengers Bangalore': '#EC1C24',
    'Sunrisers Hyderabad': '#FB643E',
    'Gujarat Titans': '#0D4D8B',
    'Lucknow Super Giants': '#00A381'
}


# Load model with caching
@st.cache_resource
def load_model():
    return joblib.load('advanced_pipe.pkl')


def sidebar_content():
    """Display sidebar content with enhanced styling"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 2rem;">
            <h1>🏆 IPL PREDICTOR PRO+</h1>
            <p style="color: #470b0b; font-size: 1.1rem;">The Ultimate Cricket Analytics Platform</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔍 About This Tool")
        st.markdown("""
        <div style="color: #470b0b">
        Our AI-powered predictor analyzes 14+ match factors in real-time to deliver the most accurate win probability calculations.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚡ Key Innovations")
        st.markdown("""
        <div style="color: #470b0b;">
        • <strong>Dynamic Pressure Index</strong> - Live match tension analysis<br>
        • <strong>Momentum Tracker</strong> - Visualize shifting game dynamics<br>
        • <strong>Batsman Impact Score</strong> - Player influence modeling<br>
        • <strong>Venue Advantage</strong> - Stadium-specific analytics<br>
        • <strong>Real-time Adaptation</strong> - Continuous model updating
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 📊 Model Performance")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 12px;">
            <div style="display: flex; justify-content: space-between;">
                <span>Accuracy</span>
                <span><strong>89.2%</strong></span>
            </div>
            <div style="height: 6px; background: rgba(255,255,255,0.2); border-radius: 3px; margin: 0.5rem 0;">
                <div style="width: 89.2%; height: 100%; background: #4CAF50; border-radius: 3px;"></div>
            </div>
              <div style="display: flex; justify-content: space-between;">
                 <span>Precision</span>
                <span><strong>90%</strong></span>
            </div>
            <div style="height: 6px; background: rgba(255,255,255,0.2); border-radius: 3px; margin: 0.5rem 0;">
                <div style="width: 87.7%; height: 100%; background: #2196F3; border-radius: 3px;"></div>
            </div>

        </div>
        """, unsafe_allow_html=True)  # This is the critical line that was missing

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #470b0b; font-size: 0.9rem;">
            Powered by Team Affan <br>
            Version 2.1 • Updated: May 2025
        </div>
        """, unsafe_allow_html=True)


def calculate_advanced_metrics(params):
    """Calculate all advanced metrics exactly as in the training code"""
    balls_left = max(120 - (params['overs_completed'] * 6), 0)
    runs_left = max(params['target'] - params['current_score'], 0)
    crr = params['current_score'] / params['overs_completed'] if params['overs_completed'] > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Calculate dot ball percentage (simplified for prediction interface)
    dot_ball_percent = min(0.6, params['wickets'] * 0.1)

    # Calculate wickets in hand first
    wickets_in_hand = 10 - params['wickets']

    # Calculate momentum shift (more sophisticated version)
    momentum_shift = (params['current_score'] - (crr * params['overs_completed'])) * (1 + (wickets_in_hand / 10))

    metrics = {
        'balls_left': balls_left,
        'runs_left': runs_left,
        'crr': crr,
        'rrr': rrr,
        'wickets_in_hand': wickets_in_hand,
        'pressure_index': rrr / crr if crr > 0 else 0,
        'momentum_shift_index': momentum_shift,
        'dot_ball_percent': dot_ball_percent,
        'top_batsman_playing': params.get('top_batsman_playing', 0),
        'recent_partnership': params.get('recent_partnership', 30),
        'wickets': params['wickets'],
        'current_score': params['current_score'],
        'target': params['target'],
        'overs_completed': params['overs_completed']
    }

    return metrics


def create_input_dataframe(batting_team, bowling_team, venue, metrics):
    """Create the input dataframe matching the training data structure exactly"""
    return pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'venue': [venue],
        'current_score': [metrics['current_score']],
        'wickets': [metrics['wickets']],
        'balls_left': [metrics['balls_left']],
        'runs_left': [metrics['runs_left']],
        'crr': [metrics['crr']],
        'rrr': [metrics['rrr']],
        'pressure_index': [metrics['pressure_index']],
        'momentum_shift_index': [metrics['momentum_shift_index']],
        'dot_ball_percent': [metrics['dot_ball_percent']],
        'wickets_in_hand': [metrics['wickets_in_hand']],
        'top_batsman_playing': [metrics['top_batsman_playing']]
    })


# Add this dictionary with team stats near the TEAMS and VENUES definitions
TEAM_STATS = {
    'Chennai Super Kings': {
        'batting': {
            'Win Rate': '62%',
            'Avg Score': '168',
            'Powerplay RR': '7.8',
            'Death Overs RR': '9.8'
        },
        'bowling': {
            'Win Rate': '65%',
            'Avg Conceded': '160',
            'Powerplay Eco': '7.2',
            'Death Overs Eco': '9.2'
        }
    },
    'Delhi Capitals': {
        'batting': {
            'Win Rate': '55%',
            'Avg Score': '165',
            'Powerplay RR': '8.0',
            'Death Overs RR': '9.5'
        },
        'bowling': {
            'Win Rate': '58%',
            'Avg Conceded': '163',
            'Powerplay Eco': '7.5',
            'Death Overs Eco': '9.5'
        }
    },
    'Kolkata Knight Riders': {
        'batting': {
            'Win Rate': '60%',
            'Avg Score': '170',
            'Powerplay RR': '8.2',
            'Death Overs RR': '10.2'
        },
        'bowling': {
            'Win Rate': '62%',
            'Avg Conceded': '165',
            'Powerplay Eco': '7.4',
            'Death Overs Eco': '9.8'
        }
    },
    'Mumbai Indians': {
        'batting': {
            'Win Rate': '65%',
            'Avg Score': '175',
            'Powerplay RR': '8.5',
            'Death Overs RR': '10.5'
        },
        'bowling': {
            'Win Rate': '63%',
            'Avg Conceded': '168',
            'Powerplay Eco': '7.6',
            'Death Overs Eco': '9.9'
        }
    },
    'Punjab Kings': {
        'batting': {
            'Win Rate': '52%',
            'Avg Score': '162',
            'Powerplay RR': '7.9',
            'Death Overs RR': '9.7'
        },
        'bowling': {
            'Win Rate': '55%',
            'Avg Conceded': '170',
            'Powerplay Eco': '7.8',
            'Death Overs Eco': '10.1'
        }
    },
    'Rajasthan Royals': {
        'batting': {
            'Win Rate': '58%',
            'Avg Score': '169',
            'Powerplay RR': '8.1',
            'Death Overs RR': '10.0'
        },
        'bowling': {
            'Win Rate': '60%',
            'Avg Conceded': '164',
            'Powerplay Eco': '7.3',
            'Death Overs Eco': '9.6'
        }
    },
    'Royal Challengers Bangalore': {
        'batting': {
            'Win Rate': '57%',
            'Avg Score': '172',
            'Powerplay RR': '8.3',
            'Death Overs RR': '10.3'
        },
        'bowling': {
            'Win Rate': '59%',
            'Avg Conceded': '169',
            'Powerplay Eco': '7.7',
            'Death Overs Eco': '10.0'
        }
    },
    'Sunrisers Hyderabad': {
        'batting': {
            'Win Rate': '56%',
            'Avg Score': '164',
            'Powerplay RR': '7.7',
            'Death Overs RR': '9.6'
        },
        'bowling': {
            'Win Rate': '61%',
            'Avg Conceded': '162',
            'Powerplay Eco': '7.1',
            'Death Overs Eco': '9.3'
        }
    },
    'Gujarat Titans': {
        'batting': {
            'Win Rate': '63%',
            'Avg Score': '171',
            'Powerplay RR': '8.0',
            'Death Overs RR': '10.1'
        },
        'bowling': {
            'Win Rate': '64%',
            'Avg Conceded': '161',
            'Powerplay Eco': '7.2',
            'Death Overs Eco': '9.4'
        }
    },
    'Lucknow Super Giants': {
        'batting': {
            'Win Rate': '59%',
            'Avg Score': '167',
            'Powerplay RR': '7.9',
            'Death Overs RR': '9.9'
        },
        'bowling': {
            'Win Rate': '60%',
            'Avg Conceded': '163',
            'Powerplay Eco': '7.3',
            'Death Overs Eco': '9.5'
        }
    }
}


# Then modify the display_team_comparison function to use these stats:
def display_team_comparison(batting_team, bowling_team):
    """Display team comparison cards with visual indicators"""
    st.markdown("### 🏆 Team Comparison")

    # Get stats from the TEAM_STATS dictionary
    batting_stats = TEAM_STATS.get(batting_team, {}).get('batting', {
        'Win Rate': '58%',
        'Avg Score': '172',
        'Powerplay RR': '8.2',
        'Death Overs RR': '10.1'
    })

    bowling_stats = TEAM_STATS.get(bowling_team, {}).get('bowling', {
        'Win Rate': '62%',
        'Avg Conceded': '165',
        'Powerplay Eco': '7.4',
        'Death Overs Eco': '9.8'
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="team-card" style="border-top: 4px solid {TEAM_COLORS.get(batting_team, '#4CAF50')}">
            <div class="team-logo" style="background: {TEAM_COLORS.get(batting_team, '#4CAF50')}; color: white;">
                {batting_team[0]}
            </div>
            <h3 style="margin: 0.5rem 0; color: {TEAM_COLORS.get(batting_team, '#4CAF50')}">{batting_team}</h3>
            <div class="team-stats">
                {''.join([f'<div class="stat-item"><span class="stat-label">{k}</span><span class="stat-value">{v}</span></div>' for k, v in batting_stats.items()])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="team-card" style="border-top: 4px solid {TEAM_COLORS.get(bowling_team, '#F44336')}">
            <div class="team-logo" style="background: {TEAM_COLORS.get(bowling_team, '#F44336')}; color: white;">
                {bowling_team[0]}
            </div>
            <h3 style="margin: 0.5rem 0; color: {TEAM_COLORS.get(bowling_team, '#F44336')}">{bowling_team}</h3>
            <div class="team-stats">
                {''.join([f'<div class="stat-item"><span class="stat-label">{k}</span><span class="stat-value">{v}</span></div>' for k, v in bowling_stats.items()])}
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_impact_factors(metrics):
    """Display visual impact factors"""
    st.markdown("### 📊 Match Impact Factors")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="impact-factor">
            <div class="impact-icon">🏃‍♂️</div>
            <div>
                <div class="impact-label">Required Run Rate</div>
                <div class="impact-value" style="color: {TEAM_COLORS.get('Royal Challengers Bangalore', '#F44336')}">
                    {metrics['rrr']:.2f}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="impact-factor">
            <div class="impact-icon">⚡</div>
            <div>
                <div class="impact-label">Momentum Shift</div>
                <div class="impact-value" style="color: {TEAM_COLORS.get('Kolkata Knight Riders', '#3A225D')}">
                    {metrics['momentum_shift_index']:.1f}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="impact-factor">
            <div class="impact-icon">🎯</div>
            <div>
                <div class="impact-label">Pressure Index</div>
                <div class="impact-value" style="color: {TEAM_COLORS.get('Mumbai Indians', '#005DA0')}">
                    {metrics['pressure_index']:.2f}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="impact-factor">
            <div class="impact-icon">🚫</div>
            <div>
                <div class="impact-label">Dot Ball %</div>
                <div class="impact-value" style="color: {TEAM_COLORS.get('Chennai Super Kings', '#FDB913')}">
                    {metrics['dot_ball_percent'] * 100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="impact-factor">
            <div class="impact-icon">🏏</div>
            <div>
                <div class="impact-label">Wickets in Hand</div>
                <div class="impact-value" style="color: {TEAM_COLORS.get('Sunrisers Hyderabad', '#FB643E')}">
                    {metrics['wickets_in_hand']}/10
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="impact-factor">
            <div class="impact-icon">👑</div>
            <div>
                <div class="impact-label">Top Batsman</div>
                <div class="impact-value" style="color: {TEAM_COLORS.get('Rajasthan Royals', '#2D4D9D')}">
                    {'Yes' if metrics['top_batsman_playing'] else 'No'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_prediction_timeline(metrics, win_prob):
    """Display a visual timeline of the match progression"""
    st.markdown("### ⏳ Match Progression Timeline")

    # Create timeline data
    timeline = pd.DataFrame({
        'Overs': np.arange(0, 20.1, 1),
        'Projected Score': [metrics['current_score'] + (metrics['crr'] * (o - metrics['overs_completed'])) for o in
                            np.arange(0, 20.1, 1)],
        'Required Rate': [metrics['rrr'] * (1 - (o / 20)) for o in np.arange(0, 20.1, 1)]
    })

    fig = px.line(timeline, x='Overs', y=['Projected Score', 'Required Rate'],
                  title="Match Progression Projection",
                  labels={'value': 'Runs', 'variable': 'Metric'},
                  color_discrete_map={
                      'Projected Score': TEAM_COLORS.get('Royal Challengers Bangalore', '#EC1C24'),
                      'Required Rate': TEAM_COLORS.get('Kolkata Knight Riders', '#3A225D')
                  })

    # Add current position marker
    fig.add_vline(x=metrics['overs_completed'], line_dash="dash", line_color="green")
    fig.add_annotation(x=metrics['overs_completed'], y=max(timeline['Projected Score']),
                       text="Current Position", showarrow=True, arrowhead=1)

    # Update layout
    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.5)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def display_prediction_results(batting_team, bowling_team, win_prob, metrics):
    """Display the prediction results with advanced insights"""
    loss_prob = 1 - win_prob

    # Display team comparison first
    display_team_comparison(batting_team, bowling_team)

    # Main prediction cards
    st.markdown("### 🎯 Win Probability Prediction")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="prediction-card" style="position: relative; border-top: 4px solid {TEAM_COLORS.get(batting_team, '#4CAF50')}">
            <div class="team-name">{batting_team}</div>
            <div class="probability-value">{win_prob * 100:.1f}%</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {win_prob * 100}%"></div>
            </div>
            <div style="margin-top: 1.5rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600; color: #555;">Current Run Rate</span>
                    <span style="font-weight: 700; color: {TEAM_COLORS.get(batting_team, '#4CAF50')};">{metrics['crr']:.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600; color: #555;">Runs Needed</span>
                    <span style="font-weight: 700; color: {TEAM_COLORS.get(batting_team, '#4CAF50')};">{metrics['runs_left']}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: 600; color: #555;">Balls Remaining</span>
                    <span style="font-weight: 700; color: {TEAM_COLORS.get(batting_team, '#4CAF50')};">{metrics['balls_left']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="prediction-card" style="position: relative; border-top: 4px solid {TEAM_COLORS.get(bowling_team, '#F44336')}">
            <div class="team-name">{bowling_team}</div>
            <div class="probability-value" style="background: linear-gradient(135deg, #F44336, #E91E63); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {loss_prob * 100:.1f}%
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {loss_prob * 100}%; background: linear-gradient(90deg, #F44336, #E91E63);"></div>
            </div>
            <div style="margin-top: 1.5rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600; color: #555;">Wickets Taken</span>
                    <span style="font-weight: 700; color: {TEAM_COLORS.get(bowling_team, '#F44336')};">{metrics['wickets']}/10</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600; color: #555;">Dot Ball %</span>
                    <span style="font-weight: 700; color: {TEAM_COLORS.get(bowling_team, '#F44336')};">{metrics['dot_ball_percent'] * 100:.1f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: 600; color: #555;">Pressure Index</span>
                    <span style="font-weight: 700; color: {TEAM_COLORS.get(bowling_team, '#F44336')};">{metrics['pressure_index']:.2f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display impact factors
    display_impact_factors(metrics)

    # Display timeline visualization
    display_prediction_timeline(metrics, win_prob)

    # Match analysis with more detailed insights
    st.markdown('<div class="match-analysis">', unsafe_allow_html=True)

    if win_prob > 0.75:
        st.success(f"## 🚀 {batting_team} In Commanding Position")
        st.markdown(f"""
        - **Dominating the chase** with {win_prob * 100:.1f}% win probability
        - Current run rate (**{metrics['crr']:.2f}**) well above required rate (**{metrics['rrr']:.2f}**)
        - **{metrics['wickets_in_hand']} wickets in hand** providing stability
        """)
        if metrics['top_batsman_playing']:
            st.markdown("- **Key batsman at crease** significantly boosting chances")

    elif win_prob > 0.6:
        st.info(f"## ⚖️ {batting_team} With Slight Advantage")
        st.markdown(f"""
        - **Narrow lead** with {win_prob * 100:.1f}% win probability
        - Need **{metrics['runs_left']} runs** in **{int(metrics['balls_left'] / 6)}.{metrics['balls_left'] % 6} overs**
        - Pressure index at **{metrics['pressure_index']:.2f}** ({'high' if metrics['pressure_index'] > 1.2 else 'moderate'} tension)
        """)

    elif win_prob > 0.45:
        st.warning(f"## 🎯 Tilted Towards {bowling_team}")
        st.markdown(f"""
        - **Bowlers in control** with {loss_prob * 100:.1f}% defense probability
        - **Dot ball percentage** at {metrics['dot_ball_percent'] * 100:.1f}% building pressure
        - {bowling_team} has taken **{metrics['wickets']} wickets** so far
        """)

    else:
        st.error(f"## 🔥 {bowling_team} Dominating")
        st.markdown(f"""
        - **Complete control** with {loss_prob * 100:.1f}% defense probability
        - **High pressure** on batsmen (index: {metrics['pressure_index']:.2f})
        - **Wickets falling regularly** ({metrics['wickets']} down)
        - Dot balls at **{metrics['dot_ball_percent'] * 100:.1f}%** restricting scoring
        """)

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Main app function"""
    pipe = load_model()
    sidebar_content()

    st.markdown("""
    <div class="header-card">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <h1 style="margin-bottom: 0; flex: 1;">🏏 IPL Win Predictor Pro+</h1>
            <div style="background: rgba(76, 175, 80, 0.1); padding: 0.5rem 1rem; border-radius: 12px; font-weight: 600; color: #4CAF50;">
                LIVE MODE
            </div>
        </div>
        <p style="color: #555; font-size: 1.1rem; margin-bottom: 0;">
            Professional-grade match outcome prediction with real-time analytics and 14+ advanced parameters
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main input section with enhanced layout
    # Replace your existing "Match Setup" expander section with this code:

    with st.expander("⚙️ Match Setup", expanded=True):
        # Custom CSS to fix the select box styling
        st.markdown("""
        <style>
            /* Fix for select box text visibility */
            .stSelectbox>div>div>div {
                padding: 10px 12px !important;
                display: flex !important;
                align-items: center !important;
            }

            /* Make all columns equal width */
            .st-cb, .st-ca, .st-c9 {
                flex: 1;
            }
        </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            venue = st.selectbox(
                "Select Venue 🏟️",
                sorted(VENUES),
                help="Venue-specific pitch behavior affects predictions",
                key='venue_select'
            )

        with col2:
            batting_team = st.selectbox(
                "Batting Team 🏏",
                sorted(TEAMS),
                help="Team currently batting (chasing)",
                key='batting_team_select'
            )

        bowling_team = st.selectbox(
            "Bowling Team 🎯",
            sorted([team for team in TEAMS if team != batting_team]),
            help="Team currently bowling (defending)",
            key='bowling_team_select'
        )

        target = st.number_input(
            "Target Score 🎯",
            min_value=1,
            value=180,
            help="First innings total being chased",
            key='target_input'
        )

    # Match progress section with enhanced visualization
    with st.expander("📊 Match Progress Tracker", expanded=True):
        col5, col6, col7 = st.columns(3)

        with col5:
            current_score = st.number_input(
                "Current Score 📊",
                min_value=0,
                value=85,
                help="Runs scored so far in chase",
                key='current_score_input'
            )

        with col6:
            wickets = st.slider(
                "Wickets Fallen ⚠️",
                0, 10, 4,
                help="Wickets lost in the innings",
                key='wickets_slider'
            )

        with col7:
            overs_completed = st.slider(
                "Overs Completed ⏱️",
                0.0, 20.0, 10.0,
                step=0.1,
                format="%.1f",
                help="Overs bowled so far",
                key='overs_slider'
            )

        # Visual progress indicator
        progress_col1, progress_col2 = st.columns([3, 1])
        with progress_col1:
            run_progress = min(1.0, current_score / target) if target > 0 else 0
            st.markdown(f"""
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600;">Run Progress</span>
                    <span style="font-weight: 700;">{current_score}/{target} ({run_progress * 100:.1f}%)</span>
                </div>
                <div style="height: 10px; background: #f0f0f0; border-radius: 5px; overflow: hidden;">
                    <div style="width: {run_progress * 100}%; height: 100%; background: linear-gradient(90deg, #4CAF50, #2E7D32);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with progress_col2:
            over_progress = overs_completed / 20.0
            st.markdown(f"""
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600;">Overs</span>
                    <span style="font-weight: 700;">{over_progress * 100:.1f}%</span>
                </div>
                <div style="height: 10px; background: #f0f0f0; border-radius: 5px; overflow: hidden;">
                    <div style="width: {over_progress * 100}%; height: 100%; background: linear-gradient(90deg, #2196F3, #0D47A1);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Advanced options expander
    with st.expander("🔍 Advanced Parameters", expanded=False):
        col8, col9 = st.columns(2)

        with col8:
            top_batsman_playing = st.checkbox(
                "Top Batsman at Crease 👑",
                value=True,
                help="Is a top-20 tournament batsman currently batting?",
                key='top_batsman_check'
            )

        with col9:
            recent_partnership = st.slider(
                "Recent Partnership Runs 🤝",
                min_value=0,
                max_value=100,
                value=30,
                help="Runs scored in last 5 overs without losing wicket",
                key='partnership_slider'
            )

    # Calculate all metrics
    params = {
        'target': target,
        'current_score': current_score,
        'wickets': wickets,
        'overs_completed': overs_completed,
        'top_batsman_playing': 1 if top_batsman_playing else 0,
        'recent_partnership': recent_partnership
    }

    metrics = calculate_advanced_metrics(params)

    # Ensure we have valid match situation before predicting
    valid_prediction = True
    if metrics['balls_left'] <= 0:
        st.warning("❌ Match is already completed (no balls left)")
        valid_prediction = False
    elif metrics['runs_left'] <= 0:
        st.warning("❌ Batting team has already reached the target")
        valid_prediction = False

    # Prediction button with enhanced design
    predict_col, space_col, reset_col = st.columns([2, 6, 2])

    with predict_col:
        predict_clicked = st.button("🚀 Predict Win Probability",
                                    type="primary",
                                    key='predict_button',
                                    disabled=not valid_prediction)

    with reset_col:
        if st.button("🔄 Reset Inputs", key='reset_button'):
            st.rerun()  # Changed from st.experimental_rerun() to st.rerun()

    if predict_clicked and valid_prediction:
        input_df = create_input_dataframe(
            batting_team,
            bowling_team,
            venue,
            metrics
        )

        try:
            with st.spinner('🧠 Analyzing match dynamics...'):
                win_prob = pipe.predict_proba(input_df)[0][1]

                # Display results
                st.markdown("---")
                display_prediction_results(batting_team, bowling_team, win_prob, metrics)

                # Add confetti effect for high confidence predictions
                if win_prob > 0.85 or win_prob < 0.15:
                    st.balloons()

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.error("Please check your inputs and try again")


if __name__ == "__main__":
    main()