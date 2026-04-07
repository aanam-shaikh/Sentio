"""
Streamlit Web Interface for Mental Health Video Analysis
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
from deepface import DeepFace
import pandas as pd
import tempfile
import os
from mental_health_analyzer import MentalHealthAnalyzer
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="SENTIO",
    layout="wide"
)

# Custom CSS
st.markdown("""
<div style='text-align: center; margin-top: -20px;'>

<h1 style='
font-size: 3.5rem; 
font-weight: 700; 
margin-bottom: 5px;
'>Sentio</h1>

<p style='
font-size: 1.8rem; 
font-weight: 650; 
color: #A7F3D0;
margin-top: 0;
'>
Understand Emotions From Video Using AI
</p>

</div>
""", unsafe_allow_html=True)


st.markdown("""
<style>
  /* ── Your existing classes (Sage Mind updated) ── */
  .main-header {
      font-size: 3rem;
      color: #3dba7e;
      text-align: center;
      margin-bottom: 2rem;
  }
  .metric-card {
      background-color: #192e23;
      padding: 1rem;
      border-radius: 0.75rem;
      margin: 0.5rem 0;
      border: 1px solid #234033;
  }

  /* ── Sage Mind theme ── */
  .stTabs [data-baseweb="tab"][aria-selected="true"] {
      color: #3dba7e !important;
      border-bottom: 2px solid #3dba7e !important;
  }
  .stTabs [data-baseweb="tab"] {
      color: #527a65 !important;
  }
  [data-testid="metric-container"] {
      background: #192e23 !important;
      border: 1px solid #234033;
      border-radius: 8px;
      padding: 12px;
  }
  [data-testid="metric-container"] label {
      color: #5ec994 !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      color: #e8f3ed !important;
  }
  .stFileUploader label {
      background: #3dba7e !important;
      color: #0f1f18 !important;
      border-radius: 8px !important;
  }
  [data-testid="stSidebar"] {
      background: #192e23 !important;
      border-right: 1px solid #234033 !important;
  }
  [data-testid="stSidebar"] .stRadio label,
  [data-testid="stSidebar"] p {
      color: #e8f3ed !important;
  }
  /* Radio buttons */
  .stRadio [data-baseweb="radio"] div[aria-checked="true"] {
      background-color: #3dba7e !important;
      border-color: #3dba7e !important;
  }
  /* Progress/status bars */
  .stProgress > div > div {
      background-color: #3dba7e !important;
  }
  /* Buttons */
  .stButton > button {
      background-color: #3dba7e !important;
      color: #0f1f18 !important;
      border: none !important;
      border-radius: 8px !important;
      font-weight: 500 !important;
  }
  .stButton > button:hover {
      background-color: #34a870 !important;
  }
  /* Highlight accent (amber/gold for contrast) */
  .highlight {
      color: #f0c96b;
      font-weight: 500;
  }
  /* Upload drop zone */
  [data-testid="stFileUploadDropzone"] {
      background: #192e23 !important;
      border: 2px dashed #234033 !important;
      border-radius: 10px !important;
  }
  /* Dataframes / tables */
  [data-testid="stDataFrame"] {
      border: 1px solid #234033 !important;
  }
  /* Selectbox / dropdowns */
  [data-baseweb="select"] {
      background-color: #192e23 !important;
  }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.df = None

# Sidebar
with st.sidebar:
    st.header("About This Project")

    st.markdown("""
    <div style='
    background-color: #064E3B;
    padding: 15px;
    border: 1px solid #10B981;
    border-radius: 10px;
    color: #D1FAE5;
    '>

    This system uses <b>AI</b> and <b>Computer Vision</b> to analyze mental health indicators from facial emotions.

    <br>

    <b>Mental Health Indicators:</b>
    <ul>
        <li>Depression Score</li>
        <li>Anxiety Level</li>
        <li>Stress Assessment</li>
        <li>Emotional Stability</li>
        <li>Overall Wellness</li>
    </ul>

    <b>Features:</b>
    <ul>
        <li>Clinical score interpretation</li>
        <li>Personalized recommendations</li>
        <li>Trend analysis</li>
        <li>Professional reports</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)
    
    st.header("Settings")
    skip_frames = st.slider("Frame Skip (for faster processing)", 1, 10, 5)
    max_frames = st.number_input("Max frames to analyze (0 = all)", 0, 1000, 0)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Video Analysis", "Mental Health Report", "Detailed Statistics", "Project Info"])

with tab1:
    st.header("Upload and Analyze Video")
    
    analysis_mode = st.radio(
        "Choose Analysis Mode:",
        ["Upload Video File", "Use Webcam (Coming Soon)"],
        horizontal=True
    )
    
    if analysis_mode == "Upload Video File":
        
        st.markdown("### Upload Video")

        uploaded_file = st.file_uploader(
                "",
                type=['mp4', 'avi', 'mov', 'mkv'],
                label_visibility="collapsed"
            )

        
        if uploaded_file is not None:
            # Display video
            st.video(uploaded_file)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # Analyze button
            if st.button(" Start Analysis", type="primary"):
                with st.spinner("Analyzing emotions and mental health indicators... This may take a while."):
                    try:
                        analyzer = MentalHealthAnalyzer()
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Processing video frames...")
                        
                        # Analyze video
                        max_f = max_frames if max_frames > 0 else None
                        df = analyzer.analyze_video_file(
                            video_path, 
                            skip_frames=skip_frames,
                            max_frames=max_f
                        )
                        
                        progress_bar.progress(50)
                        status_text.text("Calculating mental health scores...")
                        
                        # Calculate mental health scores
                        scores = analyzer.calculate_mental_health_scores(df)
                        interpretations = analyzer.interpret_scores(scores)
                        recommendations = analyzer.generate_recommendations(scores, interpretations)
                        
                        progress_bar.progress(75)
                        status_text.text("Generating visualizations...")
                        
                        # Create visualizations
                        analyzer.create_visualizations(df, scores)
                        
                        # Save to session state
                        st.session_state.df = df
                        st.session_state.scores = scores
                        st.session_state.interpretations = interpretations
                        st.session_state.recommendations = recommendations
                        st.session_state.analyzed = True
                        st.session_state.analyzer = analyzer
                        
                        progress_bar.progress(100)
                        status_text.text("Analysis Complete!")
                        
                        st.success(" Mental Health Analysis Completed Successfully!")
                        st.balloons()
                        
                        # Clean up temp file
                        os.unlink(video_path)
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        st.error("Please ensure DeepFace is installed: pip install deepface")

with tab2:
    st.header("Mental Health Analysis Report")
    
    if not st.session_state.analyzed:
        st.warning("Please analyze a video first in the 'Video Analysis' tab.")
    else:
        df = st.session_state.df
        scores = st.session_state.scores
        interpretations = st.session_state.interpretations
        recommendations = st.session_state.recommendations
        
        # Mental Health Scores
        st.subheader(" Mental Health Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            depression_color = "🔴" if scores['depression_score'] > 60 else "🟡" if scores['depression_score'] > 30 else "🟢"
            st.metric(
                "Depression Score",
                f"{scores['depression_score']:.1f}/100",
                delta=None,
                help="Lower is better"
            )
            st.write(f"{depression_color} {interpretations['depression']['level']}")
        
        with col2:
            anxiety_color = "🔴" if scores['anxiety_score'] > 60 else "🟡" if scores['anxiety_score'] > 30 else "🟢"
            st.metric(
                "Anxiety Score",
                f"{scores['anxiety_score']:.1f}/100",
                help="Lower is better"
            )
            st.write(f"{anxiety_color} {interpretations['anxiety']['level']}")
        
        with col3:
            stress_color = "🔴" if scores['stress_score'] > 60 else "🟡" if scores['stress_score'] > 30 else "🟢"
            st.metric(
                "Stress Score",
                f"{scores['stress_score']:.1f}/100",
                help="Lower is better"
            )
            st.write(f"{stress_color} {interpretations['stress']['level']}")
        
        with col4:
            wellness_color = "🟢" if scores['wellness_score'] > 70 else "🟡" if scores['wellness_score'] > 40 else "🔴"
            st.metric(
                "Wellness Score",
                f"{scores['wellness_score']:.1f}/100",
                help="Higher is better"
            )
            st.write(f"{wellness_color} {interpretations['wellness']['level']}")
        
        # Score visualization
        st.subheader(" Mental Health Metrics Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of scores
            fig_scores = go.Figure()
            
            categories = ['Depression', 'Anxiety', 'Stress']
            values = [scores['depression_score'], scores['anxiety_score'], scores['stress_score']]
            colors_list = ['red' if v > 60 else 'orange' if v > 30 else 'green' for v in values]
            
            fig_scores.add_trace(go.Bar(
                x=categories,
                y=values,
                marker_color=colors_list,
                text=[f"{v:.1f}" for v in values],
                textposition='outside'
            ))
            
            fig_scores.update_layout(
                title="Mental Health Indicator Scores",
                yaxis_title="Score (0-100)",
                yaxis=dict(range=[0, 100]),
                showlegend=False
            )
            
            # Add threshold lines
            fig_scores.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5,
                                annotation_text="Low Threshold")
            fig_scores.add_hline(y=60, line_dash="dash", line_color="orange", opacity=0.5,
                                annotation_text="High Threshold")
            
            st.plotly_chart(fig_scores, use_container_width=True)
        
        with col2:
            # Wellness gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=scores['wellness_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Wellness"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightcoral"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Clinical Interpretations
        st.subheader("Clinical Interpretation")
        
        for key, interp in interpretations.items():
            if key == 'wellness':
                continue
            
            color_map = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}
            icon = color_map.get(interp['color'], '⚪')
            
            with st.expander(f"{icon} {key.upper()} - {interp['level']} Risk", expanded=True):
                st.write(interp['description'])

                if key == 'depression':
                    st.markdown("""
                    <div style='
                    background-color: #065F46;
                    padding: 10px;
                    border-radius: 10px;
                    color: #D1FAE5;
                    border: 1px solid #022C22;
                    margin-bottom: 10px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    position: relative;
                    z-index: 1;
                    '>

                    <b>What this means:</b>
                    <ul>
                        <li>Depression indicators are based on prolonged sad, neutral, or angry expressions</li>
                        <li>Consistent low mood across the video suggests potential depressive patterns</li>
                        <li>This is NOT a clinical diagnosis</li>
                    </ul>

                    </div>
                    """, unsafe_allow_html=True)
                
                
                elif key == 'anxiety':
                    st.markdown("""
                    <div style='
                    background-color: #065F46;
                    padding: 10px;
                    border-radius: 10px;
                    color: #D1FAE5;
                    border: 1px solid #022C22;
                    margin-bottom: 10px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    position: relative;
                    z-index: 1;
                    '>
                                
                    <b>What this means:</b>
                    <ul>
                    <li>Anxiety indicators come from fear, surprise, and agitated expressions</li>
                    <li>Elevated scores suggest heightened worry or nervousnesss</li>
                    <li>Physical anxiety may not always show in facial expressions</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)


                elif key == 'stress':
                    st.markdown("""
                    <div style='
                    background-color: #065F46;
                    padding: 10px;
                    border-radius: 10px;
                    color: #D1FAE5;
                    border: 1px solid #022C22;
                    margin-bottom: 10px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    position: relative;
                    z-index: 1;
                    '>
                    <b>What this means:</b>
                    <ul>
                        <li>Stress scores reflect anger, disgust, and tension in expressions</li>
                        <li>High scores indicate possible pressure or overwhelm</li>
                        <li>Chronic stress can impact mental and physical health</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("Personalized Recommendations")
        
        if not recommendations:
            st.success(" No significant concerns detected. Continue maintaining good mental health practices!")
        else:
            for rec in recommendations:
                priority_icon = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Moderate' else "🟢"
                
                with st.expander(f"{priority_icon} {rec['category']} - {rec['priority']} Priority", expanded=(rec['priority'] == 'High')):
                    for suggestion in rec['suggestions']:
                        st.write(f"✓ {suggestion}")
        
        # Emotional Stability
        st.subheader(" Emotional Stability Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Stability over time
            window_size = max(1, len(df) // 20)
            stability_timeline = []
            timestamps = []
            
            for i in range(0, len(df) - window_size):
                window = df['dominant_emotion'].iloc[i:i+window_size]
                changes = (window != window.shift()).sum()
                stability = 100 - (changes / len(window) * 100)
                stability_timeline.append(stability)
                timestamps.append(df['timestamp'].iloc[i])
            
            if stability_timeline:
                fig_stability = go.Figure()
                fig_stability.add_trace(go.Scatter(
                    x=timestamps,
                    y=stability_timeline,
                    mode='lines',
                    fill='tozeroy',
                    name='Stability',
                    line=dict(color='royalblue', width=2)
                ))
                
                fig_stability.add_hline(
                    y=scores['emotional_stability'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Average: {scores['emotional_stability']:.1f}%"
                )
                
                fig_stability.update_layout(
                    title="Emotional Stability Timeline",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Stability (%)",
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig_stability, use_container_width=True)
        
        with col2:
            st.metric(
                "Emotional Stability",
                f"{scores['emotional_stability']:.1f}%",
                help="Higher stability = more consistent emotional state"
            )
            
            if scores['emotional_stability'] > 80:
                st.success("Very stable emotional state")
            elif scores['emotional_stability'] > 60:
                st.info("Moderately stable emotional state")
            else:
                st.warning("High emotional variability detected")
            
            st.write("**What this means:**")
            st.write(f"Your emotions changed {(100 - scores['emotional_stability']):.0f}% of the time analyzed.")
        
        # Disclaimer
        st.divider()
        st.error("""
        **IMPORTANT DISCLAIMER**
        
        This analysis is based on facial emotion recognition and provides **general indicators only**.
        
        - This is **NOT a diagnostic tool**
        - Results should NOT replace professional mental health assessment
        - Emotions can be influenced by many temporary factors
        - PLEASE consult a qualified mental health professional for proper evaluation
        
        If you're experiencing mental health concerns, please reach out to:
        - A licensed therapist or counselor
        - Your primary care physician
        - Mental health crisis hotline (14416 (Tele-MANAS) in India)
        """)

with tab3:
    st.header("Analysis Results")
    
    if not st.session_state.analyzed:
        st.warning("Please analyze a video first in the 'Video Analysis' tab.")
    else:
        df = st.session_state.df
        analyzer = st.session_state.analyzer
       
        
        # Key Metrics
        # Key Metrics
        st.subheader("Key Metrics")

        total_frames = len(df)
        duration = df['timestamp'].max()

        emotion_counts = df['dominant_emotion'].value_counts()
        emotion_percentages = (emotion_counts / total_frames) * 100

        most_common = emotion_counts.idxmax()

        emotion_changes = (df['dominant_emotion'] != df['dominant_emotion'].shift()).sum()
        stability = 100 - (emotion_changes / total_frames * 100)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Frames Analyzed", total_frames)

        with col2:
            st.metric("Duration", f"{duration:.1f}s")

        with col3:
            st.metric("Dominant Emotion", most_common.capitalize())

        with col4:
            st.metric("Stability Score", f"{stability:.0f}%")
        # Emotion Distribution
        st.subheader( "Emotion Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            emotion_counts = df['dominant_emotion'].value_counts()
            emotion_percentages = (emotion_counts / len(df)) * 100
            
            fig_pie = px.pie(
               values=emotion_counts.values,
                names=emotion_counts.index,
                title="Emotion Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
               x=emotion_percentages.index,
               y=emotion_percentages.values,
                title="Emotion Percentages",
                labels={'x': 'Emotion', 'y': 'Percentage (%)'},
                color=emotion_percentages.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Emotion Timeline
        st.subheader("Emotion Timeline")
        
        # Interactive timeline
        fig_timeline = go.Figure()
        
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        colors = ['red', 'brown', 'purple', 'gold', 'blue', 'orange', 'gray']
        
        for emotion, color in zip(emotion_labels, colors):
            if emotion in df.columns:
                fig_timeline.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[emotion],
                    mode='lines',
                    name=emotion.capitalize(),
                    line=dict(color=color, width=2)
                ))
        
        fig_timeline.update_layout(
            title="Emotion Scores Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Emotion Score",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Dominant emotion over time
        st.subheader("Dominant Emotion Over Time")
        
        fig_dominant = px.scatter(
            df,
            x='timestamp',
            y='dominant_emotion',
            color='dominant_emotion',
            title="Dominant Emotion Timeline",
            labels={'timestamp': 'Time (seconds)', 'dominant_emotion': 'Emotion'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        st.plotly_chart(fig_dominant, use_container_width=True)
        
        # Average Scores
        st.subheader(" Average Emotion Scores")
        
        avg_scores = df[['angry','disgust','fear','happy','sad','surprise','neutral']].mean()

        avg_scores_df = pd.DataFrame({
            'Emotion': avg_scores.index,
            'Average Score': avg_scores.values
        }).sort_values('Average Score', ascending=False)
        
        fig_avg = px.bar(
            avg_scores_df,
            x='Emotion',
            y='Average Score',
            title="Average Emotion Scores Across Video",
            color='Average Score',
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig_avg, use_container_width=True)
        
        # Detailed Statistics Table
        st.subheader(" Detailed Statistics")
        
        emotion_counts = df['dominant_emotion'].value_counts()
        emotion_percentages = (emotion_counts / len(df)) * 100

        avg_scores = df[['angry','disgust','fear','happy','sad','surprise','neutral']].mean()

        stats_table = pd.DataFrame({
            'Emotion': emotion_counts.index,
            'Frame Count': emotion_counts.values,
            'Percentage': [f"{pct:.2f}%" for pct in emotion_percentages.values],
            'Avg Score': [f"{avg_scores[emo]:.2f}" for emo in emotion_counts.index]
        }).sort_values('Frame Count', ascending=False)
        
        st.dataframe(stats_table, use_container_width=True)
        
        # Download buttons
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Raw Data (CSV)",
                data=csv,
                file_name="emotion_analysis.csv",
                mime="text/csv"
            )
        
        with col2:
            # Generate report
            report = f"""
            MENTAL HEALTH VIDEO ANALYSIS - EMOTION RECOGNITION REPORT
            {'=' * 60}

            Total Frames Analyzed: {len(df)}
            Video Duration: {df['timestamp'].max():.2f} seconds

            EMOTION DISTRIBUTION
            {'-' * 60}
            """

# Emotion calculations
            emotion_counts = df['dominant_emotion'].value_counts()
            emotion_percentages = (emotion_counts / len(df)) * 100

            # Add distribution to report
            for emotion, pct in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
                report += f"{emotion.capitalize():12} : {pct:6.2f}% ({emotion_counts[emotion]} frames)\n"

            # Additional metrics
            most_common = emotion_counts.idxmax()

            emotion_changes = (df['dominant_emotion'] != df['dominant_emotion'].shift()).sum()
            stability = 100 - (emotion_changes / len(df) * 100)

            report += f"\nMost Prevalent Emotion: {most_common.capitalize()} ({emotion_percentages[most_common]:.1f}%)\n"
            report += f"Emotional Stability Score: {stability:.1f}%\n"
            
            st.download_button(
                label="Download Report (TXT)",
                data=report,
                file_name="analysis_report.txt",
                mime="text/plain"
            )

with tab4:
    st.header("Project Information")
    
    st.markdown("""
    ### Project Overview
    
    This **Mental Health Video Analysis** system uses artificial intelligence to detect and interpret 
    facial emotions, providing insights into potential mental health indicators including:
    
    - **Depression Assessment**: Analyzes prolonged sadness and low mood patterns
    - **Anxiety Detection**: Identifies fear, worry, and nervousness indicators  
    - **Stress Evaluation**: Measures tension and overwhelm from facial expressions
    - **Emotional Stability**: Tracks consistency of emotional states
    - **Overall Wellness**: Provides a comprehensive mental health overview
    
    ### Clinical Applications
    
    This technology can support:
    - **Therapists**: Objective data to supplement clinical observations
    - **Researchers**: Large-scale emotion pattern analysis
    - **Self-monitoring**: Personal mental health awareness
    - **Remote care**: Monitoring patients between sessions
    - **Early intervention**: Detecting concerning patterns early
    
    ### Technology Stack
    
    - **DeepFace**: State-of-the-art facial emotion recognition models
    - **Computer Vision**: Video processing and face detection (OpenCV)
    - **Machine Learning**: Pre-trained deep neural networks
    - **Data Science**: Statistical analysis and pattern recognition (Pandas)
    - **Web Framework**: Interactive dashboard (Streamlit)
    - **Visualization**: Charts and graphs (Plotly, Matplotlib)
    
    ### How the Analysis Works
    
    1. **Video Input**: User uploads video or uses webcam
    2. **Frame Processing**: System extracts individual frames
    3. **Face Detection**: Identifies faces using computer vision
    4. **Emotion Recognition**: AI analyzes facial expressions for 7 emotions:
       - Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
    5. **Mental Health Scoring**: Weighted algorithm calculates indicators:
       - Each emotion has research-based weights for depression/anxiety/stress
       - Scores normalized to 0-100 scale
       - Temporal patterns analyzed for stability
    6. **Clinical Interpretation**: Scores classified into Low/Moderate/High
    7. **Recommendations**: Personalized suggestions based on findings
    
    ### Scoring Methodology
    
    **Depression Score:**
    - Weighted heavily on sad, neutral, and angry expressions
    - Low happy emotions increase score
    - Based on emotional patterns in depression research
    
    **Anxiety Score:**
    - Emphasizes fear, surprise, and worried expressions
    - Considers emotional variability
    - Reflects nervousness and apprehension patterns
    
    **Stress Score:**
    - Focuses on anger, disgust, and tension
    - Measures sustained negative affect
    - Indicates pressure and overwhelm
    
    **Wellness Score:**
    - Inverse of average negative indicators
    - Higher positive emotions improve score
    - Overall mental health snapshot
    
    ### Educational Value
    
    This project demonstrates:
    - Deep learning applications in healthcare
    - Video processing and computer vision
    - Statistical analysis and data interpretation
    - Ethical AI development for sensitive applications
    - Building user-friendly ML applications
    - Integration of multiple AI technologies
    
    ###  Limitations & Ethical Considerations
    
    **Limitations:**
    - Not a diagnostic tool - provides indicators only
    - Facial expressions don't capture full emotional state
    - Cultural variations in emotion expression exist
    - Temporary factors can influence results
    - Model accuracy varies by lighting/angle/quality
    - Cannot detect internal mental states directly
    
    **Ethical Guidelines:**
    - **Privacy**: Always obtain informed consent
    - **Transparency**: Clear about what the system does/doesn't do
    - **Non-maleficence**: Avoid causing harm through misinterpretation
    - **Beneficence**: Use to support, not replace, professional care
    - **Autonomy**: Respect individual choice in using results
    - **Justice**: Ensure fair access and unbiased analysis
    
    ### Research Foundation
    
    This system is based on established research:
    - Facial Action Coding System (FACS)
    - Emotion recognition in mental health literature
    - Deep learning for facial expression analysis
    - Clinical indicators of depression, anxiety, and stress
    
    ### Future Enhancements
    
    Potential improvements:
    - Multi-modal analysis (voice, text, physiological)
    - Longitudinal tracking over time
    - Personalized baseline comparison
    - Integration with wearable devices
    - Cultural adaptation of models
    - Mobile application deployment
    - Real-time intervention triggers
    
    ### When to Seek Professional Help
    
    Please consult a mental health professional if you experience:
    - Persistent sadness or hopelessness
    - Excessive worry or fear
    - Changes in sleep or appetite
    - Loss of interest in activities
    - Difficulty concentrating
    - Thoughts of self-harm
    - Significant life impairment
    
    **Crisis Resources:**
    - National Suicide Prevention Lifeline: 14416 (Tele-MANAS)
    - Indian Association for Suicide Prevention: https://spif.in/about-spif/
    
    ### Disclaimer
    
    This system is designed for **educational and research purposes**. It provides 
    general mental health indicators based on facial emotion patterns.
    
    **This is NOT:**
    - A diagnostic tool for mental illness
    - A replacement for professional mental health care
    - Medical advice or clinical assessment
    - Suitable for making treatment decisions
    
    **This IS:**
    - An educational demonstration of AI in mental health
    - A tool for general emotional pattern awareness
    - Support for self-monitoring and reflection
    - Technology to assist (not replace) professionals
    
    ---
    
    ### Acknowledgments
    
    - **DeepFace**: For accessible emotion recognition models
    - **Mental Health Research Community**: For clinical insights
    - **Open Source Community**: For tools and frameworks
    
    ### About
    
    Developed as an educational project demonstrating the application of 
    artificial intelligence in mental health awareness and support.
    
    **Remember**: Your mental health matters. This tool is one resource among many. 
    Professional support is available and encouraged when needed.
    """)
    
    st.success("Thank you for using Mental Health Video Analysis System!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Mental Health Video Analysis | "
    "Educational Project | Not for Clinical Diagnosis | "
    "Powered by DeepFace & AI</div>",
    unsafe_allow_html=True
)
