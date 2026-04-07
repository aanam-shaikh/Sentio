"""
Mental Health Analysis System
Analyzes emotions to detect potential mental health indicators
"""

import cv2
from deepface import DeepFace
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt

class MentalHealthAnalyzer:
    """Analyzes mental health indicators from facial emotion patterns"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotions_data = []
        
        # Mental health scoring weights based on research
        # These weights indicate how each emotion correlates with mental health issues
        self.depression_weights = {
            'sad': 3.0,
            'neutral': 1.5,
            'angry': 2.0,
            'fear': 1.5,
            'happy': -2.0,  # Negative weight (reduces depression score)
            'surprise': -0.5,
            'disgust': 1.0
        }
        
        self.anxiety_weights = {
            'fear': 3.0,
            'surprise': 1.5,
            'angry': 2.0,
            'sad': 1.5,
            'neutral': 1.0,
            'happy': -2.0,
            'disgust': 1.0
        }
        
        self.stress_weights = {
            'angry': 3.0,
            'disgust': 2.0,
            'fear': 2.5,
            'sad': 1.5,
            'neutral': 0.5,
            'happy': -2.0,
            'surprise': 0.5
        }
    
    def analyze_video_file(self, video_path, skip_frames=5, max_frames=None):
        """Analyze video and extract emotion data"""
        print(f"Analyzing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing {total_frames} frames at {fps} FPS")
        print(f"Analyzing every {skip_frames} frames...")
        
        self.emotions_data = []
        frame_count = 0
        processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            if max_frames and processed >= max_frames:
                break
            
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                emotion_data = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'dominant_emotion': result[0]['dominant_emotion']
                }
                
                for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                    emotion_data[emotion] = result[0]['emotion'][emotion]
                
                self.emotions_data.append(emotion_data)
                processed += 1
                
                if processed % 10 == 0:
                    print(f"Processed {processed} frames...")
                    
            except Exception as e:
                print(f"Error at frame {frame_count}: {e}")
            
            frame_count += 1
        
        cap.release()
        print(f"\nComplete! Analyzed {processed} frames")
        
        df = pd.DataFrame(self.emotions_data)
        return df
    
    def calculate_mental_health_scores(self, df):
        """Calculate mental health indicator scores"""
        scores = {
            'depression_score': 0,
            'anxiety_score': 0,
            'stress_score': 0
        }
        
        # Calculate weighted scores based on emotion distribution
        emotion_counts = df['dominant_emotion'].value_counts()
        total_frames = len(df)
        
        for emotion, count in emotion_counts.items():
            percentage = count / total_frames
            
            scores['depression_score'] += self.depression_weights.get(emotion, 0) * percentage
            scores['anxiety_score'] += self.anxiety_weights.get(emotion, 0) * percentage
            scores['stress_score'] += self.stress_weights.get(emotion, 0) * percentage
        
        # Normalize to 0-100 scale
        scores['depression_score'] = max(0, min(100, scores['depression_score'] * 50))
        scores['anxiety_score'] = max(0, min(100, scores['anxiety_score'] * 50))
        scores['stress_score'] = max(0, min(100, scores['stress_score'] * 50))
        
        # Calculate emotional variability (instability indicator)
        emotion_changes = (df['dominant_emotion'] != df['dominant_emotion'].shift()).sum()
        scores['emotional_stability'] = max(0, 100 - (emotion_changes / len(df) * 100))
        
        # Overall mental wellness score (inverse of average negative indicators)
        scores['wellness_score'] = 100 - (
            (scores['depression_score'] + scores['anxiety_score'] + scores['stress_score']) / 3
        )
        
        return scores
    
    def interpret_scores(self, scores):
        """Provide clinical interpretation of scores"""
        interpretations = {}
        
        # Depression interpretation
        if scores['depression_score'] < 30:
            interpretations['depression'] = {
                'level': 'Low',
                'description': 'Minimal depressive indicators detected',
                'color': 'green'
            }
        elif scores['depression_score'] < 60:
            interpretations['depression'] = {
                'level': 'Moderate',
                'description': 'Some depressive patterns observed',
                'color': 'yellow'
            }
        else:
            interpretations['depression'] = {
                'level': 'High',
                'description': 'Significant depressive indicators detected',
                'color': 'red'
            }
        
        # Anxiety interpretation
        if scores['anxiety_score'] < 30:
            interpretations['anxiety'] = {
                'level': 'Low',
                'description': 'Minimal anxiety indicators',
                'color': 'green'
            }
        elif scores['anxiety_score'] < 60:
            interpretations['anxiety'] = {
                'level': 'Moderate',
                'description': 'Some anxiety patterns present',
                'color': 'yellow'
            }
        else:
            interpretations['anxiety'] = {
                'level': 'High',
                'description': 'Elevated anxiety indicators',
                'color': 'red'
            }
        
        # Stress interpretation
        if scores['stress_score'] < 30:
            interpretations['stress'] = {
                'level': 'Low',
                'description': 'Low stress levels detected',
                'color': 'green'
            }
        elif scores['stress_score'] < 60:
            interpretations['stress'] = {
                'level': 'Moderate',
                'description': 'Moderate stress indicators',
                'color': 'yellow'
            }
        else:
            interpretations['stress'] = {
                'level': 'High',
                'description': 'High stress levels observed',
                'color': 'red'
            }
        
        # Overall wellness
        if scores['wellness_score'] > 70:
            interpretations['wellness'] = {
                'level': 'Good',
                'description': 'Generally positive mental state',
                'color': 'green'
            }
        elif scores['wellness_score'] > 40:
            interpretations['wellness'] = {
                'level': 'Fair',
                'description': 'Some concerns present',
                'color': 'yellow'
            }
        else:
            interpretations['wellness'] = {
                'level': 'Poor',
                'description': 'Multiple concerns detected',
                'color': 'red'
            }
        
        return interpretations
    
    def generate_recommendations(self, scores, interpretations):
        """Generate personalized recommendations"""
        recommendations = []
        
        # Depression recommendations
        if scores['depression_score'] > 60:
            recommendations.append({
                'category': 'Depression',
                'priority': 'High',
                'suggestions': [
                    'Consider speaking with a mental health professional',
                    'Engage in regular physical activity',
                    'Maintain social connections',
                    'Practice mindfulness or meditation',
                    'Establish a regular sleep schedule'
                ]
            })
        elif scores['depression_score'] > 30:
            recommendations.append({
                'category': 'Depression',
                'priority': 'Moderate',
                'suggestions': [
                    'Monitor your mood patterns',
                    'Engage in activities you enjoy',
                    'Spend time outdoors',
                    'Connect with supportive friends/family'
                ]
            })
        
        # Anxiety recommendations
        if scores['anxiety_score'] > 60:
            recommendations.append({
                'category': 'Anxiety',
                'priority': 'High',
                'suggestions': [
                    'Practice deep breathing exercises',
                    'Consider professional counseling',
                    'Limit caffeine intake',
                    'Try progressive muscle relaxation',
                    'Keep a worry journal'
                ]
            })
        elif scores['anxiety_score'] > 30:
            recommendations.append({
                'category': 'Anxiety',
                'priority': 'Moderate',
                'suggestions': [
                    'Practice relaxation techniques',
                    'Maintain regular exercise routine',
                    'Limit news/social media consumption',
                    'Practice grounding techniques'
                ]
            })
        
        # Stress recommendations
        if scores['stress_score'] > 60:
            recommendations.append({
                'category': 'Stress',
                'priority': 'High',
                'suggestions': [
                    'Identify and address stress sources',
                    'Practice time management',
                    'Take regular breaks',
                    'Consider stress counseling',
                    'Engage in stress-relief activities (yoga, meditation)'
                ]
            })
        elif scores['stress_score'] > 30:
            recommendations.append({
                'category': 'Stress',
                'priority': 'Moderate',
                'suggestions': [
                    'Practice stress management techniques',
                    'Ensure adequate sleep',
                    'Maintain work-life balance',
                    'Exercise regularly'
                ]
            })
        
        # General wellness
        if scores['wellness_score'] > 70:
            recommendations.append({
                'category': 'Wellness',
                'priority': 'Maintenance',
                'suggestions': [
                    'Continue current positive practices',
                    'Maintain social connections',
                    'Stay physically active',
                    'Practice gratitude'
                ]
            })
        
        return recommendations
    
    def create_mental_health_report(self, df, output_file='mental_health_report.txt'):
        """Generate comprehensive mental health report"""
        scores = self.calculate_mental_health_scores(df)
        interpretations = self.interpret_scores(scores)
        recommendations = self.generate_recommendations(scores, interpretations)
        
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MENTAL HEALTH ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration Analyzed: {df['timestamp'].max():.1f} seconds\n")
            f.write(f"Frames Analyzed: {len(df)}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("MENTAL HEALTH INDICATORS\n")
            f.write("-" * 70 + "\n\n")
            
            f.write(f"Depression Score:      {scores['depression_score']:.1f}/100  [{interpretations['depression']['level']}]\n")
            f.write(f"Anxiety Score:         {scores['anxiety_score']:.1f}/100  [{interpretations['anxiety']['level']}]\n")
            f.write(f"Stress Score:          {scores['stress_score']:.1f}/100  [{interpretations['stress']['level']}]\n")
            f.write(f"Emotional Stability:   {scores['emotional_stability']:.1f}/100\n")
            f.write(f"Overall Wellness:      {scores['wellness_score']:.1f}/100  [{interpretations['wellness']['level']}]\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("CLINICAL INTERPRETATION\n")
            f.write("-" * 70 + "\n\n")
            
            for key, interp in interpretations.items():
                f.write(f"{key.upper()}: {interp['level']}\n")
                f.write(f"  {interp['description']}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 70 + "\n\n")
            
            for rec in recommendations:
                f.write(f"{rec['category']} (Priority: {rec['priority']})\n")
                for suggestion in rec['suggestions']:
                    f.write(f"  • {suggestion}\n")
                f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("EMOTION DISTRIBUTION\n")
            f.write("-" * 70 + "\n\n")
            
            emotion_counts = df['dominant_emotion'].value_counts()
            for emotion, count in emotion_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"{emotion.capitalize():12} : {percentage:5.1f}% ({count} frames)\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("DISCLAIMER\n")
            f.write("=" * 70 + "\n\n")
            f.write("This analysis is based on facial emotion recognition and should NOT be\n")
            f.write("used as a diagnostic tool. It provides general indicators only.\n")
            f.write("Please consult a qualified mental health professional for proper\n")
            f.write("assessment and treatment.\n")
            f.write("=" * 70 + "\n")
        
        print(f"Mental health report saved to {output_file}")
        return scores, interpretations, recommendations
    
    def create_visualizations(self, df, scores, output_dir='results'):
        """Create mental health visualization charts"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Mental Health Scores Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Mental Health Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Depression, Anxiety, Stress scores
        indicators = ['Depression', 'Anxiety', 'Stress']
        indicator_scores = [scores['depression_score'], scores['anxiety_score'], scores['stress_score']]
        colors = ['#ff6b6b' if s > 60 else '#ffd93d' if s > 30 else '#6bcf7f' for s in indicator_scores]
        
        axes[0, 0].barh(indicators, indicator_scores, color=colors)
        axes[0, 0].set_xlim(0, 100)
        axes[0, 0].set_xlabel('Score (0-100)')
        axes[0, 0].set_title('Mental Health Indicators')
        axes[0, 0].axvline(30, color='green', linestyle='--', alpha=0.5)
        axes[0, 0].axvline(60, color='orange', linestyle='--', alpha=0.5)
        
        # Wellness gauge
        wellness_color = '#6bcf7f' if scores['wellness_score'] > 70 else '#ffd93d' if scores['wellness_score'] > 40 else '#ff6b6b'
        axes[0, 1].pie([scores['wellness_score'], 100-scores['wellness_score']], 
                       colors=[wellness_color, '#e0e0e0'],
                       startangle=90,
                       counterclock=False)
        axes[0, 1].text(0, 0, f"{scores['wellness_score']:.0f}%", 
                       ha='center', va='center', fontsize=30, fontweight='bold')
        axes[0, 1].set_title('Overall Wellness Score')
        
        # Emotion distribution
        emotion_counts = df['dominant_emotion'].value_counts()
        axes[1, 0].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Emotion Distribution')
        
        # Emotional stability timeline
        window_size = max(1, len(df) // 20)
        stability_timeline = []
        for i in range(0, len(df) - window_size):
            window = df['dominant_emotion'].iloc[i:i+window_size]
            changes = (window != window.shift()).sum()
            stability = 100 - (changes / len(window) * 100)
            stability_timeline.append(stability)
        
        if stability_timeline:
            axes[1, 1].plot(stability_timeline, color='#4a90e2', linewidth=2)
            axes[1, 1].fill_between(range(len(stability_timeline)), stability_timeline, alpha=0.3, color='#4a90e2')
            axes[1, 1].set_xlabel('Time Progress')
            axes[1, 1].set_ylabel('Stability %')
            axes[1, 1].set_title('Emotional Stability Over Time')
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].axhline(scores['emotional_stability'], color='red', linestyle='--', 
                              label=f'Average: {scores["emotional_stability"]:.1f}%')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mental_health_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Indicator Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Depression\nScore', 'Anxiety\nScore', 'Stress\nScore', 
                     'Emotional\nStability', 'Wellness\nScore']
        values = [scores['depression_score'], scores['anxiety_score'], 
                 scores['stress_score'], scores['emotional_stability'], 
                 scores['wellness_score']]
        
        bars = ax.bar(categories, values, 
                     color=['#ff6b6b', '#ff8c42', '#ffd93d', '#6bcf7f', '#4a90e2'])
        ax.set_ylim(0, 100)
        ax.set_ylabel('Score (0-100)', fontsize=12)
        ax.set_title('Mental Health Metrics Overview', fontsize=14, fontweight='bold')
        ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
        ax.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mental_health_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")


def main():
    """Main analysis function"""
    print("=" * 70)
    print("MENTAL HEALTH VIDEO ANALYSIS SYSTEM")
    print("=" * 70)
    print("\nNOTE: This tool provides general indicators based on facial emotions.")
    print("It is NOT a diagnostic tool. Consult a professional for proper assessment.\n")
    
    analyzer = MentalHealthAnalyzer()
    
    video_path = input("Enter path to video file: ").strip()
    
    print("\nAnalyzing video for mental health indicators...")
    df = analyzer.analyze_video_file(video_path, skip_frames=5)
    
    if len(df) == 0:
        print("No data collected. Please check video file.")
        return
    
    print("\nCalculating mental health scores...")
    scores, interpretations, recommendations = analyzer.create_mental_health_report(df)
    
    print("\nGenerating visualizations...")
    analyzer.create_visualizations(df, scores)
    
    # Save emotion data
    df.to_csv('emotion_data.csv', index=False)
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nDepression Score:    {scores['depression_score']:.1f}/100  [{interpretations['depression']['level']}]")
    print(f"Anxiety Score:       {scores['anxiety_score']:.1f}/100  [{interpretations['anxiety']['level']}]")
    print(f"Stress Score:        {scores['stress_score']:.1f}/100  [{interpretations['stress']['level']}]")
    print(f"Wellness Score:      {scores['wellness_score']:.1f}/100  [{interpretations['wellness']['level']}]")
    
    print("\n" + "-" * 70)
    print("FILES GENERATED:")
    print("-" * 70)
    print("✓ mental_health_report.txt")
    print("✓ emotion_data.csv")
    print("✓ results/mental_health_dashboard.png")
    print("✓ results/mental_health_metrics.png")
    print("\n" + "=" * 70)
    print("Analysis complete! Please review the generated report.")
    print("=" * 70)


if __name__ == "__main__":
    main()
