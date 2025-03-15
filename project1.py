import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show():
    """Main function to display Project 1 content"""
    
    # Project header
    st.markdown('<div class="main-header">Project 1: [Your Project Title]</div>', unsafe_allow_html=True)
    
    # Project description
    st.markdown(
        '<div class="project-description">'
        'This is where you will describe your first project, including the problem statement, '
        'methodology, key findings, and technologies used. Tailor this to showcase your skills '
        'relevant to finance, fintech, energy markets, or insurance domains.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Create tabs for organization
    tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Results & Insights"])
    
    with tab1:
        st.header("Project Overview")
        st.write("""
        Use this section to explain the project background, objectives, and relevance 
        to the industry you're targeting. Highlight the business problem you're solving.
        
        For example, if this is a financial market prediction dashboard:
        - What market data are you analyzing?
        - What prediction models are you implementing?
        - What metrics are you using to evaluate performance?
        """)
        
        # Example - Sample mockup of data
        st.subheader("Data Sample")
        # Replace with your actual data loading logic
        date_range = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
        sample_data = pd.DataFrame({
            'Date': date_range,
            'Value': np.random.normal(100, 15, 30).cumsum(),
            'Another_Metric': np.random.normal(50, 10, 30).cumsum(),
            'Category': np.random.choice(['A', 'B', 'C'], 30)
        })
        st.dataframe(sample_data)
    
    with tab2:
        st.header("Data Analysis")
        
        # Example - Interactive parameters
        st.subheader("Analysis Parameters")
        col1, col2 = st.columns(2)
        with col1:
            param1 = st.slider("Parameter 1", min_value=0, max_value=100, value=50)
        with col2:
            param2 = st.selectbox("Parameter 2", options=["Option A", "Option B", "Option C"])
        
        # Example - Sample visualization
        st.subheader("Data Visualization")
        fig = px.line(
            sample_data, 
            x='Date', 
            y=['Value', 'Another_Metric'],
            title=f"Sample Time Series (Parameter: {param1})"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualization
        fig2 = px.histogram(sample_data, x='Category', y='Value', color='Category',
                         title="Distribution by Category")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.header("Results & Insights")
        
        # Example - Key metrics display
        st.subheader("Key Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Metric 1", value="92.7%", delta="4.2%")
        with col2:
            st.metric(label="Metric 2", value="$1.24M", delta="-0.8%")
        with col3:
            st.metric(label="Metric 3", value="38.2", delta="12.5%")
        
        # Example - Conclusions
        st.subheader("Key Insights")
        st.write("""
        Here you would summarize your findings and their implications for business decisions.
        For example:
        
        1. Insight 1: Analysis shows a strong correlation between X and Y
        2. Insight 2: The model predicts a 23% increase in Z over the next quarter
        3. Insight 3: Risk factors A and B account for 78% of variability in outcomes
        """)
        
        # Example - Next steps or recommendations
        st.subheader("Recommendations")
        st.write("""
        Based on your analysis, what actions would you recommend? This demonstrates your 
        ability to translate data science insights into business value.
        """)
    
    # Final section - methodology details
    with st.expander("Technical Methodology"):
        st.write("""
        This expandable section can contain more technical details about your project,
        including:
        
        - Data collection methods and sources
        - Data preprocessing steps
        - Model selection and architecture
        - Training and evaluation procedures
        - Technical challenges and solutions
        
        This is a good place to demonstrate your technical depth without overwhelming
        non-technical readers.
        """)
        
        # Example - Model performance metrics
        st.code("""
# Sample code snippet showing model implementation
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Model training
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
        """)

# This function will be called from the main app
if __name__ == "__main__":
    show()