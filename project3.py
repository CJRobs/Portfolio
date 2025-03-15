import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def show():
    """Main function to display Project 3 content"""
    
    # Project header
    st.markdown('<div class="main-header">Project 3: [Your Project Title]</div>', unsafe_allow_html=True)
    
    # Project description
    st.markdown(
        '<div class="project-description">'
        'This is where you will describe your third project. For a well-rounded portfolio, '
        'consider making this project showcase different skills from the previous two. '
        'For example, if the others focused on structured data, this one could involve NLP, '
        'image processing, or deep learning.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar options for this project
    st.sidebar.markdown("### Project Controls")
    time_period = st.sidebar.selectbox(
        "Select Time Period",
        ["Last 30 Days", "Last Quarter", "Last Year", "All Time"]
    )
    
    # Create tabs for organization
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Analysis", "Technical Details"])
    
    with tab1:
        st.header("Interactive Dashboard")
        
        # Project metrics overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Key Metric 1", value="$3.45M", delta="12.3%")
        with col2:
            st.metric(label="Key Metric 2", value="87.2%", delta="-2.1%")
        with col3:
            st.metric(label="Key Metric 3", value="1,243", delta="5.6%")
        with col4:
            st.metric(label="Key Metric 4", value="0.0043", delta="0.0012")
        
        # Example - Sample time series data
        st.subheader("Time Series Analysis")
        
        # Generate sample data based on selected time period
        days = 30
        if time_period == "Last Quarter":
            days = 90
        elif time_period == "Last Year":
            days = 365
        elif time_period == "All Time":
            days = 1000
        
        date_range = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Create some fake time series data
        np.random.seed(42)
        ts_data = pd.DataFrame({
            'date': date_range,
            'value1': np.cumsum(np.random.normal(0.1, 1, days)),
            'value2': np.cumsum(np.random.normal(0.2, 1.2, days)),
            'value3': np.cumsum(np.random.normal(0.05, 0.8, days))
        })
        
        # Allow user to select metrics to display
        metrics = st.multiselect(
            "Select Metrics to Display",
            options=['value1', 'value2', 'value3'],
            default=['value1', 'value2']
        )
        
        if metrics:
            fig = px.line(
                ts_data, 
                x='date', 
                y=metrics,
                title=f"Performance Metrics Over {time_period}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one metric to display")
        
        # Example - Interactive map visualization
        st.subheader("Geographical Analysis")
        
        # Create sample geographical data
        geo_data = pd.DataFrame({
            'state': ['New York', 'California', 'Texas', 'Florida', 'Illinois', 
                     'Pennsylvania', 'Ohio', 'Georgia', 'North Carolina', 'Michigan'],
            'value': np.random.uniform(10, 100, 10),
            'growth': np.random.uniform(-20, 30, 10)
        })
        
        fig = px.choropleth(
            geo_data,
            locations='state',
            locationmode="USA-states",
            color='value',
            scope="usa",
            color_continuous_scale="Viridis",
            hover_data=['growth'],
            title="Regional Performance"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("In-Depth Analysis")
        
        # Example - Sample multivariate data
        n_samples = 500
        analysis_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(3, 2, n_samples),
            'feature3': np.random.exponential(2, n_samples),
            'feature4': np.random.uniform(-1, 1, n_samples),
            'target': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.6, 0.3, 0.1])
        })
        
        # Interactive data exploration
        st.subheader("Data Explorer")
        
        # Feature selection
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis Feature", options=analysis_data.columns[:-1])
        with col2:
            y_feature = st.selectbox("Y-axis Feature", options=analysis_data.columns[:-1],
                                    index=1)  # Default to second feature
        
        # Create scatter plot with the selected features
        fig = px.scatter(
            analysis_data,
            x=x_feature,
            y=y_feature,
            color='target',
            title=f"{y_feature} vs {x_feature} by Target Class",
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution by target class
        st.subheader("Feature Distributions by Class")
        
        feature_to_analyze = st.selectbox(
            "Select Feature to Analyze",
            options=analysis_data.columns[:-1]
        )
        
        fig = px.histogram(
            analysis_data,
            x=feature_to_analyze,
            color='target',
            marginal='box',
            barmode='overlay',
            opacity=0.7,
            title=f"Distribution of {feature_to_analyze} by Target Class"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        
        # Calculate correlation matrix
        corr = analysis_data.drop(columns=['target']).corr()
        
        # Create heatmap
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Technical Implementation")
        
        # Information about the technical approach
        st.write("""
        This section can provide more detailed information about the technical implementation
        of your project, including:
        
        - Data sources and collection methodology
        - Data preprocessing and feature engineering steps
        - Model architecture and design decisions
        - Implementation challenges and solutions
        - Technical performance metrics
        """)
        
        # Model architecture visualization (placeholder)
        st.subheader("Model Architecture")
        
        # Simple diagram of model architecture
        architecture_data = {
            'Layer': ['Input', 'Hidden 1', 'Hidden 2', 'Output'],
            'Nodes': [10, 64, 32, 3],
            'Activation': ['None', 'ReLU', 'ReLU', 'Softmax']
        }
        
        st.table(architecture_data)
        
        # Example code snippet
        st.subheader("Sample Implementation Code")
        
        st.code("""
# Example code for model implementation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model(input_dim, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train model
model = create_model(input_dim=10, num_classes=3)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)
        """)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        # Sample performance data
        perf_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Training': [0.92, 0.89, 0.88, 0.885],
            'Validation': [0.87, 0.84, 0.83, 0.835],
            'Test': [0.86, 0.83, 0.82, 0.825]
        })
        
        # Plot performance metrics
        fig = px.bar(
            perf_data, 
            x='Metric', 
            y=['Training', 'Validation', 'Test'],
            barmode='group',
            title='Model Performance Metrics'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Discussion of results
        st.subheader("Technical Insights")
        st.write("""
        This is where you can discuss the technical insights gained from your project, such as:
        
        1. How different model architectures performed
        2. Interesting feature interactions discovered
        3. Technical limitations and future improvements
        4. Trade-offs between different approaches
        5. Lessons learned from the implementation
        """)

# This function will be called from the main app
if __name__ == "__main__":
    show()