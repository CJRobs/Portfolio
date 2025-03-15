import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def show():
    """Main function to display Project 2 content"""
    
    # Project header
    st.markdown('<div class="main-header">Project 2: [Your Project Title]</div>', unsafe_allow_html=True)
    
    # Project description
    st.markdown(
        '<div class="project-description">'
        'This is where you will describe your second project. Consider making this project '
        'different from the first to showcase breadth of skills. For example, if Project 1 '
        'focused on prediction, this one could focus on classification or clustering.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Create tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Exploratory Analysis", "Models", "Insights"])
    
    with tab1:
        st.header("Project Overview")
        
        # Project context and objectives
        st.subheader("Business Context")
        st.write("""
        Explain the business context and problem you're addressing. For example, if this is an 
        insurance risk assessment tool:
        - What risk factors are you analyzing?
        - What business decisions will your analysis inform?
        - What is the potential impact of your findings?
        """)
        
        # Data overview
        st.subheader("Data & Methodology")
        st.write("""
        Describe your data sources and methodology at a high level. Include:
        - Data sources and time periods
        - Key variables
        - Analysis approach
        """)
        
        # Example - Infographic or process flow
        st.subheader("Project Workflow")
        
        # Simple process flow chart using Streamlit
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown("##### 1. Data Collection")
        col1.markdown("• Source 1\n• Source 2\n• Source 3")
        
        col2.markdown("##### 2. Preprocessing")
        col2.markdown("• Cleaning\n• Feature Engineering\n• Transformation")
        
        col3.markdown("##### 3. Modeling")
        col3.markdown("• Algorithm 1\n• Algorithm 2\n• Ensemble")
        
        col4.markdown("##### 4. Deployment")
        col4.markdown("• Interactive Dashboard\n• API Integration\n• Monitoring")
    
    with tab2:
        st.header("Exploratory Data Analysis")
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Example - Insurance claims data
        data = pd.DataFrame({
            'age': np.random.normal(45, 15, n_samples).clip(18, 85),
            'policy_tenure': np.random.gamma(shape=2, scale=3, size=n_samples).clip(0, 20),
            'claim_amount': np.random.exponential(scale=5000, size=n_samples),
            'risk_score': np.random.normal(50, 20, n_samples).clip(0, 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'has_claimed': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        # Interactive filtering
        st.subheader("Data Explorer")
        col1, col2 = st.columns(2)
        with col1:
            age_range = st.slider("Age Range", int(data['age'].min()), int(data['age'].max()), 
                                 (int(data['age'].min()), int(data['age'].max())))
        with col2:
            regions = st.multiselect("Regions", options=data['region'].unique(), 
                                    default=data['region'].unique())
        
        # Filter data based on selections
        filtered_data = data[(data['age'] >= age_range[0]) & 
                            (data['age'] <= age_range[1]) & 
                            (data['region'].isin(regions))]
        
        # Display filtered dataset
        st.dataframe(filtered_data.head(10))
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        # Distribution plots
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(filtered_data, x='age', color='has_claimed',
                             marginal='box', title='Age Distribution by Claim Status')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(filtered_data, x='risk_score', y='claim_amount', color='region',
                           size='policy_tenure', hover_data=['age'],
                           title='Risk Score vs Claim Amount')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        corr = filtered_data.drop(columns=['region']).corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.header("Models & Predictions")
        
        # Model selection
        st.subheader("Model Selection")
        model_type = st.selectbox("Select Model Type", 
                                 ["Logistic Regression", "Random Forest", "Gradient Boosting", "Neural Network"])
        
        # Model parameters
        st.subheader("Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            if model_type == "Logistic Regression":
                st.slider("Regularization Strength", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
            elif model_type in ["Random Forest", "Gradient Boosting"]:
                st.slider("Number of Estimators", min_value=10, max_value=500, value=100, step=10)
                st.slider("Max Depth", min_value=2, max_value=30, value=10)
            else:  # Neural Network
                st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
                st.slider("Hidden Layers", min_value=1, max_value=5, value=2)
        
        with col2:
            # Feature importance placeholder
            st.subheader("Feature Importance")
            features = ['age', 'policy_tenure', 'risk_score', 'region_encoded']
            importance = np.array([0.3, 0.25, 0.35, 0.1])
            fig = px.bar(x=importance, y=features, orientation='h',
                       title="Feature Importance", labels={"x": "Importance", "y": "Features"})
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance metrics
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", "87.3%", "2.1%")
        col2.metric("Precision", "83.5%", "1.7%")
        col3.metric("Recall", "79.8%", "-0.5%")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        conf_matrix = np.array([[800, 50], [100, 150]])
        fig = px.imshow(conf_matrix, 
                       labels=dict(x="Predicted", y="Actual"),
                       x=['Negative', 'Positive'],
                       y=['Negative', 'Positive'],
                       text_auto=True,
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab4:
        st.header("Business Insights & Recommendations")
        
        # Key findings
        st.subheader("Key Findings")
        st.write("""
        Summarize 3-5 key findings from your analysis. Be specific and quantitative.
        For example:
        
        1. High-risk customers (risk score > 75) are 3.2x more likely to file a claim within the first year
        2. Policy tenure has a non-linear relationship with claim frequency, with a significant drop after 3 years
        3. The East region shows anomalous claim patterns that warrant further investigation
        """)
        
        # Business impact
        st.subheader("Business Impact")
        
        # Example - Business impact visualization
        impact_data = pd.DataFrame({
            'Scenario': ['Current', 'Optimized'],
            'Revenue': [1000000, 1120000],
            'Costs': [800000, 780000],
            'Profit': [200000, 340000]
        })
        
        fig = go.Figure()
        for column in ['Revenue', 'Costs', 'Profit']:
            fig.add_trace(go.Bar(
                x=impact_data['Scenario'],
                y=impact_data[column],
                name=column
            ))
        
        fig.update_layout(title='Financial Impact of Model Implementation',
                         barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Recommendations")
        st.write("""
        Based on the analysis, what actions do you recommend? For example:
        
        1. Implement a targeted retention program for high-tenure, low-risk customers
        2. Adjust pricing strategy for specific age groups in the East region
        3. Develop an early warning system for high-risk customers showing behavioral changes
        4. Allocate additional resources to investigate anomalous patterns in the East region
        """)
        
        # Next steps
        st.subheader("Next Steps")
        st.write("""
        What additional work would enhance this analysis? For example:
        
        1. Integrate external data sources (e.g., economic indicators, weather patterns)
        2. Develop a real-time monitoring system for risk factors
        3. Conduct A/B testing of different intervention strategies
        4. Expand analysis to include additional product lines
        """)

# This function will be called from the main app
if __name__ == "__main__":
    show()