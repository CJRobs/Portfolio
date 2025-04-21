import streamlit as st
from PIL import Image
import pandas as pd
import base64

def show():
    """Main function to display About Me content"""
    
    # Add custom CSS for styling
    apply_custom_css()
    
    # Apply page configuration and styling
    display_header()
    
    # Profile section
    display_profile_section()
    
    st.markdown("---")  # Line between sections
    
    # Professional Experience section (new)
    display_professional_experience()
    
    st.markdown("---")  # Line between sections
    
    # Skills section
    display_skills_section()
    
    st.markdown("---")  # Line between sections
    
    # Experience section
    display_experience_section()
    
    st.markdown("---")  # Line between sections
    
    # Education section
    display_education_section()
    
    st.markdown("---")  # Line between sections
    
    # Projects highlight section
    display_projects_section()
    
    st.markdown("---")  # Line between sections
    
    # Call to action
    display_call_to_action()

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    .edu-header, .exp-header {
        margin-bottom: 0;
        color: #4da6ff;
    }
    .edu-institution, .exp-company {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.2rem;
        color: white;
    }
    .edu-degree, .exp-position {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4da6ff;
        margin-bottom: 0.2rem;
    }
    .edu-date, .exp-date {
        font-style: italic;
        color: #cccccc;
        margin-bottom: 0.2rem;
    }
    .edu-grade, .exp-details {
        font-weight: 500;
        margin-top: 0.5rem;
        color: white;
    }
    .section-container {
        margin-bottom: 10px;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .pillar-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #4da6ff;
        margin-bottom: 0.5rem;
    }
    .pillar-emoji {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    /* Override Streamlit's default text colors */
    p, li, ol, ul, a {
        color: white !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    a {
        color: #4da6ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the page header"""
    st.markdown('<div class="main-header">About Me: Cameron Roberts</div>', unsafe_allow_html=True)

def display_profile_section():
    """Display profile information"""
    col1, col2, col3 = st.columns([1, 2, 3])
    
    with col1:
        st.image("stock_images/Cameron Roberts.jpg", width=260)        
        
    with col2:            
        st.subheader("Contact")
        
        # Contact links
        st.markdown("""
        * 📧 [thecjrobs@gmail.com](mailto:thecjrobs@gmail.com)
        * 🔗 [LinkedIn](https://www.linkedin.com/in/cjrobs/)
        * 💻 [GitHub](https://github.com/CJRobs)
        """)
        
        # Resume download button
        create_resume_download_button()
        
    with col3:
        st.markdown("""
        I'm a Lead Data Analyst at an international data and analytics consultancy, specializing in market intelligence and customer insight.

        While I’ve partnered with clients across a range of industries, my passion lies in applying data analytics and data science to solve complex problems in the financial markets. I'm currently seeking opportunities to transition full-time into this sector, where I can bring rigorous analytical thinking and creative modeling to high-impact financial challenges.

        Projects involving financial data and market behavior are where I thrive. I'm drawn to their complexity, their impact, and the opportunity to bring structure and signal to noisy, dynamic systems.

        Whether it's building predictive models, running simulations and forecasts, or applying algorithms to evaluate performance, the finance industry offers the perfect blend of technical depth and strategic value and that's exactly where I want to grow next.
        """)

def display_professional_experience():
    """Display Professional Experience section with Two Core Pillars"""
    st.header("Professional Experience")
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            '<div class="pillar-title"><span class="pillar-emoji">🚀</span>Product Development & Innovation</div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        I’ve led the end-to-end development of internal analytics products that combine statistical rigor with practical utility. These tools have enhanced decision-making and scaled our ability to deliver advanced analytics across clients:

        - **Geo Experiment Toolkit** – A causal inference platform for measuring the effectiveness of marketing strategies across geographic regions using test-and-control frameworks.
        
        - **Forecasting Engine** – Processes large-scale datasets, extracts relevant features, and generates interpretable forecasts to support strategic planning and investment decisions.
        
        - **Curve Optimizer** – Leverages differential evolution to simulate and recommend optimal strategies across business constraints and objectives.
        
        - **Customer Segmentation Tool** – Enables precise targeting by identifying high-value customer groups using unsupervised learning and clustering techniques.
        """)
        
    with col2:
        st.markdown(
            '<div class="pillar-title"><span class="pillar-emoji">📊</span>Bespoke Analytics & Client Delivery</div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        I specialize in transforming open-ended client problems into actionable insights. My approach blends stakeholder management, strategic thinking, and analytical rigor to deliver tailored, data-driven solutions that drive real impact.

        - **Royal Bank of Canada** – Following the merger between RBC and Brewin Dolphin, we led a customer analytics consolidation to ensure consistent targeting of high-net-worth (HNW) individuals. Using our consumer segmentation tool, we discovered this group was highly engaged with podcasts—prompting a strategic pivot in media planning.
        
        - **Clearscore** – Supported Clearscore’s expansion into the Australian market by designing and executing A/B tests to identify and implement highly effective user acquisition strategies.
        
        - **Ruggable** – Developed robust econometric models leveraging our in-house forecasting and optimization tools. These insights supported Ruggable’s PE investors with demand forecasts and helped build the investment case for accelerating top-line growth.
        """)


def create_resume_download_button():
    """Create a download button for the resume"""
    with open("stock_images/Cameron-Roberts-Resume-Two-Pages.pdf", "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
    
    st.download_button(
        label="📄 Download Resume",
        data=pdf_bytes,
        file_name="Cameron-Roberts-Resume.pdf",
        mime="application/pdf"
    )

def display_skills_section():
    """Display skills section"""
    st.header("Skills")
    
    col1, col2, col3 = st.columns(3)
    
    skill_categories = {
        "Languages & Packages": [
            "Python (pandas, numpy)",
            "R (dplyr, tidyrm, ggplot2, shiny)",
            "SQL & Database Management (Big Query & ERD diagrams)",
            "Cypher (Graph Query Language)",
            "Data Storage (JSON & Parquets)",
            "Javascript, HTML & CSS",
            "Haskell (Functional Programming)"
        ],
        "Domain Expertise": [
            "Geo Experiments, A/B & Incrementally Tests",
            "Model Forecasting",
            "Customer Segmentation",
            "Bayesian Time Series Regression Modelling",
            "Optimisations",
            "Operational Efficiency & Automation",
            "Knowledge Graphs"
        ],
        "Tools & ": [
            "Jupyter Notebooks, VS Code, R Studio",
            "Git & GitHub",
            "Neo4j & Langchain",
            "Streamlit, Dash, Flask, Shiny",
            "PowerBI",
            "PostgreSQL",
            "Scikit-learn, PyTorch, SciPy"
        ]
    }
    
    columns = [col1, col2, col3]
    for i, (category, skills) in enumerate(skill_categories.items()):
        with columns[i]:
            st.subheader(category)
            skill_list = "\n".join([f"* {skill}" for skill in skills])
            st.markdown(skill_list)

def display_experience_section():
    """Display work experience section with better styling"""
    st.header("Employment History")
    
    experiences = [
        {
            "company": "Independent Marketing Sciences",
            "company_url": "https://im-sciences.com/",
            "position": "Lead Data Analyst",
            "period": "01 June 2023 - Present",
        },
        {
            "company": "Omnicom Group",
            "company_url": "https://www.omnicomgroup.com/",
            "position": "Data Analyst",
            "period": "01 February 2022 - 31 May 2023",
        }
    ]
    
    # Create columns for side-by-side display
    cols = st.columns(len(experiences))
    
    # Display each experience in its own column with enhanced styling
    for i, exp in enumerate(experiences):
        with cols[i]:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            
            if "company_url" in exp:
                st.markdown(f'<div class="exp-company"><a href="{exp["company_url"]}">{exp["company"]}</a></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="exp-company">{exp["company"]}</div>', unsafe_allow_html=True)
                
            st.markdown(f'<div class="exp-position">{exp["position"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="exp-date">{exp["period"]}</div>', unsafe_allow_html=True)
                            
            st.markdown('</div>', unsafe_allow_html=True)

def display_education_section():
    """Display education section with better styling"""
    st.header("Education")
    
    education = [
        {
            "institution": "University of Bath",
            "institution_url": "https://www.bath.ac.uk/",
            "degree": "MSc Computer Science",
            "period": "2020 - 2021",
            "grade": "Distinction"
        },
        {
            "institution": "University of Reading",
            "institution_url": "https://www.reading.ac.uk/",
            "degree": "BSc Economics",
            "period": "2017 - 2020",
            "grade": "First Class Honors"
        }
    ]
    
    # Create columns for side-by-side display
    cols = st.columns(len(education))
    
    # Display each education entry in its own column with enhanced styling
    for i, edu in enumerate(education):
        with cols[i]:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            
            if "institution_url" in edu:
                st.markdown(f'<div class="edu-institution"><a href="{edu["institution_url"]}">{edu["institution"]}</a></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="edu-institution">{edu["institution"]}</div>', unsafe_allow_html=True)
                
            st.markdown(f'<div class="edu-degree">{edu["degree"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="edu-date">{edu["period"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="edu-grade">Grade: {edu["grade"]}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

def display_projects_section():
    """Display projects section"""
    st.header("Project Highlights")
    
    projects = [
        {
            "title": "Currency Exchange Network",
            "description": "Detecting Arbitrage Opportunities through Graph-Based Optimization"
        },
        {
            "title": "Equity Research",
            "description": "Applied fundamental equity valuation methodologies—including DCF, comparables, and multiples—to analyze and assess the intrinsic value of BP, ASML, and Warner Bros. Discovery."
        },
        {
            "title": "Customer Segmentation", 
            "description": "Performed customer segmentation on a Kaggle dataset using clustering techniques; delivered actionable segment insights to inform targeted strategies"
        }
    ]
    
    st.markdown("Explore my projects using the navigation menu on the left. Here's a quick overview:")
    
    for project in projects:
        st.markdown(f"**{project['title']}:** {project['description']}")
        
def display_call_to_action():
    """Display call to action section"""
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #3a3a3a; border-radius: 10px; color: white;">
    <h3 style="color: white;">Thanks so much for taking the time to read through my profile and projects</h3>
    <p>I'm always open to discussing new projects, opportunities, and ideas.</p>
    <p>📧 <a href="mailto:thecjrobs@gmail.com" style="color: #4da6ff;">thecjrobs@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)

# This function will be called from the main app
if __name__ == "__main__":
    show()