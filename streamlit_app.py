import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from nutriscore import NutriScoreCalculator

# Page config
st.set_page_config(
    page_title="Nutri-Score Calculator",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: scale(1.02);
    }
    .nutriscore-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .grade-A { background: linear-gradient(135deg, #038141 0%, #05a854 100%); }
    .grade-B { background: linear-gradient(135deg, #85BB2F 0%, #9ed145 100%); }
    .grade-C { background: linear-gradient(135deg, #FECB02 0%, #ffd633 100%); }
    .grade-D { background: linear-gradient(135deg, #EE8100 0%, #ff9419 100%); }
    .grade-E { background: linear-gradient(135deg, #E63E11 0%, #ff5722 100%); }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }

    .header-text {
        font-size: 42px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 10px;
    }

    .subheader-text {
        font-size: 18px;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-text">Nutri-Score Calculator</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">Calculate the nutritional quality of food products</div>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About Nutri-Score")
    st.markdown("""
    The **Nutri-Score** is a nutrition label that converts the nutritional value of products into a simple code consisting of **5 letters** (A to E), each with its own color.

    ### How it works:
    - **A (Dark Green)**: Best nutritional quality
    - **B (Light Green)**: Good quality
    - **C (Yellow)**: Average quality
    - **D (Orange)**: Poor quality
    - **E (Red)**: Worst quality

    ### Calculation:
    - **Negative Points (N)**: Energy, sugars, saturated fat, salt
    - **Positive Points (P)**: Proteins, fiber, fruits/vegetables
    - **Score = N - P**

    Lower scores = Better nutrition
    """)

    st.divider()

    st.header("Example Products")
    st.markdown("""
    **Yogurt (Grade A)**
    - Energy: 234 kJ
    - Sugars: 5.0 g
    - Salt: 0.13 g

    **Chocolate (Grade D)**
    - Energy: 2224 kJ
    - Sugars: 51.0 g
    - Salt: 0.38 g
    """)

# Main content in two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Input Nutritional Values (per 100g/100ml)")

    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🔴 Nutrients to Limit**")

        energy_kj = st.number_input(
            "Energy (kJ) *",
            min_value=0.0,
            max_value=5000.0,
            value=1500.0,
            step=10.0,
            help="Energy content in kilojoules"
        )

        saturated_fat = st.number_input(
            "Saturated Fat (g) *",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="Saturated fatty acids content"
        )

        sugars = st.number_input(
            "Sugars (g) *",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1,
            help="Simple sugars content"
        )

        salt = st.number_input(
            "Salt (g) *",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="Salt (sodium chloride) content"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🟢 Nutrients to Favor**")

        proteins = st.number_input(
            "Proteins (g) *",
            min_value=0.0,
            max_value=100.0,
            value=8.0,
            step=0.1,
            help="Protein content"
        )

        fiber = st.number_input(
            "Fiber (g) *",
            min_value=0.0,
            max_value=100.0,
            value=3.0,
            step=0.1,
            help="Dietary fiber content"
        )

        fruit_veg = st.number_input(
            "Fruits/Vegetables/Nuts (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            help="Percentage of fruits, vegetables, legumes and nuts"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Calculate button
    calculate_btn = st.button("Calculate Nutri-Score", type="primary", use_container_width=True)

with col2:
    st.markdown("###  Results")

    if calculate_btn or st.session_state.get('auto_calculate', False):
        # Calculate Nutri-Score
        result = NutriScoreCalculator.calculate_nutriscore(
            energy_kj=energy_kj,
            saturated_fat_g=saturated_fat,
            sugars_g=sugars,
            salt_g=salt,
            proteins_g=proteins,
            fiber_g=fiber,
            fruit_veg_pct=fruit_veg
        )

        # Display Nutri-Score grade with color
        grade = result['class']
        grade_class = f"grade-{grade}"

        st.markdown(f"""
        <div class="nutriscore-box {grade_class}">
            Nutri-Score: {grade}
        </div>
        """, unsafe_allow_html=True)

        # Display visual Nutri-Score bar
        st.markdown("#### Nutri-Score Scale")

        # Create Plotly figure for visual scale
        grades = ['A', 'B', 'C', 'D', 'E']
        colors = {
            'A': '#038141',
            'B': '#85BB2F',
            'C': '#FECB02',
            'D': '#EE8100',
            'E': '#E63E11'
        }

        fig = go.Figure()

        for i, g in enumerate(grades):
            # Larger bar for selected grade
            height = 1.5 if g == grade else 1.0
            y_pos = 0.5 if g == grade else 0.25

            fig.add_trace(go.Bar(
                x=[1],
                y=[height],
                name=g,
                marker_color=colors[g],
                text=g,
                textposition='inside',
                textfont=dict(size=24 if g == grade else 18, color='white', family='Arial Black'),
                hovertemplate=f'<b>{g}</b><extra></extra>',
                showlegend=False
            ))

        fig.update_layout(
            barmode='group',
            height=150,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display breakdown in metrics
        st.markdown("#### Score Breakdown")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                label="Negative Points (N)",
                value=result['negative_points'],
                delta=None,
                help="Points from nutrients to limit"
            )

        with metric_col2:
            st.metric(
                label="Positive Points (P)",
                value=result['positive_points'],
                delta=None,
                help="Points from nutrients to favor"
            )

        with metric_col3:
            st.metric(
                label="Final Score",
                value=result['score'],
                delta=None,
                help="Score = N - P (lower is better)"
            )

        # Interpretation
        st.markdown("#### Interpretation")

        interpretations = {
            'A': "**Excellent nutritional quality!** This product is among the best choices for your health.",
            'B': "**Good nutritional quality.** A solid choice for a balanced diet.",
            'C': "**Average nutritional quality.** Consume in moderation as part of a varied diet.",
            'D': "**Poor nutritional quality.** Limit consumption of this product.",
            'E': "**Very poor nutritional quality.** Avoid regular consumption."
        }

        st.info(interpretations[grade])

        # Display calculation details
        with st.expander(" View Calculation Details"):
            st.markdown("##### Negative Component (N)")
            st.markdown(f"""
            - Energy: {energy_kj} kJ → Points: {NutriScoreCalculator._get_points_from_thresholds(energy_kj, NutriScoreCalculator.ENERGY_THRESHOLDS)}
            - Saturated Fat: {saturated_fat} g → Points: {NutriScoreCalculator._get_points_from_thresholds(saturated_fat, NutriScoreCalculator.SATURATED_FAT_THRESHOLDS)}
            - Sugars: {sugars} g → Points: {NutriScoreCalculator._get_points_from_thresholds(sugars, NutriScoreCalculator.SUGARS_THRESHOLDS)}
            - Salt: {salt} g → Points: {NutriScoreCalculator._get_points_from_thresholds(salt, NutriScoreCalculator.SALT_THRESHOLDS)}
            - **Total N: {result['negative_points']}**
            """)

            st.markdown("##### Positive Component (P)")
            protein_note = " *(not counted due to N≥11 and fruit/veg≤80%)*" if (result['negative_points'] >= 11 and fruit_veg <= 80) else ""
            st.markdown(f"""
            - Proteins: {proteins} g{protein_note}
            - Fiber: {fiber} g
            - Fruits/Vegetables: {fruit_veg}%
            - **Total P: {result['positive_points']}**
            """)

            st.markdown(f"##### Final Calculation")
            st.markdown(f"**Score = {result['negative_points']} - {result['positive_points']} = {result['score']}**")
            st.markdown(f"**Grade: {grade}** (Score range: {NutriScoreCalculator.CLASS_THRESHOLDS[grade][0]} to {NutriScoreCalculator.CLASS_THRESHOLDS[grade][1]})")

    else:
        st.info(" Enter nutritional values on the left and click 'Calculate' to see the Nutri-Score!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p><strong>Nutri-Score Calculator</strong> | Decision Modeling Project 2025-2026</p>
    <p>Based on the official French Nutri-Score algorithm (ANSES methodology)</p>
    <p style='font-size: 12px;'>⚠️ This calculator is for educational purposes. Always refer to official product labels.</p>
</div>
""", unsafe_allow_html=True)
