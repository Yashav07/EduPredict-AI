
import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
h1 {
    color: #173b70;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD AND TRAIN MACHINE LEARNING MODEL
# -----------------------------------


@st.cache_resource
def train_model():

    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as z:

        # Find the CSV even if it is inside a subfolder
        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith("student-mat.csv")
        ]

        if not csv_files:
            raise FileNotFoundError(
                "student-mat.csv was not found. "
                "Available files: " + str(z.namelist())
            )

        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f, sep=";")

    features = [
        "age",
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2",
        "sex",
        "internet"
    ]

    X = df[features]
    y = df["G3"]

    numeric_features = [
        "age", "studytime", "failures",
        "absences", "G1", "G2"
    ]

    categorical_features = ["sex", "internet"]

    preprocessor = ColumnTransformer([
        ("numbers", "passthrough", numeric_features),
        ("categories",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, r2, len(df)


    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        with z.open("student-mat.csv") as f:
            df = pd.read_csv(f, sep=";")

    features = [
        "age",
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2",
        "sex",
        "internet"
    ]

    X = df[features]
    y = df["G3"]

    numeric_features = [
        "age", "studytime", "failures",
        "absences", "G1", "G2"
    ]

    categorical_features = ["sex", "internet"]

    preprocessor = ColumnTransformer([
        ("numbers", "passthrough", numeric_features),
        ("categories", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, r2, len(df)


# -----------------------------------
# HEADER
# -----------------------------------

st.title("🎓 EduPredict AI")
st.subheader("AI-Based Student Performance Prediction System")

st.write(
    "Analyze academic performance, predict future grades "
    "and discover ways to improve learning."
)

st.divider()

# -----------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------

st.sidebar.title("🎓 EduPredict AI")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Performance Calculator",
        "AI Prediction",
        "Analytics",
        "About Project"
    ]
)

# -----------------------------------
# HOME PAGE
# -----------------------------------

if page == "Home":

    st.title("Welcome to EduPredict AI")

    st.write(
        "EduPredict AI is a machine-learning-based educational "
        "system designed to estimate student academic performance."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Main Features", "2")
    col2.metric("ML Algorithm", "Random Forest")
    col3.metric("Dataset", "UCI Student Data")

    st.divider()

    st.subheader("What can this website do?")

    st.markdown("""
    - Calculate expected student performance.
    - Predict final grades using Machine Learning.
    - Analyze study habits and attendance.
    - Display academic performance analytics.
    - Provide suggestions to improve academic results.
    """)

    st.info(
        "Choose a feature from the sidebar to get started."
    )

# -----------------------------------
# PERFORMANCE CALCULATOR
# -----------------------------------

elif page == "Performance Calculator":

    st.header("📊 Student Performance Calculator")

    st.write(
        "Enter academic details to calculate an estimated "
        "performance score out of 100."
    )

    col1, col2 = st.columns(2)

    with col1:
        student_name = st.text_input("Student Name")

        marks = st.slider(
            "Previous examination marks (%)",
            0, 100, 70
        )

        attendance = st.slider(
            "Attendance (%)",
            0, 100, 80
        )

    with col2:
        study_hours = st.slider(
            "Daily study hours",
            0.0, 12.0, 3.0, 0.5
        )

        assignment = st.slider(
            "Assignment performance (%)",
            0, 100, 75
        )

    if st.button("Calculate Performance", type="primary"):

        study_score = min(study_hours / 5, 1) * 100

        final_score = (
            marks * 0.35 +
            attendance * 0.25 +
            study_score * 0.20 +
            assignment * 0.20
        )

        st.divider()

        st.subheader(
            f"Performance Report: {student_name or 'Student'}"
        )

        st.metric(
            "Estimated Performance",
            f"{final_score:.2f}%"
        )

        st.progress(int(final_score))

        if final_score >= 80:
            st.success("Excellent Performance! 🌟")
        elif final_score >= 60:
            st.info("Good Performance! Keep improving.")
        elif final_score >= 40:
            st.warning("Average Performance. More practice is needed.")
        else:
            st.error("Needs Improvement. Follow a study plan.")

        st.caption(
            "This is a rule-based calculator, not the ML prediction. "
            "The weights are illustrative project assumptions."
        )

# -----------------------------------
# MACHINE LEARNING PREDICTION
# -----------------------------------

elif page == "AI Prediction":

    st.header("🤖 Machine Learning Prediction")

    st.write(
        "Enter student details to estimate the final Mathematics "
        "grade using a trained Random Forest model."
    )

    try:
        with st.spinner("Loading dataset and training AI model..."):
            model, mae, r2, total = train_model()

        st.success("Machine Learning model is ready!")

        with st.form("prediction_form"):

            st.subheader("Student Information")

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input(
                    "Student Age", 15, 22, 17
                )

                studytime = st.selectbox(
                    "Weekly Study Time",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {
                        1: "Less than 2 hours",
                        2: "2 to 5 hours",
                        3: "5 to 10 hours",
                        4: "More than 10 hours"
                    }[x]
                )

                failures = st.selectbox(
                    "Previous Class Failures",
                    [0, 1, 2, 3]
                )

                absences = st.number_input(
                    "Number of Absences",
                    0, 93, 2
                )

            with col2:
                g1 = st.slider(
                    "First Period Grade (0–20)",
                    0, 20, 12
                )

                g2 = st.slider(
                    "Second Period Grade (0–20)",
                    0, 20, 13
                )

                sex = st.selectbox(
                    "Sex (dataset category)",
                    ["F", "M"]
                )

                internet = st.selectbox(
                    "Internet Access at Home",
                    ["yes", "no"]
                )

            submitted = st.form_submit_button(
                "Predict Final Performance",
                type="primary"
            )

        if submitted:

            input_data = pd.DataFrame([{
                "age": age,
                "studytime": studytime,
                "failures": failures,
                "absences": absences,
                "G1": g1,
                "G2": g2,
                "sex": sex,
                "internet": internet
            }])

            prediction = float(model.predict(input_data)[0])
            prediction = np.clip(prediction, 0, 20)

            percentage = prediction / 20 * 100

            st.divider()
            st.subheader("AI Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Final Grade",
                f"{prediction:.2f} / 20"
            )

            col2.metric(
                "Equivalent Percentage",
                f"{percentage:.1f}%"
            )

            st.progress(int(percentage))

            if prediction >= 15:
                st.success("High predicted performance 🌟")
            elif prediction >= 10:
                st.info("Passing-range predicted performance.")
            else:
                st.warning(
                    "At-risk predicted performance. "
                    "Additional academic support may help."
                )

            st.subheader("AI Recommendations")

            if studytime <= 2:
                st.write(
                    "📚 Increase weekly study time and follow "
                    "a consistent study timetable."
                )

            if failures > 0:
                st.write(
                    "🎯 Focus on subjects where you have "
                    "previously experienced difficulties."
                )

            if absences > 10:
                st.write(
                    "🏫 Improve class attendance and avoid "
                    "unnecessary absences."
                )

            if g2 < 10:
                st.write(
                    "✍️ Revise important topics and practise "
                    "previous examination questions."
                )

            if studytime > 2 and failures == 0 and absences <= 10 and g2 >= 10:
                st.write(
                    "🌟 Maintain your current learning routine "
                    "and continue practising."
                )

            st.caption(
                "Predictions are estimates based on historical "
                "Portuguese secondary-school Mathematics data. "
                "They are not guaranteed examination results."
            )

        with st.expander("Model evaluation"):
            st.write(f"Training dataset records: {total}")
            st.write(f"Test-set Mean Absolute Error: {mae:.2f}")
            st.write(f"Test-set R² score: {r2:.3f}")

    except Exception as e:
        st.error("Unable to load or train the model.")
        st.write(str(e))
        st.info(
            "Check your internet connection and try again."
        )

# -----------------------------------
# ANALYTICS PAGE
# -----------------------------------

elif page == "Analytics":

    st.header("📈 Student Performance Analytics")

    st.write(
        "Explore an example of academic performance "
        "and the factors that influence it."
    )

    sample = pd.DataFrame({
        "Category": [
            "Examination Marks",
            "Attendance",
            "Study Score",
            "Assignments"
        ],
        "Score": [75, 85, 65, 80]
    })

    st.subheader("Example Performance Overview")

    st.bar_chart(
        sample.set_index("Category")
    )

    st.subheader("Performance Distribution")

    distribution = pd.DataFrame({
        "Performance": [
            "Excellent",
            "Good",
            "Average",
            "Needs Improvement"
        ],
        "Example Students": [25, 35, 25, 15]
    })

    st.bar_chart(
        distribution.set_index("Performance")
    )

    st.caption(
        "These charts use illustrative sample values, "
        "not live student records."
    )

# -----------------------------------
# ABOUT PAGE
# -----------------------------------

elif page == "About Project":

    st.header("About EduPredict AI")

    st.write("""
    EduPredict AI is an educational technology project
    that combines rule-based calculations and Machine Learning
    to estimate student academic performance.

    **Technologies Used**

    - Python
    - Streamlit
    - Pandas and NumPy
    - Scikit-learn
    - Random Forest Regression
    - UCI Student Performance Dataset

    **Machine Learning Process**

    1. Collect historical student data.
    2. Select academic input features.
    3. Preprocess numerical and categorical data.
    4. Divide the dataset into training and testing sets.
    5. Train a Random Forest Regression model.
    6. Evaluate the model.
    7. Predict final grades for new inputs.

    **Project Objective**

    To demonstrate how Artificial Intelligence and
    Machine Learning can be applied to educational
    performance analysis.
    """)

    st.info(
        "This is an educational demonstration, not an "
        "official student grading or admission system."
    )

# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.caption(
    "EduPredict AI | Student Performance Prediction System | "
    "Developed using Python and Machine Learning"
)
