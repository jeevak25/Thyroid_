import pandas as pd
import streamlit as st
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)



# Define diagnosis mapping dictionary
diagnoses = {
    0: 'Negative',
    1: 'Hypothyroid',
    2: 'Hyperthyroid'
}
# Predicted diagnosis color
diagnosis_color = '#F63366'
title_color = '#F63366'  # Title color
title_css = f"<h1 style='text-align: center; color: {title_color};'>Thyroid Diagnosis Predictor</h1>"

# Detect button color
detect_button_color = '#F63366'

# Function to preprocess inputs before prediction
def preprocess_inputs(age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant,
                      thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium,
                      goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI):



    return [age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant,
                      thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium,
                      goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI]


# Function to predict the diagnosis based on inputs
def predict_diagnosis(inputs):
    # Assuming 'model' is a trained machine learning model
    # Replace 'model.predict()' with the actual function to make predictions
    output = model.predict([inputs])[0]
    return output



def main():
    # Title
    st.markdown(title_css, unsafe_allow_html=True)
    st.subheader("Autofill The Lab Record")

    # Upload file
    uploaded_file = st.file_uploader("Upload file", type=["xls", "xlsx", "csv"])

    if uploaded_file is not None:
        # Check file type and read data accordingly
        if uploaded_file.type == "application/vnd.ms-excel" or uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            # Excel file
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.type == "text/csv":
            # CSV file
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file format")
            return



        # Autofill data into input fields
        st.subheader("Autofill Data into Input Fields")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age', value=df.iloc[0]['age'] if not df.empty else None)
            TSH = st.number_input('TSH', value=df.iloc[0]['TSH'] if not df.empty else None)
            T4U = st.number_input('T4U', value=df.iloc[0]['T4U'] if not df.empty else None)
            T3 = st.number_input('T3', value=df.iloc[0]['T3'] if not df.empty else None)
            FTI = st.number_input('FTI', value=df.iloc[0]['FTI'] if not df.empty else None)
            TT4 = st.number_input('TT4', value=df.iloc[0]['TT4'] if not df.empty else None)


            boolean_options = [0,1]

            query_on_thyroxine_value = df.iloc[0]['query_on_thyroxine'] if not df.empty else None
            query_on_thyroxine_index = boolean_options.index(query_on_thyroxine_value) if query_on_thyroxine_value in boolean_options else 0
            query_on_thyroxine = st.selectbox('Query On Thyroxine', options=boolean_options,
                                              format_func=lambda x: 'True' if x else 'False',
                                              index=query_on_thyroxine_index)
        with col2:
            boolean_options = [0, 1]
            pregnant_value = df.iloc[0]['pregnant'] if not df.empty else None
            pregnant_index = boolean_options.index(pregnant_value) if pregnant_value in boolean_options else 0
            pregnant = st.selectbox('Pregnant', options=boolean_options,format_func=lambda x: 'True' if x else 'False', index=pregnant_index)

            query_hypothyroid_value = df.iloc[0]['query_hypothyroid'] if not df.empty else None
            query_hypothyroid_index = boolean_options.index(query_hypothyroid_value) if query_hypothyroid_value in boolean_options else 0
            query_hypothyroid = st.selectbox('Query Hypothyroid', options=boolean_options,format_func=lambda x: 'True' if x else 'False',
                                             index=query_hypothyroid_index)

            on_thyroxine_value = df.iloc[0]['on_thyroxine'] if not df.empty else None
            on_thyroxine_index = boolean_options.index(on_thyroxine_value) if on_thyroxine_value in boolean_options else 0
            on_thyroxine = st.selectbox('On Thyroxine', options=boolean_options,format_func=lambda x: 'True' if x else 'False', index=on_thyroxine_index)

            sick_value = df.iloc[0]['sick'] if not df.empty else None
            sick_index = boolean_options.index(sick_value) if sick_value in boolean_options else 0
            sick = st.selectbox('Sick', options=boolean_options,format_func=lambda x: 'True' if x else 'False', index=sick_index)

            I131_treatment_value = df.iloc[0]['I131_treatment'] if not df.empty else None
            I131_treatment_index = boolean_options.index(I131_treatment_value) if I131_treatment_value in boolean_options else 0
            I131_treatment = st.selectbox('I131 Treatment', options=boolean_options,format_func=lambda x: 'True' if x else 'False', index=I131_treatment_index)

            lithium_value = df.iloc[0]['lithium'] if not df.empty else None
            lithium_index = boolean_options.index(lithium_value) if lithium_value in boolean_options else 0
            lithium = st.selectbox('Lithium', options=boolean_options,format_func=lambda x: 'True' if x else 'False', index=lithium_index)

            hypopituitary_value = df.iloc[0]['hypopituitary'] if not df.empty else None
            hypopituitary_index = boolean_options.index(hypopituitary_value) if hypopituitary_value in boolean_options else 0
            hypopituitary = st.selectbox('Hypopituitary', options=boolean_options,format_func=lambda x: 'True' if x else 'False', index=hypopituitary_index)

        with col3:


            sex_value = df.iloc[0]['sex'] if not df.empty else None
            sex_index = boolean_options.index(sex_value) if sex_value in boolean_options else 0
            sex = st.selectbox('Sex', options=boolean_options, format_func=lambda x: 'True' if x else 'False',
                               index=sex_index)

            on_antithyroid_meds_value = df.iloc[0]['on_antithyroid_meds'] if not df.empty else None
            on_antithyroid_meds_index = boolean_options.index(
                on_antithyroid_meds_value) if on_antithyroid_meds_value in boolean_options else 0
            on_antithyroid_meds = st.selectbox('On Antithyroid Medication', options=boolean_options,
                                                     format_func=lambda x: 'True' if x else 'False',
                                                     index=on_antithyroid_meds_index)

            thyroid_surgery_value = df.iloc[0]['thyroid_surgery'] if not df.empty else None
            thyroid_surgery_index = boolean_options.index(
                thyroid_surgery_value) if thyroid_surgery_value in boolean_options else 0
            thyroid_surgery = st.selectbox('Thyroid Surgery', options=boolean_options,
                                           format_func=lambda x: 'True' if x else 'False', index=thyroid_surgery_index)

            query_hyperthyroid_value = df.iloc[0]['query_hyperthyroid'] if not df.empty else None
            query_hyperthyroid_index = boolean_options.index(
                query_hyperthyroid_value) if query_hyperthyroid_value in boolean_options else 0
            query_hyperthyroid = st.selectbox('Query Hyperthyroid', options=boolean_options,
                                              format_func=lambda x: 'True' if x else 'False',
                                              index=query_hyperthyroid_index)

            tumor_value = df.iloc[0]['tumor'] if not df.empty else None
            tumor_index = boolean_options.index(tumor_value) if tumor_value in boolean_options else 0
            tumor = st.selectbox('Tumor', options=boolean_options, format_func=lambda x: 'True' if x else 'False',
                                 index=tumor_index)

            goitre_value = df.iloc[0]['goitre'] if not df.empty else None
            goitre_index = boolean_options.index(goitre_value) if goitre_value in boolean_options else 0
            goitre = st.selectbox('Goitre', options=boolean_options, format_func=lambda x: 'True' if x else 'False',
                                  index=goitre_index)

            psych_value = df.iloc[0]['psych'] if not df.empty else None
            psych_index = boolean_options.index(psych_value) if psych_value in boolean_options else 0
            psych = st.selectbox('Psych', options=boolean_options, format_func=lambda x: 'True' if x else 'False',
                                 index=psych_index)
    # Detect button
        with col2:
            detect_button = st.button('Detect', key='predict_button')
            detect_button_container = st.container()
            with detect_button_container:
                detect_button_css = f"""
                           <style>
                               .stButton > button:first-child {{
                                   width: 50%;
                                   color: white;
                                   border-color: {detect_button_color};
                                   border-radius: 5px;
                                   padding: 10px;
                                   margin: 0 auto; /* Center the button horizontally */
                                   display: block;
                               }}
                           </style>
                       """
                st.markdown(detect_button_css, unsafe_allow_html=True)

            if detect_button:
                # Preprocess inputs
                inputs = age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_meds, sick, pregnant,thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium,goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI
                # Get prediction
                diagnosis_num = predict_diagnosis(inputs)
                diagnosis_label = diagnoses.get(diagnosis_num, 'Unknown')
                st.markdown(
                    f"<h1 style='text-align: center; color: {diagnosis_color};'>{diagnosis_label}</h1>",
                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()
