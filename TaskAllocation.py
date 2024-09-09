import streamlit as st
import base64
import pandas as pd

# ------------ TITLE 
st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"A visual machine scheduler simulator for machine optimization"}</h1>', unsafe_allow_html=True)

# ================ Background image ===
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.jpg')   

# Upload CSV file
st.header('Upload Dataset')
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

# Function to approximately allocate tasks to combinations
def approximate_allocate_tasks(tasks, combinations):
    allocated_combinations = []
    for task in tasks:
        task_lower = task.lower()
        best_match = None
        best_comb = None
        highest_match_score = 0

        for comb in combinations:
            for _, desc in comb:
                # Simple substring match score
                match_score = 0
                if task_lower in desc.lower():
                    match_score = len(task_lower) / len(desc.lower())  # Example score based on length ratio
                
                # Update best match if this score is higher
                if match_score > highest_match_score:
                    highest_match_score = match_score
                    best_match = task
                    best_comb = comb
        
        if best_comb:
            allocated_combinations.append((best_match, best_comb))
        else:
            allocated_combinations.append((task, "No suitable combination found"))
    
    return allocated_combinations

if uploaded_file is not None:
    # Load dataset
    data_frame1 = pd.read_csv(uploaded_file)
    st.write("### Dataset Sample")
    st.write(data_frame1.head())
    
    data_frame1 = data_frame1.fillna(0)
    
    # Get user inputs
    description_input1 = st.text_input("Enter the first description:")
    description_input2 = st.text_input("Enter the second description:")
    description_input3 = st.text_input("Enter the third description:")
    
    description_inputs = [description_input1, description_input2, description_input3]
    
    if st.button("Allocate Tasks"):
        data_label = data_frame1['Description']
        data_frame1_c = data_frame1['Machine']

        allocated_machines = set()  # To keep track of allocated machines
        
        for description_input in description_inputs:
            if description_input:  # Ensure input is not empty
                # Find the index of the description_input
                idx_series = data_label[data_label == description_input]
                if not idx_series.empty:
                    idx = idx_series.index[0]
                    Req_data_c = data_frame1_c[idx]
                    
                    if Req_data_c in allocated_machines:
                        # Find the next available machine
                        available_machines = data_frame1_c[~data_frame1_c.isin(allocated_machines)].unique()
                        if len(available_machines) > 0:
                            Req_data_c = available_machines[0]
                        else:
                            Req_data_c = "No available machines"
                    
                    # Update the set of allocated machines
                    if Req_data_c != "No available machines":
                        allocated_machines.add(Req_data_c)
                    
                    if Req_data_c == 0 or Req_data_c == "No available machines":
                        st.write(f"Unable to allocate tasks for '{description_input}'.")
                    else:
                        st.write(f"Task '{description_input}' approximately allocated to {Req_data_c}.")
                else:
                    st.write(f"Description '{description_input}' not found.")
            else:
                st.write("Please enter a description in all fields.")
