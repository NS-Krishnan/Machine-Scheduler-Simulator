# ====================== IMPORT PACKAGES ==============

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 

import streamlit as st
import base64

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

# ===-------------------------= INPUT DATA -------------------- 


filenamee = st.file_uploader("Choose a Dataset", ['csv'])

if filenamee is None:
    
    st.text("Please Upload Dataset")

else:

    dataframe=pd.read_csv("Dataset.csv")
        
    print("--------------------------------")
    print("Data Selection")
    print("--------------------------------")
    print()
    print(dataframe.head(15))    
    
    
    
    st.write("--------------------------------")
    st.write("Data Selection")
    st.write("--------------------------------")
    print()
    st.write(dataframe.head(15))    
    
    
 #-------------------------- PRE PROCESSING --------------------------------
    
    #------ checking missing values --------
    
    print("----------------------------------------------------")
    print("              Handling Missing values               ")
    print("----------------------------------------------------")
    print()
    print(dataframe.isnull().sum())
    
    
    st.write("----------------------------------------------------")
    st.write("              Handling Missing values               ")
    st.write("----------------------------------------------------")
    print()
    st.write(dataframe.isnull().sum())
    
    res = dataframe.isnull().sum().any()
        
    if res == False:
        
        print("--------------------------------------------")
        print("  There is no Missing values in our dataset ")
        print("--------------------------------------------")
        print()    
        
        st.write("--------------------------------------------")
        st.write("  There is no Missing values in our dataset ")
        st.write("--------------------------------------------")
        print()   
        
    else:
    
        print("--------------------------------------------")
        print(" Missing values is present in our dataset   ")
        print("--------------------------------------------")
        print()    
        
        st.write("--------------------------------------------")
        st.write(" Missing values is present in our dataset   ")
        st.write("--------------------------------------------")
        print()   
        
        dataframe = dataframe.fillna(0)
        
        resultt = dataframe.isnull().sum().any()
        
        if resultt == False:
            
            print("--------------------------------------------")
            print(" Data Cleaned !!!   ")
            print("--------------------------------------------")
            print()    
            print(dataframe.isnull().sum())

            st.write("--------------------------------------------")
            st.write(" Data Cleaned !!!   ")
            st.write("--------------------------------------------")
            st.write() 
            
            st.write(dataframe.isnull().sum())
                
      # ---- LABEL ENCODING
            
    print("--------------------------------")
    print("Before Label Encoding")
    print("--------------------------------")   
    
    df_class=dataframe['Description']
    
    print(dataframe['Description'].head(15))
    
                
    st.write("--------------------------------")
    st.write("Before Label Encoding")
    st.write("--------------------------------")   
    
    df_class=dataframe['Description']
    
    st.write(dataframe['Description'].head(15))
    
    
    
    print("--------------------------------")
    print("After Label Encoding")
    print("--------------------------------")            
            
    label_encoder = preprocessing.LabelEncoder() 
    
    dataframe['Description']=label_encoder.fit_transform(dataframe['Description'].astype(str))                  
    
    dataframe['Material']=label_encoder.fit_transform(dataframe['Material'].astype(str))     

    dataframe['Treatment']=label_encoder.fit_transform(dataframe['Treatment'].astype(str))     

    dataframe['Tolerance']=label_encoder.fit_transform(dataframe['Tolerance'].astype(str))     

    dataframe['Supplier']=label_encoder.fit_transform(dataframe['Supplier'].astype(str))   
                       
    dataframe['Code']=label_encoder.fit_transform(dataframe['Code'].astype(str))        
            
    dataframe['Vice']=label_encoder.fit_transform(dataframe['Vice'].astype(str))    
    
    dataframe['Machine']=label_encoder.fit_transform(dataframe['Machine'].astype(str))     
                
    print(dataframe['Description'].head(15))       
    
    
    st.write("--------------------------------")
    st.write("After Label Encoding")
    st.write("--------------------------------")            
            
    label_encoder = preprocessing.LabelEncoder() 
    
    dataframe['Description']=label_encoder.fit_transform(dataframe['Description'].astype(str))       

    
    dataframe['Material']=label_encoder.fit_transform(dataframe['Material'].astype(str))     

    dataframe['Treatment']=label_encoder.fit_transform(dataframe['Treatment'].astype(str))     

    
    dataframe['Tolerance']=label_encoder.fit_transform(dataframe['Tolerance'].astype(str))     

    dataframe['Supplier']=label_encoder.fit_transform(dataframe['Supplier'].astype(str))   
           
                
    dataframe['Code']=label_encoder.fit_transform(dataframe['Code'].astype(str))        
            
    dataframe['Vice']=label_encoder.fit_transform(dataframe['Vice'].astype(str))    
    
    dataframe['Machine']=label_encoder.fit_transform(dataframe['Machine'].astype(str))     
           
            
    st.write(dataframe['Description'].head(15))      
    
    #------ checking missing values --------

    
    dataframe = dataframe.drop(['Dimensions','Priority'],axis=1)


   
   # ================== DATA SPLITTING  ====================
    
    
    X=dataframe.drop('Group',axis=1)
    
    y=dataframe['Group']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    print("---------------------------------------------")
    print("             Data Splitting                  ")
    print("---------------------------------------------")
    
    print()
    
    print("Total no of input data   :",dataframe.shape[0])
    print("Total no of test data    :",X_test.shape[0])
    print("Total no of train data   :",X_train.shape[0])
    
    
    
    st.write("---------------------------------------------")
    st.write("             Data Splitting                  ")
    st.write("---------------------------------------------")
    
    print()
    
    st.write("Total no of input data   :",dataframe.shape[0])
    st.write("Total no of test data    :",X_test.shape[0])
    st.write("Total no of train data   :",X_train.shape[0])
    
    # ================== CLASSIFCATION  ====================
    
    # ------ RANDOM FOREST ------
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier()
    
    rf.fit(X_train,y_train)
    
    pred_rf = rf.predict(X_test)
    
    
    from sklearn import metrics
    
    acc_rf = metrics.accuracy_score(pred_rf,y_test) * 100
    
    print("---------------------------------------------")
    print("       Classification - Random Forest        ")
    print("---------------------------------------------")
    
    print()
    
    print("1) Accuracy = ", acc_rf , '%')
    print()
    print("2) Classification Report")
    print(metrics.classification_report(pred_rf,y_test))
    print()
    print("3) Error Rate = ", 100 - acc_rf, '%')
    
    
    st.write("---------------------------------------------")
    st.write("       Classification - Random Forest        ")
    st.write("---------------------------------------------")
    
    print()
    
    st.write("1) Accuracy = ", acc_rf , '%')
    print()
    st.write("2) Classification Report")
    st.write(metrics.classification_report(pred_rf,y_test))
    print()
    st.write("3) Error Rate = ", 100 - acc_rf, '%')
    
    
    # -------- Support Vector Machine---------------------------
    
    from sklearn.svm import SVC
    
    lr = SVC()
    
    lr.fit(X_train,y_train)
    
    pred_lr = lr.predict(X_test)
    
    
    from sklearn import metrics
    
    acc_lr = metrics.accuracy_score(pred_lr,y_test) * 100    
     
    
    print("-----------------------------------------------")
    print("       Classification - Support Vector Machine ")
    print("-----------------------------------------------")
    
    print()
    
    print("1) Accuracy = ", acc_lr , '%')
    print()
    print("2) Classification Report")
    print(metrics.classification_report(pred_lr,y_test))
    print()
    print("3) Error Rate = ", 100 - acc_lr, '%')
    

    st.write("---------------------------------------------")
    st.write("       Classification - Support Vector Machine  ")
    st.write("---------------------------------------------")
    
    print()
    
    st.write("1) Accuracy = ", acc_lr , '%')
    print()
    st.write("2) Classification Report")
    st.write(metrics.classification_report(pred_lr,y_test))
    print()
    st.write("3) Error Rate = ", 100 - acc_lr, '%')        
    
        
    #============================= COMPARISON GRAPH  ==============================
    
    import matplotlib.pyplot as plt
    
    vals=[acc_rf,acc_lr]
    inds=range(len(vals))
    labels=["RF ","SVM" ]
    fig,ax = plt.subplots()
    rects = ax.bar(inds, vals,color=['blue', 'green'])
    ax.set_xticks([ind for ind in inds])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Comparison of Classification Models')
    st.pyplot(fig)   
    
    
    # -------------- GENETIC 
   
    import random
    
    # Define the task scheduling problem
    tasks = ["Task1", "Task2", "Task3", "Task4", "Task5"]
    machines = ["Machine1", "Machine2", "Machine3"]
    
    # Define the number of generations and population size
    num_generations = 100
    population_size = 20
    
    # Function to initialize a random schedule
    def initialize_schedule(tasks, machines):
        schedule = {}
        for task in tasks:
            schedule[task] = random.choice(machines)
        return schedule
    
    # Function to calculate the fitness of a schedule (objective: minimize makespan)
    def calculate_fitness(schedule):
        machine_time = {machine: 0 for machine in machines}
        for task, machine in schedule.items():
            machine_time[machine] += random.randint(1, 10)  # Replace with actual task duration
        return max(machine_time.values())
    
    # Function for tournament selection
    def tournament_selection(population, tournament_size):
        selected = random.sample(population, tournament_size)
        return min(selected, key=lambda x: x[1])
    
    # Function for crossover (two-point crossover)
    def crossover(parent1, parent2):
        point1 = random.randint(0, len(tasks) - 1)
        point2 = random.randint(point1, len(tasks))
        
        child = {}
        
        # Take the keys and values up to point1 from parent1
        child.update({key: parent1[key] for key in list(parent1.keys())[:point1]})
        
        # Take the keys and values between point1 and point2 from parent2
        child.update({key: parent2[key] for key in list(parent2.keys())[point1:point2]})
        
        # Take the keys and values after point2 from parent1
        child.update({key: parent1[key] for key in list(parent1.keys())[point2:]})
        
        return child
    
    
    # Function for mutation (swap mutation)
    def mutate(schedule):
        task1, task2 = random.sample(tasks, 2)
        schedule[task1], schedule[task2] = schedule[task2], schedule[task1]
        return schedule
    
    # Genetic Algorithm
    def genetic_algorithm(tasks, machines, num_generations, population_size, tournament_size):
        population = [(initialize_schedule(tasks, machines), 0) for _ in range(population_size)]
    
        for generation in range(num_generations):
            # Evaluate fitness
            population = [(schedule, calculate_fitness(schedule)) for schedule, _ in population]
    
            # Select parents using tournament selection
            parents = [tournament_selection(population, tournament_size) for _ in range(population_size)]
    
            # Create offspring through crossover
            offspring = [crossover(parents[i], parents[i + 1]) for i in range(0, population_size, 2)]
    
            # Apply mutation to offspring
            offspring = [mutate(schedule) for schedule in offspring]
    
            # Replace the old population with the combined population of parents and offspring
            population = parents + [(schedule, 0) for schedule in offspring]
    
        # Return the best schedule found
        best_schedule, _ = min(population, key=lambda x: x[1])
        return best_schedule    
    
    
    #============================= TASK SCHEDULING ==============================
    
    #===== ROUND ROBIN ==========
    
    
    print("-----------------------------------------------")
    print("Round Robin (Task Scheduling)")
    print("-----------------------------------------------")
    print()
    
    
    class RoundRobin:
    
        def processData(self, no_of_processes):
            process_data = []
            for i in range(no_of_processes):
                temporary = []
                import random
                process_id=random.randint(0,10)
    #            process_id = int(input("Enter Process ID: "))
    
                arrival_time = 2
    
                burst_time = 3
    
                temporary.extend([process_id, arrival_time, burst_time, 0, burst_time])
    
                process_data.append(temporary)
    
            time_slice = 1
    
            RoundRobin.schedulingProcess(self, process_data, time_slice)
    
        def schedulingProcess(self, process_data, time_slice):
            start_time = []
            exit_time = []
            executed_process = []
            ready_queue = []
            s_time = 0
            process_data.sort(key=lambda x: x[1])
    
            while 1:
                normal_queue = []
                temp = []
                for i in range(len(process_data)):
                    if process_data[i][1] <= s_time and process_data[i][3] == 0:
                        present = 0
                        if len(ready_queue) != 0:
                            for k in range(len(ready_queue)):
                                if process_data[i][0] == ready_queue[k][0]:
                                    present = 1
    
                        if present == 0:
                            temp.extend([process_data[i][0], process_data[i][1], process_data[i][2], process_data[i][4]])
                            ready_queue.append(temp)
                            temp = []
    
                        if len(ready_queue) != 0 and len(executed_process) != 0:
                            for k in range(len(ready_queue)):
                                if ready_queue[k][0] == executed_process[len(executed_process) - 1]:
                                    ready_queue.insert((len(ready_queue) - 1), ready_queue.pop(k))
    
                    elif process_data[i][3] == 0:
                        temp.extend([process_data[i][0], process_data[i][1], process_data[i][2], process_data[i][4]])
                        normal_queue.append(temp)
                        temp = []
                if len(ready_queue) == 0 and len(normal_queue) == 0:
                    break
                if len(ready_queue) != 0:
                    if ready_queue[0][2] > time_slice:
    
                        start_time.append(s_time)
                        s_time = s_time + time_slice
                        e_time = s_time
                        exit_time.append(e_time)
                        executed_process.append(ready_queue[0][0])
                        for j in range(len(process_data)):
                            if process_data[j][0] == ready_queue[0][0]:
                                break
                        process_data[j][2] = process_data[j][2] - time_slice
                        ready_queue.pop(0)
                    elif ready_queue[0][2] <= time_slice:
    
                        start_time.append(s_time)
                        s_time = s_time + ready_queue[0][2]
                        e_time = s_time
                        exit_time.append(e_time)
                        executed_process.append(ready_queue[0][0])
                        for j in range(len(process_data)):
                            if process_data[j][0] == ready_queue[0][0]:
                                break
                        process_data[j][2] = 0
                        process_data[j][3] = 1
                        process_data[j].append(e_time)
                        ready_queue.pop(0)
                elif len(ready_queue) == 0:
                    if s_time < normal_queue[0][1]:
                        s_time = normal_queue[0][1]
                    if normal_queue[0][2] > time_slice:
    
                        start_time.append(s_time)
                        s_time = s_time + time_slice
                        e_time = s_time
                        exit_time.append(e_time)
                        executed_process.append(normal_queue[0][0])
                        for j in range(len(process_data)):
                            if process_data[j][0] == normal_queue[0][0]:
                                break
                        process_data[j][2] = process_data[j][2] - time_slice
                    elif normal_queue[0][2] <= time_slice:
    
                        start_time.append(s_time)
                        s_time = s_time + normal_queue[0][2]
                        e_time = s_time
                        exit_time.append(e_time)
                        executed_process.append(normal_queue[0][0])
                        for j in range(len(process_data)):
                            if process_data[j][0] == normal_queue[0][0]:
                                break
                        process_data[j][2] = 0
                        process_data[j][3] = 1
                        process_data[j].append(e_time)
            t_time = RoundRobin.calculateTurnaroundTime(self, process_data)
            w_time = RoundRobin.calculateWaitingTime(self, process_data)
            RoundRobin.printData(self, process_data, t_time, w_time, executed_process)
    
        def calculateTurnaroundTime(self, process_data):
            total_turnaround_time = 0
            for i in range(len(process_data)):
                turnaround_time = process_data[i][5] - process_data[i][1]
                '''
                turnaround_time = completion_time - arrival_time
                '''
                total_turnaround_time = total_turnaround_time + turnaround_time
                process_data[i].append(turnaround_time)
            average_turnaround_time = total_turnaround_time / len(process_data)
            '''
            average_turnaround_time = total_turnaround_time / no_of_processes
            '''
            return average_turnaround_time
    
        def calculateWaitingTime(self, process_data):
            total_waiting_time = 0
            for i in range(len(process_data)):
                waiting_time = process_data[i][6] - process_data[i][4]
                '''
                waiting_time = turnaround_time - burst_time
                '''
                total_waiting_time = total_waiting_time + waiting_time
                process_data[i].append(waiting_time)
            average_waiting_time = total_waiting_time / len(process_data)
            '''
            average_waiting_time = total_waiting_time / no_of_processes
            '''
            return average_waiting_time
    
        def printData(self, process_data, average_turnaround_time, average_waiting_time, executed_process):
            process_data.sort(key=lambda x: x[0])
            '''
            Sort processes according to the Process ID
            '''
            print("P.ID  Arrival.T  Rem_Burst.T  Completed  Origl.Burst.T  Completion.T Turnaround.T Wait.T")
            for i in range(len(process_data)):
                for j in range(len(process_data[i])):
    
                    print(process_data[i][j], end="		   ")
                print()
    
            print(f'Average Turnaround Time: {average_turnaround_time}')
    
            print(f'Average Waiting Time: {average_waiting_time}')
    
            print(f'Sequence of Processes: {executed_process}')
    
    
    if __name__ == "__main__":
        no_of_processes = 2
        rr = RoundRobin()
        rr.processData(no_of_processes)   
    
    
    
    
    
    
    
    
    
    
    
    
    