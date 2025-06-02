import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib

model = joblib.load('RD_model.pkl')

def predict_churn():
    age = int(age_entry.get())
    frequent_flyer = frequent_flyer_combobox.get()
    annual_income = annual_income_combobox.get()
    services_opted = int(services_opted_entry.get())
    social_media_sync = social_media_sync_combobox.get()
    booked_hotel = booked_hotel_combobox.get()
    
    frequent_flyer_no = 1 if frequent_flyer == 'No' else 0
    frequent_flyer_yes = 1 if frequent_flyer == 'Yes' else 0
    
    annual_income_low = 1 if annual_income == 'Low Income' else 0
    annual_income_middle = 1 if annual_income == 'Middle Income' else 0
    annual_income_high = 1 if annual_income == 'High Income' else 0
    
    social_media_sync_no = 1 if social_media_sync == 'No' else 0
    social_media_sync_yes = 1 if social_media_sync == 'Yes' else 0
    
    booked_hotel_no = 1 if booked_hotel == 'No' else 0
    booked_hotel_yes = 1 if booked_hotel == 'Yes' else 0
    
    input_data = {
        'Age': [age],
        'FrequentFlyer': [frequent_flyer],
        'AnnualIncomeClass': [annual_income],
        'ServicesOpted': [services_opted],
        'AccountSyncedToSocialMedia': [social_media_sync],
        'BookedHotelOrNot': [booked_hotel]
    }
    
    input_df = pd.DataFrame(input_data)
    
    input_df = pd.get_dummies(input_df)
    
    missing_columns = set(model.feature_names_in_).difference(input_df.columns)
    for col in missing_columns:
        input_df[col] = 0
    
    input_df = input_df[model.feature_names_in_]  
    
    churn_prediction = model.predict(input_df)

    if churn_prediction[0] == 1:
        prediction_label.config(text="Dự đoán: Khách hàng ngưng sử dụng dịch vụ", fg='red')
    else:
        prediction_label.config(text="Dự đoán : Khách hàng vẫn sử dụng dịch vụ", fg='green')

root = tk.Tk()
root.title("Dự đoán ngưng sử dụng")
root.geometry("400x300")  

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(fill=tk.BOTH, expand=True)

frame.grid_columnconfigure(1, weight=1)

age_label = tk.Label(frame, text="Age:")
age_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
age_entry = tk.Entry(frame)
age_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

frequent_flyer_label = tk.Label(frame, text="Frequent Flyer:")
frequent_flyer_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
frequent_flyer_combobox = ttk.Combobox(frame, values=["No", "Yes"])
frequent_flyer_combobox.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

annual_income_label = tk.Label(frame, text="Annual Income:")
annual_income_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
annual_income_combobox = ttk.Combobox(frame, values=["Low Income", "Middle Income", "High Income"])
annual_income_combobox.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

services_opted_label = tk.Label(frame, text="Services Opted:")
services_opted_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
services_opted_entry = tk.Entry(frame)
services_opted_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

social_media_sync_label = tk.Label(frame, text="Account Synced to Social Media:")
social_media_sync_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
social_media_sync_combobox = ttk.Combobox(frame, values=["No", "Yes"])
social_media_sync_combobox.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

booked_hotel_label = tk.Label(frame, text="Booked Hotel:")
booked_hotel_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
booked_hotel_combobox = ttk.Combobox(frame, values=["No", "Yes"])
booked_hotel_combobox.grid(row=5, column=1, sticky="ew", padx=5, pady=5)

predict_button = tk.Button(frame, text="Dự đoán", command=predict_churn)
predict_button.grid(row=6, columnspan=2, padx=5, pady=5)

prediction_label = tk.Label(frame, text="")
prediction_label.grid(row=7, columnspan=2, padx=5, pady=5)

root.mainloop()