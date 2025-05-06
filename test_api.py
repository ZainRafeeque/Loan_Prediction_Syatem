import requests

url = "http://192.168.0.141:5000/predict"
json_input = {
   
    "Gender": 1,                # 1 for Male, 0 for Female
    "Married": 1,               # 1 for Married, 0 for Single
    "Dependents": 0,            # Number of dependents
    "Education": 1,             # 1 for Graduate, 0 for Non-Graduate
    "Self_Employed": 0,         # 1 for Yes, 0 for No
    "ApplicantIncome": 10000,   # Income of the applicant
    "CoapplicantIncome": 4000,  # Income of the co-applicant
    "LoanAmount": 4500,         # Loan amount in thousands
    "Loan_Amount_Term": 30,     # Loan term in months
    "Credit_History": 1,        # 1 for good credit history, 0 for bad
    "Property_Area": 2          # 2 for Urban, 1 for Semi-Urban, 0 for Rural
}




response = requests.post(url, json=json_input)
print(response.json())
