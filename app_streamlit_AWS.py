import json
import os
import pandas as pd

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features):
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))

st.set_page_config(page_title="Credit Score Prediction", page_icon="💳", layout="centered")

def main():
    st.title("💳 Credit Score Prediction")
    st.caption("Isi data nasabah di bawah, lalu klik tombol untuk memprediksi kategori credit score.")

    with st.form("prediction_form"):

        st.subheader("Identitas")
        col1, col2 = st.columns(2)
        with col1:
            id_ = st.text_input("ID")
            name = st.text_input("Name")
        with col2:
            ssn = st.text_input("SSN")
            customer_id = st.text_input("Customer ID")

        st.subheader("Data Personal & Finansial")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, step=1)
            occupation = st.selectbox("Occupation", ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer', 
                                                    'Entrepreneur', 'Journalist', 'Lawyer', 'Manager', 'Mechanic',
                                                     'Media_Manager', 'Musician', 'Scientist', 'Teacher', 'Writer'])
            annual_income = st.number_input("Annual Income", min_value=0.0, step=100.0)
            num_of_loan = st.number_input("Num of Loan", min_value=0, step=1)
        with col2:
            num_bank_accounts = st.number_input("Num Bank Accounts", min_value=1, step=1)
            num_credit_card = st.number_input("Num Credit Card", min_value=0, step=1)
            interest_rate = st.number_input("Interest Rate (%)", min_value=1, step=1)

        st.subheader("Pinjaman & Pembayaran")
        type_of_loan = st.multiselect("Type of Loan", [
            'Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan',
            'Mortgage Loan', 'Payday Loan', 'Personal Loan', 'Student Loan'])

        col1, col2 = st.columns(2)
        with col1:
            delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0, step=1)
            num_of_delayed_payment = st.number_input("Num of Delayed Payment", min_value=0, step=1)
            changed_credit_limit = st.number_input("Changed Credit Limit (%)", step=0.1)
            num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0, step=1)
        with col2:
            credit_mix = st.selectbox("Credit Mix", ['Bad', 'Standard', 'Good', 'Unknown'])
            payment_of_min_amount = st.selectbox("Payment of Min Amount", ['Yes', 'No', 'Unknown'])
            payment_behaviour = st.selectbox("Payment Behaviour", [
                'High_spent_Small_value_payments', 'High_spent_Medium_value_payments',
                'High_spent_Large_value_payments', 'Low_spent_Small_value_payments',
                'Low_spent_Medium_value_payments', 'Low_spent_Large_value_payments', 'Unknown'])

        st.subheader("Riwayat Kredit & Saldo")
        col1, col2 = st.columns(2)
        with col1:
            outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, step=10.0)
            credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, step=0.1)
            total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, step=10.0)
        with col2:
            amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, step=10.0)
            monthly_balance = st.number_input("Monthly Balance", min_value=0.0, step=10.0)
            ch_months = st.number_input("Credit History (Months)", min_value=0, step=1)

        submit = st.form_submit_button("Predict Credit Score", use_container_width=True)

    if submit:
        loan_types = ['Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan',
                    'Mortgage Loan', 'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan']
        has_loan = {
            'Has_' + lt.replace(' ', '_').replace('-', '_'): (1 if lt in type_of_loan else 0)
            for lt in loan_types
        }

        input_df = pd.DataFrame([{
            'Age': age,
            'Annual_Income': annual_income,
            'Num_Bank_Accounts': num_bank_accounts,
            'Num_Credit_Card': num_credit_card,
            'Interest_Rate': interest_rate,
            'Num_of_Loan': num_of_loan,
            'Delay_from_due_date': delay_from_due_date,
            'Num_of_Delayed_Payment': num_of_delayed_payment,
            'Changed_Credit_Limit': changed_credit_limit,
            'Num_Credit_Inquiries': num_credit_inquiries,
            'Outstanding_Debt': outstanding_debt,
            'Credit_Utilization_Ratio': credit_utilization_ratio,
            'Credit_History_Age': ch_months,
            'Total_EMI_per_month': total_emi_per_month,
            'Amount_invested_monthly': amount_invested_monthly,
            'Monthly_Balance': monthly_balance,
            'Occupation': occupation,
            'Payment_of_Min_Amount': payment_of_min_amount,
            'Payment_Behaviour': payment_behaviour,
            'Credit_Mix': credit_mix,
            **has_loan
        }])
        
        features = input_df.iloc[0].tolist()

        try:
            result = invoke_endpoint(features)
        except NoCredentialsError:
            st.error(
                "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            )
        except ClientError as e:
            st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
        else:
            st.divider()
            st.subheader("Hasil Prediksi")
            st.write(f"Nasabah: **{name or '-'}** (ID: {customer_id or '-'})")

            pred_label = result['labels'][0]

            if pred_label == 'Good':
                st.success(f"Credit Score: {pred_label}")
            elif pred_label == 'Standard':
                st.warning(f"Credit Score: {pred_label}")
            else:
                st.error(f"Credit Score: {pred_label}")

if __name__ == "__main__":
    main()
