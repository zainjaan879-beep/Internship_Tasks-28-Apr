import streamlit as st # type: ignore
import pandas as pd
import plotly.express as px # type: ignore

# Page config
st.set_page_config(page_title="Superstore Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("Global_Superstore.csv", encoding="latin1")

# Clean data
df = df.dropna()

# Sidebar filters
st.sidebar.title("Filters")

region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

subcategory = st.sidebar.multiselect(
    "Select Sub-Category",
    options=df["Sub-Category"].unique(),
    default=df["Sub-Category"].unique()
)

# Filter dataset
filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Sub-Category"].isin(subcategory))
]

# KPIs
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()

top_customers = filtered_df.groupby("Customer Name")["Sales"].sum().nlargest(5).reset_index()

# Title
st.title("📊 Global Superstore Dashboard")

# KPI Display
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Sales", f"${total_sales:,.2f}")

with col2:
    st.metric("Total Profit", f"${total_profit:,.2f}")

# Charts

st.subheader("Top 5 Customers by Sales")

fig1 = px.bar(
    top_customers,
    x="Customer Name",
    y="Sales",
    color="Sales"
)

st.plotly_chart(fig1, use_container_width=True)

# Sales by Category
st.subheader("Sales by Category")

cat_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()

fig2 = px.pie(
    cat_sales,
    values="Sales",
    names="Category"
)

st.plotly_chart(fig2, use_container_width=True)

# Profit by Region
st.subheader("Profit by Region")

region_profit = filtered_df.groupby("Region")["Profit"].sum().reset_index()

fig3 = px.bar(
    region_profit,
    x="Region",
    y="Profit",
    color="Profit"
)

st.plotly_chart(fig3, use_container_width=True)

st.success("Dashboard Loaded Successfully")