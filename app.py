import streamlit as st
import pandas as pd
import plotly.express as px
import os


def reveal_block(content):
    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    content()
    st.markdown('</div>', unsafe_allow_html=True)
# Load Excel
df = pd.read_excel("Matiks - Data Analyst Data.xlsx")

df['Last_Login'] = pd.to_datetime(df['Last_Login'])
df['Signup_Date'] = pd.to_datetime(df['Signup_Date'])
df['Last_Login_Date'] = df['Last_Login'].dt.date
df['Week'] = df['Last_Login'].dt.strftime('%Y-%U')
df['Month'] = df['Last_Login'].dt.to_period('M').astype(str)



st.sidebar.header("Filter Users")
selected_device = st.sidebar.selectbox("Device Type", options=["All"] + sorted(df['Device_Type'].dropna().unique().tolist()))
selected_subscription = st.sidebar.selectbox("Subscription Tier", options=["All"] + sorted(df['Subscription_Tier'].dropna().unique().tolist()))
selected_mode = st.sidebar.selectbox("Game Mode", options=["All"] + sorted(df['Preferred_Game_Mode'].dropna().unique().tolist()))


filtered_df = df.copy()
if selected_device != "All":
    filtered_df = filtered_df[filtered_df['Device_Type'] == selected_device]
if selected_subscription != "All":
    filtered_df = filtered_df[filtered_df['Subscription_Tier'] == selected_subscription]
if selected_mode != "All":
    filtered_df = filtered_df[filtered_df['Preferred_Game_Mode'] == selected_mode]



# DAU
dau = (
    filtered_df.groupby('Last_Login_Date')['User_ID']
    .nunique()
    .reset_index(name='DAU')
    .sort_values('Last_Login_Date')
)

# WAU
wau = (
    filtered_df.groupby('Week')['User_ID']
    .nunique()
    .reset_index(name='WAU')
    .sort_values('Week')
)
# MAU
mau = (
    filtered_df.groupby('Month')['User_ID']
    .nunique()
    .reset_index(name='MAU')
    .sort_values('Month')
)


# Revenue Trend
monthly_revenue = (
    filtered_df.groupby('Month')['Total_Revenue_USD']
    .sum()
    .reset_index()
    .sort_values('Month')
)



st.title("Matiks User Engagement Dashboard")

st.subheader("Daily Active Users")
fig_dau = px.line(dau, x='Last_Login_Date', y='DAU', title="Daily Active Users")
fig_dau.update_layout(xaxis_title="Date", yaxis_title="DAU")
st.plotly_chart(fig_dau)

st.subheader("Weekly Active Users")
fig_wau = px.line(wau, x='Week', y='WAU', title="Weekly Active Users")
fig_wau.update_layout(xaxis_title="Week", yaxis_title="WAU")
st.plotly_chart(fig_wau)

st.subheader("Monthly Active Users")
fig_mau = px.line(mau, x='Month', y='MAU', title="Monthly Active Users")
fig_mau.update_layout(xaxis_title="Month", yaxis_title="MAU")
st.plotly_chart(fig_mau)
# Average Sessions per User per Month-Year
filtered_df['Month_Year'] = filtered_df['Last_Login'].dt.to_period('M').astype(str)
avg_sessions_monthly = (
    filtered_df.groupby(['Month_Year'])[['Total_Play_Sessions', 'User_ID']]
    .agg({'Total_Play_Sessions': 'sum', 'User_ID': pd.Series.nunique})
    .reset_index()
)
avg_sessions_monthly['Avg_Sessions_Per_User'] = avg_sessions_monthly['Total_Play_Sessions'] / avg_sessions_monthly['User_ID']


st.subheader("Average Sessions Per User Over Time")
fig_sessions = px.line(
    avg_sessions_monthly,
    x="Month_Year",
    y="Avg_Sessions_Per_User",
    title="Avg. Sessions Per User Per Month",
    markers=True
)
fig_sessions.update_layout(
    xaxis_title="Month",
    yaxis_title="Avg. Sessions Per User",
    xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
    showlegend=False
)
st.plotly_chart(fig_sessions)


st.subheader("Average Sessions Per User - Daily, Weekly, Monthly")

filtered_df['Login_Date'] = filtered_df['Last_Login'].dt.date
avg_daily = (
    filtered_df.groupby('Login_Date')[['Total_Play_Sessions', 'User_ID']]
    .agg({'Total_Play_Sessions': 'sum', 'User_ID': pd.Series.nunique})
    .reset_index()
)
avg_daily['Avg_Sessions_Per_User'] = avg_daily['Total_Play_Sessions'] / avg_daily['User_ID']
fig_daily = px.line(avg_daily, x="Login_Date", y="Avg_Sessions_Per_User", title="Daily Avg. Sessions Per User")
st.plotly_chart(fig_daily)
filtered_df['Week'] = filtered_df['Last_Login'].dt.strftime('%Y-%U')
avg_weekly = (
    filtered_df.groupby('Week')[['Total_Play_Sessions', 'User_ID']]
    .agg({'Total_Play_Sessions': 'sum', 'User_ID': pd.Series.nunique})
    .reset_index()
)
avg_weekly['Avg_Sessions_Per_User'] = avg_weekly['Total_Play_Sessions'] / avg_weekly['User_ID']
fig_weekly = px.line(avg_weekly, x="Week", y="Avg_Sessions_Per_User", title="Weekly Avg. Sessions Per User")
st.plotly_chart(fig_weekly)


fig_monthly = px.line(avg_sessions_monthly, x="Month_Year", y="Avg_Sessions_Per_User", title="Monthly Avg. Sessions Per User")
fig_monthly.update_layout(xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)))
st.plotly_chart(fig_monthly)

st.subheader("Monthly Revenue")
fig = px.line(
    monthly_revenue,
    x="Month",
    y="Total_Revenue_USD",
    title="Monthly Revenue Trend",
    markers=True
)
fig.update_layout(
    yaxis_title="Revenue (USD)",
    xaxis_title="Month",
    showlegend=False,
    xaxis=dict(rangeslider=dict(visible=True))
)
st.plotly_chart(fig)

st.subheader("Average Session Duration Over Months")
monthly_duration = (
    filtered_df.groupby('Month')['Avg_Session_Duration_Min']
    .mean()
    .reset_index()
    .sort_values('Month')
)
fig_duration = px.line(
    monthly_duration,
    x="Month",
    y="Avg_Session_Duration_Min",
    title="Avg. Session Duration (min) Over Time",
    markers=True,
    line_shape="spline"
)
fig_duration.update_traces(line=dict(color="#EF553B", width=3), fill='tozeroy')
fig_duration.update_layout(
    yaxis_title="Avg. Session Duration (min)",
    xaxis_title="Month",
    showlegend=False
)
st.plotly_chart(fig_duration)


# Optional: Filter by device or mode
st.subheader("Breakdown by Device")
device_revenue = filtered_df.groupby('Device_Type')['Total_Revenue_USD'].sum().reset_index()
fig_device = px.line(device_revenue, x="Device_Type", y="Total_Revenue_USD", title="Revenue by Device Type", markers=True)
st.plotly_chart(fig_device)

# Revenue by Game Mode
st.subheader("Revenue by Game Mode")
mode_revenue = filtered_df.groupby('Preferred_Game_Mode')['Total_Revenue_USD'].sum().reset_index()
fig_mode = px.line(mode_revenue, x="Preferred_Game_Mode", y="Total_Revenue_USD", title="Revenue by Game Mode", markers=True)
st.plotly_chart(fig_mode)

# Revenue by Subscription Tier
st.subheader("Revenue by Subscription Tier")
tier_revenue = filtered_df.groupby('Subscription_Tier')['Total_Revenue_USD'].sum().reset_index()
fig_tier = px.line(tier_revenue, x="Subscription_Tier", y="Total_Revenue_USD", title="Revenue by Subscription Tier", markers=True)
st.plotly_chart(fig_tier)


# CSV Export

st.subheader("Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name='filtered_user_data.csv', mime='text/csv')




# High-Value and Churn User Insights
st.subheader("High-Value and Churn User Characteristics")

# High-value user definition: Top 10% by revenue
filtered_df['High_Value'] = filtered_df['Total_Revenue_USD'] > filtered_df['Total_Revenue_USD'].quantile(0.9)

# Churn risk: Low session count or short session duration
filtered_df['Churn_Risk'] = (filtered_df['Total_Play_Sessions'] <= 2) | (filtered_df['Avg_Session_Duration_Min'] < 5)

# Display metrics
high_value_count = filtered_df['High_Value'].sum()
churn_risk_count = filtered_df['Churn_Risk'].sum()

col1, col2 = st.columns(2)
col1.metric("High-Value Users", high_value_count)
col2.metric("Churn Risk Users", churn_risk_count)

# Show aggregated traits
st.markdown("""### Top Traits of High-Value Users""")
high_value_traits = filtered_df[filtered_df['High_Value']].groupby(
    ['Device_Type', 'Subscription_Tier', 'Preferred_Game_Mode']
).size().reset_index(name='Count').sort_values("Count", ascending=False).head(5)
st.dataframe(high_value_traits)

st.markdown("""### Common Traits of Churn Risk Users""")
churn_traits = filtered_df[filtered_df['Churn_Risk']].groupby(
    ['Device_Type', 'Subscription_Tier', 'Preferred_Game_Mode']
).size().reset_index(name='Count').sort_values("Count", ascending=False).head(5)
st.dataframe(churn_traits)





# 🧪 Bonus Insights

st.subheader("🧪 Bonus: Advanced Behavioral Analysis")

# Cohort Analysis by Signup Month
st.markdown("### 📆 Cohort Analysis")
df['Signup_Month'] = df['Signup_Date'].dt.to_period('M').astype(str)
cohort = (
    df.groupby(['Signup_Month', 'Month'])['User_ID']
    .nunique()
    .reset_index(name='Active_Users')
)
fig_cohort = px.line(cohort, x='Month', y='Active_Users', color='Signup_Month', title="Monthly Activity by Signup Cohort")
st.plotly_chart(fig_cohort)

# Funnel Tracking
st.markdown("### 🔁 Funnel: Onboarding → First Game → Repeat Session")
df['Played_First_Game'] = df['Total_Play_Sessions'] > 0
df['Repeat_Player'] = df['Total_Play_Sessions'] > 1
funnel_data = {
    "Signed Up": len(df),
    "Played First Game": df['Played_First_Game'].sum(),
    "Repeat Player": df['Repeat_Player'].sum()
}
funnel_df = pd.DataFrame(list(funnel_data.items()), columns=["Stage", "Users"])
fig_funnel = px.funnel(funnel_df, x="Users", y="Stage", title="User Engagement Funnel")
st.plotly_chart(fig_funnel)

# User Segmentation with Clustering
st.markdown("### 🧠 User Segmentation (Clustering)")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

cluster_df = df[['Total_Play_Sessions', 'Total_Revenue_USD']].dropna()
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_df)

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(cluster_scaled)

cluster_df['Segment'] = cluster_labels
fig_cluster = px.scatter(
    cluster_df,
    x='Total_Play_Sessions',
    y='Total_Revenue_USD',
    color='Segment',
    title="User Segmentation by Sessions vs Revenue",
    labels={"Total_Play_Sessions": "Play Sessions", "Total_Revenue_USD": "Revenue"}
)
st.plotly_chart(fig_cluster)
# Dashboard Summary: Key Insights (Summary Section)
st.subheader("📊 Summary Insights")


st.markdown("""
### 🔍 Behavioral Patterns
- Most activity clusters in May, with daily/weekly usage spikes.
- Console users and Multiplayer mode show highest engagement and revenue.

### ⚠️ Early Signs of Churn
- Users with ≤ 2 sessions or < 5 min average duration are flagged.
- Common among free tier, mobile, and solo players.

### 💎 High-Value User Traits
- Tend to use Console, play Multiplayer, and subscribe to Premium/Pro tiers.
- Contribute disproportionately to revenue and have longer sessions.

### 💡 Recommendations
- Personalized offers based on session behavior.
- Re-engagement prompts for inactivity > 7 days.
- Loyalty rewards for consistent usage streaks.
""")




st.subheader("💡 Suggestions to Improve Retention and Revenue")

st.markdown("""
**📈 Revenue Improvement Strategies**
- Introduce dynamic pricing for premium features based on user segment behavior.
- Launch time-limited offers and bundles during peak usage periods.
- Incentivize multiplayer sessions with referral rewards or team bonuses.

**🔄 Retention Strategies**
- Use inactivity signals (e.g. session drop, play frequency) to trigger re-engagement campaigns.
- Personalize push/email notifications based on last played mode or favorite device.
- Implement progressive rewards for consistent logins (e.g., 7-day streak bonuses).

**🧠 Insights-Driven Ideas**
- Analyze churn-prone users' behavior and proactively offer tailored incentives.
- Track high-value users' feature adoption paths and replicate in onboarding flows.
""")

