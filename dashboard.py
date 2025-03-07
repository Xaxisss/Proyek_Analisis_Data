import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd
import altair as alt
from babel.numbers import format_currency
sns.set(style='dark')

def create_customers_city_df(df):
    customers_city_df = df.groupby(by="customer_city").customer_unique_id.nunique().sort_values(ascending=False).head().reset_index()
    return customers_city_df

def create_sellers_city_df(df):
    sellers_city_df = df.groupby(by="seller_city").seller_id.nunique().sort_values(ascending=False).head().reset_index()
    return sellers_city_df

def create_delivery_status_df(df):
    return df["delivery_status"].value_counts()

def create_top_sellers_order_df(df):
    top_sellers_order_df = df.groupby("seller_name")["total_orders"].sum().sort_values(ascending=False).head(10).reset_index()
    return top_sellers_order_df

def create_top_sellers_revenue_df(df):
    top_sellers_revenue_df = df.groupby("seller_name")["total_revenue"].sum().sort_values(ascending=False).head(10).reset_index()
    return top_sellers_revenue_df

def create_customers_distribution_df(df):
    customer_purchases_df = df.groupby("customer_unique_id")["order_id"].nunique()

    # Buat kategori baru
    bins = [0, 1, 7, float("inf")]
    labels = ["1x", "2-7x", ">8x"]

    customer_purchases_category = pd.cut(customer_purchases_df, bins=bins, labels=labels)
    customer_purchases_distribution_df = customer_purchases_category.value_counts().sort_index()
    
    return customer_purchases_distribution_df

def create_rfm(df):
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    reference_date = df['order_purchase_timestamp'].max()
    
    rfm = df.groupby("customer_unique_id").agg({
        "order_purchase_timestamp": lambda x: (reference_date - x.max()).days,
        "order_id": "count",
        "price": "sum"
    }).reset_index()
    
    rfm.columns = ["customer_unique_id", "Recency", "Frequency", "Monetary"]
    rfm = rfm.drop(columns=["customer_unique_id"])  # Hanya menyisakan kolom numerik
    rfm["Recency"] = pd.to_numeric(rfm["Recency"], errors="coerce")
    rfm["Frequency"] = pd.to_numeric(rfm["Frequency"], errors="coerce")
    rfm["Monetary"] = pd.to_numeric(rfm["Monetary"], errors="coerce")

    quantiles = rfm.quantile(q=[0.2, 0.4, 0.6, 0.8]).to_dict()
    
    def r_score(x, q):
        return 5 if x <= q[0.2] else 4 if x <= q[0.4] else 3 if x <= q[0.6] else 2 if x <= q[0.8] else 1
    
    def fm_score(x, q):
        return 1 if x <= q[0.2] else 2 if x <= q[0.4] else 3 if x <= q[0.6] else 4 if x <= q[0.8] else 5
    
    rfm = rfm.fillna(0)
    rfm["R_Score"] = rfm["Recency"].apply(lambda x: r_score(x, quantiles["Recency"]))
    rfm["F_Score"] = rfm["Frequency"].apply(lambda x: fm_score(x, quantiles["Frequency"]))
    rfm["M_Score"] = rfm["Monetary"].apply(lambda x: fm_score(x, quantiles["Monetary"]))
    
    # Konversi RFM Score ke string lalu integer
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str) + 
        rfm["F_Score"].astype(str) + 
        rfm["M_Score"].astype(str)
    ).astype(int)

    # Pilih beberapa skor unik untuk sumbu X
    unique_scores = sorted(rfm["RFM_Score"].unique())
    num_labels = min(15, len(unique_scores))  # Batasi jumlah label max 15
    selected_scores = np.linspace(unique_scores[0], unique_scores[-1], num_labels, dtype=int)

    return rfm, selected_scores
# URL file CSV dari GitHub
sellers_url = "https://media.githubusercontent.com/media/Xaxisss/Proyek_Analisis_Data/refs/heads/main/sellers_data.csv"
customers_url = "https://media.githubusercontent.com/media/Xaxisss/Proyek_Analisis_Data/refs/heads/main/customers_data.csv"

# Baca data dari URL
sellers_df = pd.read_csv(sellers_url)
customers_df = pd.read_csv(customers_url)

customers_df["order_purchase_timestamp"] = pd.to_datetime(customers_df["order_purchase_timestamp"])
customers_df["price"] = pd.to_numeric(customers_df["price"], errors="coerce")

with st.sidebar:
    st.image("dashboard\Free.png")
    customers_df["order_purchase_timestamp"] = pd.to_datetime(customers_df["order_purchase_timestamp"])
    sellers_df["order_purchase_timestamp"] = pd.to_datetime(sellers_df["order_purchase_timestamp"])
    
    min_date_customers, max_date_customers = customers_df['order_purchase_timestamp'].min(), customers_df['order_purchase_timestamp'].max()
    min_date_sellers, max_date_sellers = sellers_df['order_purchase_timestamp'].min(), sellers_df["order_purchase_timestamp"].max()
    
    start_date_customers, end_date_customers = st.date_input("Select Date Range for Customers", min_value=min_date_customers, max_value=max_date_customers, value=[min_date_customers, max_date_customers])
    start_date_sellers, end_date_sellers = st.date_input("Select Data Range for Sellers", min_value=min_date_sellers, max_value=max_date_sellers, value=[min_date_customers, max_date_customers])



filtered_customers_df = customers_df[(customers_df['order_purchase_timestamp'] >= pd.to_datetime(start_date_customers)) &
                                     (customers_df['order_purchase_timestamp'] <= pd.to_datetime(end_date_customers))]
filtered_sellers_df = sellers_df[(sellers_df['order_purchase_timestamp'] >= pd.to_datetime(start_date_sellers)) &
                                 (sellers_df["order_purchase_timestamp"] <= pd.to_datetime(end_date_sellers))]

customers_city_df = create_customers_city_df(filtered_customers_df)
sellers_city_df = create_sellers_city_df(filtered_sellers_df)
delivery_status_df = create_delivery_status_df(filtered_customers_df)
rfm, selected_scores = create_rfm(filtered_customers_df)  # Pastikan kita menangkap dua nilai
customers_distribution_df = create_customers_distribution_df(filtered_customers_df)

st.header("E-COMMERCE DASHBOARD:sparkles:")

st.subheader("Total Customers & Sellers")
col1, col2 = st.columns(2)

with col1:
    total_customers = filtered_customers_df["customer_unique_id"].nunique()
    st.metric("Total Customers:", total_customers)

with col2:
    total_sellers = filtered_sellers_df["seller_id"].nunique()
    st.metric("Total Sellers:", total_sellers)

fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(customers_city_df['customer_city'], customers_city_df['customer_unique_id'], color="#90CAF9", alpha=0.8)
ax.set_xlabel("City")
ax.set_ylabel("Total Customers")
ax.set_title("Customers Distribution")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(sellers_city_df['seller_city'], sellers_city_df['seller_id'], color="#90CAF9", alpha=0.8)
ax.set_xlabel("City")
ax.set_ylabel("Total Sellers")
ax.set_title("Sellers Distribution")
st.pyplot(fig)

st.subheader("Delivery_status")
col4, col5 = st.columns(2)
with col4:
    total_on_time = delivery_status_df.get("On time", 0)
    st.metric("Total Pengiriman On Time:", total_on_time)
with col5:
    total_late = delivery_status_df.get("Late", 0)
    st.metric("Total pengiriman Telat :", total_late)

fig, ax = plt.subplots(figsize=(8, 5))
ax.pie(delivery_status_df.values, labels=delivery_status_df.index, autopct='%1.1f%%', colors=["#66b3ff", "#99ff99", "#ffcc99"])
ax.set_title("Delivery Status Distribution")
st.pyplot(fig)

# Hitung Total Order dan Total Revenue
total_orders = filtered_sellers_df["total_orders"].sum()
total_revenue = filtered_sellers_df["total_revenue"].sum()

# Menampilkan Metric di 1 Baris (2 Kolom)
st.subheader("Total Orders & Total Revenue")
col4, col5 = st.columns(2)

with col4:
    st.metric("Total Orders", total_orders)

with col5:
    st.metric("Total Revenue", f"Rp {total_revenue:,.0f}")  # Format Rupiah

# Ambil data top 10 sellers
top_sellers_orders = create_top_sellers_order_df(filtered_sellers_df)
top_sellers_revenue = create_top_sellers_revenue_df(filtered_sellers_df)

# Buat figure dan axis untuk grafik
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 baris, 2 kolom

# Plot Top Sellers by Orders
axes[0].barh(top_sellers_orders["seller_name"], top_sellers_orders["total_orders"], color="#90CAF9", alpha=0.8)
axes[0].set_xlabel("Total Orders")
axes[0].set_ylabel("Seller Name")
axes[0].set_title("Top 10 Sellers by Total Orders")
axes[0].invert_yaxis()  # Seller dengan order terbanyak di atas

# Plot Top Sellers by Revenue (dibalik arah sumbu X, label Y di kanan)
axes[1].barh(top_sellers_revenue["seller_name"], top_sellers_revenue["total_revenue"].abs(), color="#FFAB91", alpha=0.8)
axes[1].set_xlabel("Total Revenue")
axes[1].set_title("Top 10 Sellers by Revenue")

# Invert Y agar urutan seller tetap sinkron
axes[1].invert_yaxis()
axes[1].invert_xaxis()  # Membalik arah sumbu X agar tidak bertabrakan

# Pindahkan label dan tick Y ke kanan
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set_ylabel("Seller Name", labelpad=20)  # Tambahkan label Y di posisi kanan

# Tampilkan plot di Streamlit
st.pyplot(fig)

st.subheader("Distribusi Pelanggan")
# Ambil jumlah masing-masing kategori
pelanggan_baru = customers_distribution_df.get("1x", 0)
pelanggan_biasa = customers_distribution_df.get("2-7x", 0)
pelanggan_loyal = customers_distribution_df.get(">8x", 0)

# Buat metric dalam 3 kolom
col6, col7, col8 = st.columns(3)

with col6:
    st.metric("Pelanggan Baru", pelanggan_baru)

with col7:
    st.metric("Pelanggan Biasa", pelanggan_biasa)

with col8:
    st.metric("Pelanggan Loyal", pelanggan_loyal)

# Buat figure untuk bar chart horizontal
fig, ax = plt.subplots(figsize=(10, 5))

# Buat bar chart horizontal
customers_distribution_df.plot(kind="barh", color=["#42A5F5", "#66BB6A", "#FF7043"], alpha=0.8, ax=ax)

# Atur label dan judul
ax.set_title("Distribusi Pelanggan Berdasarkan Jumlah Order")
ax.set_xlabel("Jumlah Pelanggan")
ax.set_ylabel("Kategori Order")

# Tambahkan nilai di samping bar
for index, value in enumerate(customers_distribution_df):
    ax.text(value + 50, index, str(value), va="center", fontsize=10)

# Tampilkan grafik di Streamlit
st.pyplot(fig)


st.subheader("RFM Parameters")
col9, col10, col11 = st.columns(3)

with col9:
    mean_recency = round(rfm["Recency"].mean(), 1)
    st.metric("Rata-rata Recency:", mean_recency)

with col10:  # Perbaiki di sini, sebelumnya salah pakai col5
    mean_frequency = round(rfm["Frequency"].mean(), 2)
    st.metric("Rata-rata Frequency :", mean_frequency)

with col11:
    mean_monetary = format_currency(round(rfm["Monetary"].mean(), 1), "MXN", locale="es_MX")  # Ganti ke Rupiah
    st.metric("Rata-rata Monetary :", mean_monetary)



fig, ax = plt.subplots(figsize=(12, 5))
sns.histplot(rfm, x="RFM_Score", bins=10, kde=True, ax=ax)

ax.set_xticks(selected_scores)  # Hanya tampilkan nilai yang dipilih
ax.set_xticklabels(selected_scores, rotation=45)  # Putar agar mudah dibaca
ax.set_title("Distribusi RFM Score")
ax.set_xlabel("RFM Score (Sampled)")
ax.set_ylabel("Jumlah Pelanggan")
st.pyplot(fig)

with st.expander("Apa itu RFM Score?dan bagaimana cara membacanya?🤔"):
    st.markdown("""
    **RFM Score** adalah metode untuk penghitungan skor masing masing parameter pada RFM, yaitu Recency,frequency,
    dan monetary.Penjelasan RFM sendiri adalah:
    
    - **Recency (R)** : ini menghitung kapan terakhir kali belanja bertransaksi
      - Jika skor = 5, artinya Baru saja bertransaksi  
      - Jika skor = 1, artinya sudah lama tidak bertransaksi  

    - **Frequency (F)** : ini adalah parameter untuk menghitung seberapa sering pelanggan bertransaksi
      - Jika skor = 5, maka artinya sering bertransaksi  
      - Jika skor = 1, maka artinya jarang bertransaksi  

    - **Monetary (M)** : Parameter untuk menghitung total nilai transaksi yang dilakukan pelanggan 
      - Jika skor = 5, artinya nilai transaksi yang dilakukan besar  
      - Skor skor = 1, artinya  nilai transaksi yang dilakukan kecil

    **Contoh Interpretasi RFM Score:**  
    - **555** → Pelanggan sangat aktif, sering berbelanja, dan belanja besar  
    - **511** → Baru saja berbelanja, tetapi jarang dan belanja sedikit  
    - **111** → Sudah lama tidak berbelanja, jarang, dan belanja kecil  

    **RFM_Score dibuat agar memudahkan pembacaan , sehingga bisa menarik kesimpulan lebih cepat
                dan bisa membantu bisnis berkembang!** 
    """)

st.caption('Copyright (c) latenturant 2025')
