# Mounir BENYAMINA / mounir.benyamina@edu.ece.fr / Student ID: 932356486

### Commands to launch the app:
# streamlit run Mounir_streamlit.py 

import streamlit as st
import pandas as pd
import yfinance as yf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DateType, DoubleType
from pyspark.sql.functions import lit, avg, col, desc, year, month, weekofyear
from pyspark.sql.window import Window
from pyspark.sql.functions import lag
from pyspark.sql.functions import when
from pyspark.sql.functions import date_format
from pyspark.sql.functions import expr
from pyspark.sql.functions import count
from pyspark.sql.functions import stddev


import os
import sys

# Set the PYSPARK_PYTHON environment variable to the current Python executable
# This ensures that PySpark workers use the same Python version as the current script.
os.environ["PYSPARK_PYTHON"] = sys.executable

# Set the PYSPARK_DRIVER_PYTHON environment variable to the current Python executable
# This ensures that the PySpark driver (the main Python process) uses the same Python version.
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


# Multi-page structure
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Data exploration", "Descriptive statistics and correlations", "Data preparation and performance metrics", "Moving average calculation", "Correlation between stocks", "Return rate calculation by periods", "Insight 1: Most consistent returns", "Insight 2: Highest trading volume", "Insight 3: Max and min closing prices", "Insight 4: Upward vs downward days", "Insight 5: Best day of the week for returns", "Insight 6: Seasonal performance", "Insight 7: Outlier detection", "Insight 8: Momentum analysis", "Conclusion"])

if page == "Introduction":
    # Title and introduction
    st.title("Big Data Frameworks Project")
    st.markdown(
        """
        ### Objective
        This project focuses on analyzing the historical performance of Nasdaq-listed tech stocks using Spark.
        By applying big data processing techniques, we aim to extract valuable insights for a potential trader or investor.

        **Key Goals:**
        - Explore and understand stock data for multiple companies (AAPL, MSFT, GOOGL).
        - Pre-process the data to calculate indicators such as moving averages and return rates.
        - Identify the stock with the highest return rate over specific periods.
        - Provide visualizations and business insights derived from the data.
        """
    )

    st.markdown("""### Student Information:
    Name: Mounir Benyamina  
    Student ID: 932356486  
    Mail: mounir.benyamina@edu.ece.fr
    """)

elif page == "Data exploration":
    # Data exploration
    st.header("Data exploration")
    st.markdown(
        """
        #### Objective:
        - Automatically download stock data for AAPL, MSFT, and GOOGL.
        - Load the data into Spark DataFrames for further processing.
        - Display the schema and a sample of rows to inspect the data structure.
        """
    )

    # Define function to download data
    @st.cache_data
    def download_stock_data():
        tickers = ["AAPL", "MSFT", "GOOGL"]
        dataframes = {}
        for ticker in tickers:
            data = yf.download(ticker, start="2020-01-01", end="2023-01-01", auto_adjust=False)
            data = data.reset_index()
            data['Date'] = data['Date'].dt.date  # Fix for DateType() issue
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                data[col] = data[col].astype(float)
            dataframes[ticker] = data
        return dataframes

    # Download data automatically
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = download_stock_data()
        st.success("Data downloaded successfully!")

    # Initialize Spark session
    @st.cache_resource
    def create_spark_session():
        return SparkSession.builder.appName("Stock Data Exploration").getOrCreate()

    spark = create_spark_session()

    # Define schema for data
    schema = StructType([
        StructField("Date", DateType(), True),
        StructField("Open", DoubleType(), True),
        StructField("High", DoubleType(), True),
        StructField("Low", DoubleType(), True),
        StructField("Close", DoubleType(), True),
        StructField("Adj Close", DoubleType(), True),
        StructField("Volume", DoubleType(), True),
    ])

    # Convert dataframes to Spark DataFrames
    spark_dataframes = {}
    for ticker, df in st.session_state.dataframes.items():
        sdf = spark.createDataFrame(df, schema=schema)
        sdf = sdf.withColumn("Ticker", lit(ticker))
        spark_dataframes[ticker] = sdf

    st.success("Data has been successfully loaded into Spark dataframes!")

    # Display a description of the schema
    st.markdown(
        """
        ### Dataframe schema description

        The dataset contains the following columns:

        - **Date** (DateType): The trading day.
        - **Open** (DoubleType): The price at market opening.
        - **High** (DoubleType): The highest price of the day.
        - **Low** (DoubleType): The lowest price of the day.
        - **Close** (DoubleType): The price at market closing.
        - **Adj Close** (DoubleType): The adjusted closing price, accounting for splits and dividends.
        - **Volume** (DoubleType): The number of shares traded on that day.
        """
    )

    st.markdown("---")

    # Display the first and the last rows
    row_count = st.slider("Select the number of rows to display", min_value=10, max_value=40, value=10, step=10)

    if st.button("Show the first and the last rows"):
        for ticker, sdf in spark_dataframes.items():
            st.markdown(f"### First {row_count} rows of {ticker}")
            first_rows = sdf.orderBy("Date").limit(row_count).toPandas()
            st.dataframe(first_rows)

            st.markdown(f"### Last {row_count} rows of {ticker}")
            last_rows = sdf.orderBy(desc("Date")).limit(row_count).toPandas()
            st.dataframe(last_rows)

    # Initialize Spark dataframes globally if not already defined
    if "spark_dataframes" not in st.session_state:
        spark_dataframes = {}
        for ticker, df in st.session_state.dataframes.items():
            sdf = spark.createDataFrame(df, schema=schema)
            sdf = sdf.withColumn("Ticker", lit(ticker))
            spark_dataframes[ticker] = sdf
        st.session_state.spark_dataframes = spark_dataframes
    else:
        spark_dataframes = st.session_state.spark_dataframes


elif page == "Descriptive statistics and correlations":
    # Descriptive statistics and correlations
    st.header("Descriptive statistics and correlations")

    st.markdown(
        """
        ### Objective:
        - Calculate descriptive statistics for each stock (mean, median, standard deviation, etc.).
        - Analyze correlations between the selected features across stocks.
        - Provide visualizations for an easy interpretation of results.
        """
    )

    # Select stock and features for correlation
    selected_ticker = st.selectbox("Select stock", options=list(st.session_state.spark_dataframes.keys()))
    sdf = st.session_state.spark_dataframes[selected_ticker]


    st.markdown(f"### Descriptive statistics for {selected_ticker}")
    descriptive_stats = sdf.describe().toPandas()
    st.dataframe(descriptive_stats)

    st.markdown(f"### Correlations for {selected_ticker}")
    numeric_columns = [field.name for field in sdf.schema.fields if isinstance(field.dataType, DoubleType)]
    selected_features = st.multiselect("Select features for correlation", options=numeric_columns, default=numeric_columns)

    if selected_features:
        correlation_values = []
        for i, col1 in enumerate(selected_features):
            row = []
            for col2 in selected_features:
                col1_escaped = f"`{col1}`"
                col2_escaped = f"`{col2}`"
                correlation = sdf.selectExpr(f"corr({col1_escaped}, {col2_escaped}) as correlation").collect()[0].correlation
                row.append(correlation)
            correlation_values.append(row)

        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        correlation_matrix = np.array(correlation_values)
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, xticklabels=selected_features, yticklabels=selected_features, cmap="coolwarm")
        st.pyplot(plt)

elif page == "Data preparation and performance metrics":

    # Data preparation and performance metrics
    st.header("Data preparation and performance metrics")

    st.markdown(
        """
        ### Objective:
        - Group data by different periods for aggregated metrics.
        - Identify the stock with the highest daily return.
        - Calculate average daily return for custom periods.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark dataframes are not initialized. Please visit 'Data exploration' first.")
        st.stop()

    # Select stock for processing
    selected_ticker = st.selectbox("Select stock for preprocessing", options=list(st.session_state.spark_dataframes.keys()))
    sdf = st.session_state.spark_dataframes[selected_ticker]

    # Calculate daily return before grouping
    window_spec = Window.partitionBy().orderBy("Date")
    sdf = sdf.withColumn("Daily Return", (col("Close") - lag("Close", 1).over(window_spec)) / lag("Close", 1).over(window_spec))

    # Calculate daily price change
    sdf = sdf.withColumn("Daily Price Change", col("Close") - lag("Close", 1).over(window_spec))

    # Calculate monthly price change
    monthly_window = Window.partitionBy().orderBy(month("Date"))
    sdf = sdf.withColumn("Monthly Price Change", col("Close") - lag("Close", 1).over(monthly_window))

    # Grouping by periods
    st.markdown("#### Average opening/closing prices group by time period")
    frequency = st.selectbox("Select the grouping frequency", ["Weekly", "Monthly", "Yearly", "All"])

    try:
        grouped_weekly = sdf.groupBy(weekofyear("Date").alias("Week")).agg(
            avg("Open").alias("Avg_Weekly_Open"),
            avg("Close").alias("Avg_Weekly_Close"),
            avg("Daily Return").alias("Avg_Weekly_Return"),
            avg("Daily Price Change").alias("Avg_Weekly_Change"))

        grouped_monthly = sdf.groupBy(month("Date").alias("Month")).agg(
            avg("Open").alias("Avg_Monthly_Open"),
            avg("Close").alias("Avg_Monthly_Close"),
            avg("Daily Return").alias("Avg_Monthly_Return"),
            avg("Monthly Price Change").alias("Avg_Monthly_Change"))

        grouped_yearly = sdf.groupBy(year("Date").alias("Year")).agg(
            avg("Open").alias("Avg_Yearly_Open"),
            avg("Close").alias("Avg_Yearly_Close"),
            avg("Daily Return").alias("Avg_Yearly_Return"))

        if frequency == "Weekly":
            st.dataframe(grouped_weekly.toPandas().head(10))
        elif frequency == "Monthly":
            st.dataframe(grouped_monthly.toPandas().head(10))
        elif frequency == "Yearly":
            st.dataframe(grouped_yearly.toPandas().head(10))
        elif frequency == "All":
            st.markdown("### Weekly Averages")
            st.dataframe(grouped_weekly.toPandas().head(10))

            st.markdown("### Monthly Averages")
            st.dataframe(grouped_monthly.toPandas().head(10))

            st.markdown("### Yearly Averages")
            st.dataframe(grouped_yearly.toPandas().head(10))
    except Exception as e:
        st.error(f"An error occurred during grouping: {e}")

    # Displaying the highest daily return
    st.markdown("#### Stock with the highest daily return")
    try:
        max_return = sdf.orderBy(desc("Daily Return")).limit(1).toPandas()
        st.write("Highest daily return:")
        st.dataframe(max_return)
    except Exception as e:
        st.error(f"An error occurred while calculating the highest daily return: {e}")

    # Displaying average daily return
    st.markdown("#### Average daily return")
    try:
        avg_weekly = grouped_weekly.agg(avg("Avg_Weekly_Return")).collect()[0][0]
        avg_monthly = grouped_monthly.agg(avg("Avg_Monthly_Return")).collect()[0][0]
        avg_yearly = grouped_yearly.agg(avg("Avg_Yearly_Return")).collect()[0][0]

        st.write(f"Average weekly Return: {avg_weekly}")
        st.write(f"Average monthly Return: {avg_monthly}")
        st.write(f"Average yearly Return: {avg_yearly}")
    except Exception as e:
        st.error(f"An error occurred while calculating average daily returns: {e}")


elif page == "Moving average calculation":

    # Moving average calculation
    st.header("Moving average calculation")

    st.markdown(
        """
        ### Objective:
        - Calculate simple moving averages (SMA) and exponential moving averages (EMA) for selected stocks.
        - Visualize moving averages alongside the closing price.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark dataframes are not initialized. Please visit 'Data exploration' first.")
        st.stop()

    # Select stock for moving average calculation
    selected_ticker = st.selectbox("Select the stock for the moving average", options=list(st.session_state.spark_dataframes.keys()))
    sdf = st.session_state.spark_dataframes[selected_ticker]

    # Moving average window size
    window_size = st.slider("Select the moving average window size", min_value=5, max_value=50, value=20)

    # Calculate simple moving average (SMA)
    moving_avg_window = Window.orderBy("Date").rowsBetween(-window_size + 1, 0)
    sdf = sdf.withColumn("SMA", avg("Close").over(moving_avg_window))

    # Calculate exponential moving average (EMA)
    # EMA uses a recursive formula, requiring a custom implementation
    alpha = 2 / (window_size + 1)
    sdf = sdf.withColumn("EMA", col("Close") * alpha + lag("Close", 1).over(Window.orderBy("Date")) * (1 - alpha))

    # Display moving averages
    st.markdown("#### Moving average results")
    try:
        st.dataframe(sdf.select("Date", "Close", "SMA", "EMA").orderBy("Date").toPandas().head(20))
    except Exception as e:
        st.error(f"An error occurred while calculating moving averages: {e}")

    # Plot the moving averages
    st.markdown("#### Visualization of the moving averages")
    try:
        import matplotlib.pyplot as plt

        sdf_pd = sdf.select("Date", "Close", "SMA", "EMA").orderBy("Date").toPandas()
        sdf_pd["Date"] = pd.to_datetime(sdf_pd["Date"])

        plt.figure(figsize=(12, 6))
        plt.plot(sdf_pd["Date"], sdf_pd["Close"], label="Close Price", color="blue")
        plt.plot(sdf_pd["Date"], sdf_pd["SMA"], label=f"SMA ({window_size})", color="orange")
        plt.plot(sdf_pd["Date"], sdf_pd["EMA"], label=f"EMA ({window_size})", color="green")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Moving averages")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"An error occurred while visualizing moving averages: {e}")

elif page == "Correlation between stocks":

    # Correlation between stocks
    st.header("Correlation between stocks")

    st.markdown(
        """
        ### Objective:
        - Analyze correlations between the selected features of multiple stocks.
        - Provide a heatmap to visualize the correlations.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark dataframes are not initialized. Please visit 'Data exploration' first.")
        st.stop()

    # Select stocks for correlation analysis
    st.markdown("#### Select stocks and features for correlation analysis")
    selected_tickers = st.multiselect("Select stocks", 
        options=list(st.session_state.spark_dataframes.keys()), 
        default=list(st.session_state.spark_dataframes.keys()))

    if len(selected_tickers) < 2:
        st.warning("Please select at least two stocks for correlation analysis.")
        st.stop()

    # Select feature for correlation
    selected_feature = st.selectbox("Select the feature to compare", 
        options=["Open", "Close", "Volume"])

    try:
        # Combine selected dataframes for the chosen feature
        combined_sdf = None
        for ticker in selected_tickers:
            sdf = st.session_state.spark_dataframes[ticker]
            sdf = sdf.select("Date", col(selected_feature).alias(f"{ticker}_{selected_feature}"))
            combined_sdf = sdf if combined_sdf is None else combined_sdf.join(sdf, on="Date", how="inner")

        # Convert to pandas for correlation calculation
        combined_pdf = combined_sdf.toPandas()
        combined_pdf["Date"] = pd.to_datetime(combined_pdf["Date"])
        combined_pdf = combined_pdf.set_index("Date")

        # Calculate correlation
        correlation_matrix = combined_pdf.corr()

        # Display correlation matrix
        st.markdown("#### Correlation matrix")
        st.dataframe(correlation_matrix)

        # Plot the heatmap
        st.markdown("#### Heatmap of correlations")
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while visualizing correlations: {e}")

    except Exception as e:
        st.error(f"An error occurred during correlation analysis: {e}")

elif page == "Return rate calculation by periods":

    # Return rate calculation by periods
    st.header("Return rate calculation by periods")

    st.markdown(
        """
        ### Objective:
        - Calculate the return rates for each stock over different periods (weekly, monthly, yearly).
        - Identify the stock with the best return rate for a given period.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark DataFrames are not initialized. Please visit 'Data Exploration' first.")
        st.stop()

    # Select the stock for the return rate calculation
    selected_ticker = st.selectbox("Select stock for return rate calculation", options=list(st.session_state.spark_dataframes.keys()))
    sdf = st.session_state.spark_dataframes[selected_ticker]

    # Calculate daily return
    window_spec = Window.partitionBy().orderBy("Date")
    sdf = sdf.withColumn("Daily Return",
        (col("Close") - lag("Close", 1).over(window_spec)) / lag("Close", 1).over(window_spec))

    # Grouping by periods for return rate
    st.markdown("#### Return rate group by time period")
    frequency = st.selectbox("Select the grouping frequency", ["Weekly", "Monthly", "Yearly", "All"])

    try:
        grouped_weekly = sdf.groupBy(weekofyear("Date").alias("Week")).agg(
            avg("Daily Return").alias("Avg_Weekly_Return"),
            (avg("Close") - avg("Open")).alias("Weekly_Return_Rate"))

        grouped_monthly = sdf.groupBy(month("Date").alias("Month")).agg(
            avg("Daily Return").alias("Avg_Monthly_Return"),
            (avg("Close") - avg("Open")).alias("Monthly_Return_Rate"))

        grouped_yearly = sdf.groupBy(year("Date").alias("Year")).agg(
            avg("Daily Return").alias("Avg_Yearly_Return"),
            (avg("Close") - avg("Open")).alias("Yearly_Return_Rate"))

        # Display grouped data
        if frequency == "Weekly":
            st.dataframe(grouped_weekly.toPandas().head(10))
        elif frequency == "Monthly":
            st.dataframe(grouped_monthly.toPandas().head(10))
        elif frequency == "Yearly":
            st.dataframe(grouped_yearly.toPandas().head(10))
        elif frequency == "All":
            st.markdown("### Weekly return rates")
            st.dataframe(grouped_weekly.toPandas().head(10))

            st.markdown("### Monthly return rates")
            st.dataframe(grouped_monthly.toPandas().head(10))

            st.markdown("### Yearly return rates")
            st.dataframe(grouped_yearly.toPandas().head(10))
    except Exception as e:
        st.error(f"An error occurred during return rate calculation: {e}")

    # Display overall return rates
    st.markdown("#### Overall return rates")
    try:
        avg_weekly_return_rate = grouped_weekly.agg(avg("Weekly_Return_Rate")).collect()[0][0]
        avg_monthly_return_rate = grouped_monthly.agg(avg("Monthly_Return_Rate")).collect()[0][0]
        avg_yearly_return_rate = grouped_yearly.agg(avg("Yearly_Return_Rate")).collect()[0][0]

        st.write(f"Average weekly return rate: {avg_weekly_return_rate}")
        st.write(f"Average monthly return rate: {avg_monthly_return_rate}")
        st.write(f"Average yearly return rate: {avg_yearly_return_rate}")
    except Exception as e:
        st.error(f"An error occurred while calculating overall return rates: {e}")

    # Identify the stock with the best return rate for a given period
    st.markdown("#### Identify the stock with the best return rate for a given period")
    selected_period = st.selectbox("Select the period type", ["Month", "Year"])
    selected_date = st.date_input("Select the start date", value=pd.Timestamp("2022-01-01"))

    try:
        # Combine all stocks into one dataframe
        combined_sdf = None
        for ticker, sdf in st.session_state.spark_dataframes.items():
            sdf = sdf.withColumn("Ticker", lit(ticker))
            sdf = sdf.withColumn("Return Rate", (col("Close") - col("Open")) / col("Open"))
            sdf = sdf.select("Date", "Ticker", "Return Rate")
            combined_sdf = sdf if combined_sdf is None else combined_sdf.union(sdf)

        # Filter by period
        if selected_period == "Month":
            filtered_sdf = combined_sdf.filter(month("Date") == selected_date.month)
        elif selected_period == "Year":
            filtered_sdf = combined_sdf.filter(year("Date") == selected_date.year)

        # Find the stock with the best Return Rate
        best_stock = filtered_sdf.orderBy(desc("Return Rate")).limit(1).toPandas()
        st.markdown("#### Stock with the best return rate")
        st.dataframe(best_stock)
    except Exception as e:
        st.error(f"An error occurred while identifying the best stock: {e}")

elif page == "Insight 1: Most consistent returns":

    # Most consistent returns
    st.header("Insight 1: Most consistent returns")

    st.markdown(
        """
        ### Objective:
        - Identify the stock with the most stable daily returns (lowest standard deviation).
        - Compare standard deviations across all stocks.
        - Visualize the distribution of daily returns for better understanding.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark dataframes are not initialized. Please visit 'Data exploration' first.")
        st.stop()

    # Select the time period for analysis
    st.markdown("#### Select the analysis period")
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2023-01-01"))

    try:
        result = []
        for ticker, sdf in st.session_state.spark_dataframes.items():
            # Filter data by selected date range
            sdf = sdf.filter((col("Date") >= lit(str(start_date))) & (col("Date") <= lit(str(end_date))))

            # Remove rows with null values
            sdf = sdf.dropna(subset=["Open", "Close"])

            # Calculate the Daily Return
            window_spec = Window.partitionBy().orderBy("Date")
            sdf = sdf.withColumn("Daily Return",
                (col("Close") - lag("Close", 1).over(window_spec)) / lag("Close", 1).over(window_spec)).dropna(subset=["Daily Return"])  # Drop rows where Daily Return is null

            # Calculate the standard deviation of daily returns
            stats = sdf.selectExpr("stddev(`Daily Return`) as std_dev").collect()[0]
            std_dev = stats["std_dev"] if stats["std_dev"] is not None else float("inf")  # Handle cases where std_dev is None
            result.append({"Ticker": ticker, "Standard Deviation": std_dev})

        # Convert results to pandas dataframe
        result_df = pd.DataFrame(result).sort_values(by="Standard Deviation")

        # Display Top 3 most stable stocks
        st.markdown("#### Top 3 most stable stocks")
        st.dataframe(result_df.head(3))

        # Visualize standard deviations for all stocks
        st.markdown("#### Standard deviations of daily returns for all stocks")
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.bar(result_df["Ticker"], result_df["Standard Deviation"], color="skyblue")
            plt.xlabel("Stock")
            plt.ylabel("Standard deviation")
            plt.title("Standard deviation of daily returns by stock")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while visualizing standard deviations: {e}")

        # Visualize histogram of daily returns for a selected stock
        st.markdown("#### Distribution of daily returns for a stock")
        selected_ticker = st.selectbox("Select the stock for the distribution analysis", result_df["Ticker"])
        selected_sdf = st.session_state.spark_dataframes[selected_ticker]
        selected_sdf = selected_sdf.filter((col("Date") >= lit(str(start_date))) & (col("Date") <= lit(str(end_date))))

        # Remove rows with null values
        selected_sdf = selected_sdf.dropna(subset=["Open", "Close"])

        # Calculate daily return
        selected_sdf = selected_sdf.withColumn("Daily Return",
            (col("Close") - lag("Close", 1).over(window_spec)) / lag("Close", 1).over(window_spec)).dropna(subset=["Daily Return"])  # Drop rows where Daily Return is null

        # Collect Data for Histogram
        daily_returns = selected_sdf.select("Daily Return").rdd.flatMap(lambda x: x).collect()
        plt.figure(figsize=(10, 6))
        plt.hist(daily_returns, bins=50, color="orange", edgecolor="black")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of daily returns for {selected_ticker}")
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred while analyzing consistent returns: {e}")


elif page == "Insight 2: Highest trading volume":

    # Insight 2: Stocks with the highest trading volume
    st.header("Insight 2: Stocks with the highest trading volume")

    st.markdown(
        """
        ### Objective:
        - Identify the stocks with the highest average trading volumes over different periods.
        - Compare trading volumes across stocks and visualize the trends.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark dataFrames are not initialized. Please visit 'Data exploration' first.")
        st.stop()

    # Select the time period
    st.markdown("#### Select the time period for the analysis")
    frequency = st.selectbox("Select the frequency", ["Daily", "Weekly", "Monthly", "Yearly"])

    try:
        result = []
        for ticker, sdf in st.session_state.spark_dataframes.items():
            if frequency == "Daily":
                avg_volume = sdf.agg(avg("Volume").alias("Avg_Daily_Volume")).collect()[0]["Avg_Daily_Volume"]
            elif frequency == "Weekly":
                avg_volume = sdf.groupBy(weekofyear("Date").alias("Week")).agg(avg("Volume").alias("Avg_Weekly_Volume")).agg(avg("Avg_Weekly_Volume")).collect()[0][0]
            elif frequency == "Monthly":
                avg_volume = sdf.groupBy(month("Date").alias("Month")).agg(avg("Volume").alias("Avg_Monthly_Volume")).agg(avg("Avg_Monthly_Volume")).collect()[0][0]
            elif frequency == "Yearly":
                avg_volume = sdf.groupBy(year("Date").alias("Year")).agg(avg("Volume").alias("Avg_Yearly_Volume")).agg(avg("Avg_Yearly_Volume")).collect()[0][0]
            
            result.append({"Ticker": ticker, "Average Volume": avg_volume})

        # Convert results to pandas DataFrame
        result_df = pd.DataFrame(result).sort_values(by="Average Volume", ascending=False)

        # Display the stock with highest Trading Volume
        highest_volume_stock = result_df.iloc[0]
        st.markdown("#### Stock with the highest trading volume")
        st.write(f"**{highest_volume_stock['Ticker']}** with an average volume of **{highest_volume_stock['Average Volume']:.0f}**")

        # Display average volumes for all stocks
        st.markdown("#### Average trading volumes for all the stocks")
        st.dataframe(result_df)

        # Visualize trading volumes
        st.markdown("#### Visualization of trading volumes")
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.bar(result_df["Ticker"], result_df["Average Volume"], color="purple")
            plt.xlabel("Stock")
            plt.ylabel("Average Volume")
            plt.title(f"Average trading volume by stock ({frequency})")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while visualizing trading volumes: {e}")

    except Exception as e:
        st.error(f"An error occurred while calculating trading volumes: {e}")

elif page == "Insight 3: Max and min closing prices":

    # Insight 3: Maximum and Minimum closing prices
    st.header("Insight 3: Maximum and minimum closing prices")

    st.markdown(
        """
        ### Objective:
        - Identify the maximum and minimum closing prices for each stock.
        - Provide the corresponding dates for these records.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark dataframes are not initialized. Please visit 'Data exploration' first.")
        st.stop()

    try:
        result = []
        for ticker, sdf in st.session_state.spark_dataframes.items():
            # Calculate Max closing price
            max_row = sdf.orderBy(desc("Close")).select("Date", "Close").limit(1).collect()[0]
            max_close = max_row["Close"]
            max_date = max_row["Date"]

            # Calculate Min closing price
            min_row = sdf.orderBy("Close").select("Date", "Close").limit(1).collect()[0]
            min_close = min_row["Close"]
            min_date = min_row["Date"]

            result.append({
                "Ticker": ticker,
                "Max Close": max_close,
                "Max Close Date": max_date,
                "Min Close": min_close,
                "Min Close Date": min_date})

        # Convert results to pandas dataframe
        result_df = pd.DataFrame(result)

        # Display results
        st.markdown("#### Maximum and minimum closing prices for all stocks")
        st.dataframe(result_df)

        # Visualize max and min closing prices
        st.markdown("#### Visualization of the maximum and the minimum closing prices")
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.bar(result_df["Ticker"], result_df["Max Close"], label="Max Close", color="green")
            plt.bar(result_df["Ticker"], result_df["Min Close"], label="Min Close", color="red", alpha=0.7)
            plt.xlabel("Stock")
            plt.ylabel("Closing Price")
            plt.title("Maximum and Minimum closing prices by stock")
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while visualizing max and min prices: {e}")

    except Exception as e:
        st.error(f"An error occurred while calculating max and min prices: {e}")

elif page == "Insight 4: Upward vs downward days":

    # Insight 4: Upward vs downward days
    st.header("Insight 4: Upward vs downward days")

    st.markdown(
        """
        ### Objective:
        - Analyze the number of upward and downward days for each stock.
        - Provide a breakdown of these trends to identify the overall market sentiment for each stock.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark dataframes are not initialized. Please visit 'Data exploration' first.")
        st.stop()

    try:
        result = []
        for ticker, sdf in st.session_state.spark_dataframes.items():
            # Add a column to classify upward vs downward days
            sdf = sdf.withColumn("Trend",when(col("Close") > col("Open"), "Upward").when(col("Close") < col("Open"), "Downward").otherwise("Neutral"))

            # Count upward and downward days
            trend_counts = sdf.groupBy("Trend").count().toPandas()
            upward_days = trend_counts[trend_counts["Trend"] == "Upward"]["count"].values[0] if "Upward" in trend_counts["Trend"].values else 0
            downward_days = trend_counts[trend_counts["Trend"] == "Downward"]["count"].values[0] if "Downward" in trend_counts["Trend"].values else 0
            neutral_days = trend_counts[trend_counts["Trend"] == "Neutral"]["count"].values[0] if "Neutral" in trend_counts["Trend"].values else 0

            result.append({
                "Ticker": ticker,
                "Upward Days": upward_days,
                "Downward Days": downward_days,
                "Neutral Days": neutral_days})

        # Convert results to pandas dataframe
        result_df = pd.DataFrame(result)

        # Display results
        st.markdown("#### Upward and downward days for all the stocks")
        st.dataframe(result_df)

        # Visualize upward vs downward Days
        st.markdown("#### Visualization of upward and downward days")
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            ind = np.arange(len(result_df))  # the x locations for the groups
            width = 0.3  # the width of the bars

            plt.figure(figsize=(12, 6))
            plt.bar(ind - width, result_df["Upward Days"], width, label="Upward Days", color="green")
            plt.bar(ind, result_df["Downward Days"], width, label="Downward Days", color="red")
            plt.bar(ind + width, result_df["Neutral Days"], width, label="Neutral Days", color="blue", alpha=0.7)
            plt.xlabel("Stock")
            plt.ylabel("Number of days")
            plt.title("Upward vs downward days by stock")
            plt.xticks(ind, result_df["Ticker"])
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while visualizing trends: {e}")

    except Exception as e:
        st.error(f"An error occurred while analyzing upward vs downward days: {e}")

elif page == "Insight 5: Best day of the week for returns":

    # Insight 5: Best day of the week for returns
    st.header("Insight 5: Best day of the week for returns")

    st.markdown(
        """
        ### Objective:
        - Identify the best day of the week for average daily returns for each stock.
        - Provide insights into weekly trends in stock performance.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark dataframes are not initialized. Please visit 'Data exploration' first.")
        st.stop()

    try:
        result = []
        all_avg_returns = {}

        for ticker, sdf in st.session_state.spark_dataframes.items():
            # Calculate Daily Return
            window_spec = Window.partitionBy().orderBy("Date")
            sdf = sdf.withColumn("Daily Return",
                (col("Close") - lag("Close", 1).over(window_spec)) / lag("Close", 1).over(window_spec))

            # Add a column for the day of the week
            sdf = sdf.withColumn("Day of Week", date_format(col("Date"), "EEEE"))

            # Filter null daily returns
            sdf = sdf.dropna(subset=["Daily Return"])

            # Calculate average return per day of the week
            avg_returns = sdf.groupBy("Day of Week").agg(avg("Daily Return").alias("Avg Return"))
            avg_returns_pd = avg_returns.toPandas()

            # Ensure days of the week are sorted properly
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            avg_returns_pd["Day of Week"] = pd.Categorical(avg_returns_pd["Day of Week"], categories=day_order, ordered=True)
            avg_returns_pd = avg_returns_pd.sort_values("Day of Week")

            # Save for visualization
            all_avg_returns[ticker] = avg_returns_pd

            # Find the best day
            best_day = avg_returns_pd.loc[avg_returns_pd["Avg Return"].idxmax()]
            result.append({
                "Ticker": ticker,
                "Best Day": best_day["Day of Week"],
                "Average Return": best_day["Avg Return"]})

        # Convert results to pandas dataframe
        result_df = pd.DataFrame(result)

        # Display thz results
        st.markdown("#### Best day of the week for all the tocks")
        st.dataframe(result_df)

        # Visualize average returns for all the stocks
        st.markdown("#### Visualization of the average returns by day of the week")
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            for ticker, avg_returns_pd in all_avg_returns.items():
                plt.plot(avg_returns_pd["Day of Week"], avg_returns_pd["Avg Return"], label=ticker, marker="o")

            plt.xlabel("Day of the week")
            plt.ylabel("Average return")
            plt.title("Average returns by day of the week for each stock")
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while visualizing average returns: {e}")

        # Detailed distribution for selected stock
        st.markdown("#### Detailed daily returns by day of the week for a stock")
        selected_ticker = st.selectbox("Select the stock", options=result_df["Ticker"])
        selected_avg_returns = all_avg_returns[selected_ticker]

        st.bar_chart(selected_avg_returns.set_index("Day of Week")["Avg Return"])

    except Exception as e:
        st.error(f"An error occurred while calculating the best day of the week for returns: {e}")


elif page == "Insight 6: Seasonal performance":

    # Insight 6: Seasonal performance
    st.header("Insight 6: Seasonal performance")

    st.markdown(
        """
        ### Objective:
        - Analyze the average performance of stocks by season (Winter, Spring, Summer, Autumn).
        - Identify any seasonal trends in stock performance.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark dataframes are not initialized. Please visit 'Data exploration' first.")
        st.stop()

    # Allow the user to select a time range
    st.markdown("#### Select the analysis period")
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2023-01-01"))

    try:
        result = []
        for ticker, sdf in st.session_state.spark_dataframes.items():
            # Filter data by selected date range
            sdf = sdf.filter((col("Date") >= lit(str(start_date))) & (col("Date") <= lit(str(end_date))))

            # Add a column to classify seasons
            sdf = sdf.withColumn("Season",
                when((month("Date").isin(12, 1, 2)), "Winter")
                .when((month("Date").isin(3, 4, 5)), "Spring")
                .when((month("Date").isin(6, 7, 8)), "Summer")
                .when((month("Date").isin(9, 10, 11)), "Autumn"))

            # Calculate average closing price by season
            seasonal_avg = sdf.groupBy("Season").agg(avg("Close").alias("Avg Closing Price"))
            seasonal_avg_pd = seasonal_avg.orderBy("Season").toPandas()

            # Ensure seasons are sorted
            season_order = ["Winter", "Spring", "Summer", "Autumn"]
            seasonal_avg_pd["Season"] = pd.Categorical(seasonal_avg_pd["Season"], categories=season_order, ordered=True)
            seasonal_avg_pd = seasonal_avg_pd.sort_values("Season")

            # Add results to the final dataframe
            result.append({
                "Ticker": ticker,
                "Winter Avg Close": seasonal_avg_pd[seasonal_avg_pd["Season"] == "Winter"]["Avg Closing Price"].values[0] if "Winter" in seasonal_avg_pd["Season"].values else None,
                "Spring Avg Close": seasonal_avg_pd[seasonal_avg_pd["Season"] == "Spring"]["Avg Closing Price"].values[0] if "Spring" in seasonal_avg_pd["Season"].values else None,
                "Summer Avg Close": seasonal_avg_pd[seasonal_avg_pd["Season"] == "Summer"]["Avg Closing Price"].values[0] if "Summer" in seasonal_avg_pd["Season"].values else None,
                "Autumn Avg Close": seasonal_avg_pd[seasonal_avg_pd["Season"] == "Autumn"]["Avg Closing Price"].values[0] if "Autumn" in seasonal_avg_pd["Season"].values else None})

        # Convert results to pandas dataframe
        result_df = pd.DataFrame(result)

        # Display results
        st.markdown("#### Average seasonal performance for all the stocks")
        st.dataframe(result_df)

        # Visualize seasonal performance
        st.markdown("#### Visualization of seasonal performance")
        try:
            import matplotlib.pyplot as plt

            seasons = ["Winter", "Spring", "Summer", "Autumn"]
            for index, row in result_df.iterrows():
                plt.plot(seasons,[row["Winter Avg Close"], row["Spring Avg Close"], row["Summer Avg Close"], row["Autumn Avg Close"]],
                    label=row["Ticker"],
                    marker="o")

            plt.xlabel("Season")
            plt.ylabel("Average closing price")
            plt.title("Average seasonal performance for each stock")
            plt.legend()
            plt.grid()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while visualizing seasonal performance: {e}")

    except Exception as e:
        st.error(f"An error occurred while analyzing seasonal performance: {e}")


elif page == "Insight 7: Outlier detection":

    # Insight 7: Outlier detection in daily returns
    st.header("Insight 7: Outlier detection in daily returns")

    st.markdown(
        """
        ### Objective:
        - Detect outliers in daily returns using statistical thresholds (e.g., values beyond 3 standard deviations).
        - Highlight unusual days for further analysis.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark dataframes are not initialized. Please visit 'Data exploration' first.")
        st.stop()

    try:
        result = []
        for ticker, sdf in st.session_state.spark_dataframes.items():
            # Calculate Daily Return
            window_spec = Window.partitionBy().orderBy("Date")
            sdf = sdf.withColumn("Daily Return",
                (col("Close") - lag("Close", 1).over(window_spec)) / lag("Close", 1).over(window_spec))

            # Calculate mean and standard deviation
            stats = sdf.selectExpr("mean(`Daily Return`) as mean", "stddev(`Daily Return`) as stddev").collect()[0]
            mean_return = stats["mean"]
            stddev_return = stats["stddev"]

            # Filter outliers
            outliers = sdf.filter((col("Daily Return") > mean_return + 3 * stddev_return) |(col("Daily Return") < mean_return - 3 * stddev_return)
            ).select("Date", "Daily Return").orderBy("Date").toPandas()

            result.append({ "Ticker": ticker, "Number of Outliers": len(outliers), "Outliers": outliers})

        # Convert results to pandas dataframe
        summary_df = pd.DataFrame({ "Ticker": [r["Ticker"] for r in result], "Number of Outliers": [r["Number of Outliers"] for r in result]})

        # Display summary
        st.markdown("#### Outlier summary for all the stocks")
        st.dataframe(summary_df)

        # Display outliers for selected stock
        st.markdown("#### Detailed outliers for a stock")
        selected_ticker = st.selectbox("Select the stock to view outliers", summary_df["Ticker"])
        selected_outliers = next(r["Outliers"] for r in result if r["Ticker"] == selected_ticker)

        if not selected_outliers.empty:
            st.dataframe(selected_outliers)
        else:
            st.write("No outliers detected for this stock.")

        # Visualize outliers
        st.markdown("#### Visualization of outliers")
        try:
            import matplotlib.pyplot as plt

            if not selected_outliers.empty:
                plt.figure(figsize=(12, 6))
                plt.scatter(selected_outliers["Date"], selected_outliers["Daily Return"], color="red", label="Outliers")
                plt.axhline(y=mean_return, color="green", linestyle="--", label="Mean Return")
                plt.axhline(y=mean_return + 3 * stddev_return, color="blue", linestyle="--", label="Upper Threshold (+3σ)")
                plt.axhline(y=mean_return - 3 * stddev_return, color="blue", linestyle="--", label="Lower Threshold (-3σ)")
                plt.xlabel("Date")
                plt.ylabel("Daily return")
                plt.title(f"Outliers in daily returns for {selected_ticker}")
                plt.legend()
                st.pyplot(plt)
            else:
                st.write("No data to visualize.")
        except Exception as e:
            st.error(f"An error occurred while visualizing outliers: {e}")

    except Exception as e:
        st.error(f"An error occurred while detecting outliers: {e}")

elif page == "Insight 8: Momentum analysis":

    # Insight 8: Momentum analysis - Consecutive Gains/Losses
    st.header("Insight 8: Momentum analysis - consecutive gains/losses")

    st.markdown(
        """
        ### Objective:
        - Identify the longest streaks of consecutive gains or losses for each stock.
        - Provide insights into the persistence of trends in stock performance.
        """
    )

    # Ensure Spark dataframes are initialized
    if "spark_dataframes" not in st.session_state:
        st.error("Spark dataframes are not initialized. Please visit 'Data exploration' first.")
        st.stop()

    try:
        result = []
        for ticker, sdf in st.session_state.spark_dataframes.items():
            # Add a column to classify gains and losses
            sdf = sdf.withColumn("Gain/Loss", when(col("Close") > col("Open"), 1).when(col("Close") < col("Open"), -1).otherwise(0))

            # Add a column to mark streak changes
            sdf = sdf.withColumn("Streak_Change", (col("Gain/Loss") != lag("Gain/Loss", 1).over(Window.orderBy("Date"))).cast("int"))

            # Create a unique ID for streaks
            streak_window = Window.orderBy("Date").rowsBetween(Window.unboundedPreceding, 0)
            sdf = sdf.withColumn("Streak_ID", expr("sum(Streak_Change) OVER (ORDER BY Date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"))

            # Group by streak ID and calculate streak lengths
            streak_df = sdf.groupBy("Streak_ID", "Gain/Loss").agg(count("*").alias("count")).filter(col("Gain/Loss") != 0)

            # Find the longest streaks for gains and losses
            longest_gain_streak = (streak_df.filter(col("Gain/Loss") == 1).orderBy(desc("count")).limit(1).collect())
            longest_loss_streak = (streak_df.filter(col("Gain/Loss") == -1).orderBy(desc("count")).limit(1).collect())

            # Append results
            result.append({"Ticker": ticker,
                "Longest Gain Streak": longest_gain_streak[0]["count"] if longest_gain_streak else 0,
                "Longest Loss Streak": longest_loss_streak[0]["count"] if longest_loss_streak else 0})

        # Convert results to pandas dataframe
        result_df = pd.DataFrame(result)

        # Display results
        st.markdown("#### Longest streaks of gains and losses for all the stocks")
        st.dataframe(result_df)

        # Visualize momentum analysis
        st.markdown("#### Visualization of the longest streaks")
        try:
            import matplotlib.pyplot as plt

            labels = result_df["Ticker"]
            gains = result_df["Longest Gain Streak"]
            losses = result_df["Longest Loss Streak"]

            x = range(len(labels))

            plt.figure(figsize=(12, 6))
            plt.bar(x, gains, width=0.4, label="Longest Gain Streak", color="green", align="center")
            plt.bar(x, losses, width=0.4, label="Longest Loss Streak", color="red", align="edge")
            plt.xlabel("Stocks")
            plt.ylabel("Number of days")
            plt.title("Longest streaks of gains and losses")
            plt.xticks(x, labels, rotation=45)
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while visualizing momentum analysis: {e}")

    except Exception as e:
        st.error(f"An error occurred while performing momentum analysis: {e}")


elif page == "Conclusion":

    # Conclusion page
    st.header("Summary of the application")

    st.markdown(
        """
        This Streamlit application provides a comprehensive platform for analyzing NASDAQ stock data, specifically focusing on leading technology companies such as Apple, Microsoft, and Google. Designed to leverage the computational power of PySpark, the application handles large datasets efficiently, ensuring seamless data processing and analysis.

        The journey begins with an automated data download feature that retrieves historical stock data for the selected companies. The data is then loaded into Spark DataFrames, enabling users to explore and validate its structure through schema descriptions and sampling functionalities. Users can inspect data intervals, count observations, and review subsets of the data, ensuring a deep understanding of its composition and quality.

        Data preprocessing capabilities include calculating average prices across different time periods (daily, weekly, monthly, yearly), deriving daily returns, and grouping data for targeted analyses. The application also identifies significant patterns, such as trends in closing prices and seasonal performance, providing users with actionable insights into stock behavior over time.

        One of the standout features of the application is its ability to identify correlations between different stocks, helping investors make informed diversification decisions. Additionally, the app employs outlier detection to highlight abnormal stock movements, providing users with the tools to evaluate unusual market behavior critically. These insights are complemented by visualizations, including line charts, heatmaps, and scatter plots, which make the data both interpretable and actionable.

        Furthermore, the insights section delivers unique perspectives, such as determining the best days for returns, analyzing consistent stock performance, and understanding the impact of seasonal trends on stock prices. These findings are presented interactively, allowing users to explore data dynamically and tailor the analysis to their specific needs.

        This application culminates in a powerful blend of exploratory, analytical, and visualization tools, serving as an indispensable resource for traders, investors, and researchers. By combining the scalability of PySpark with an intuitive Streamlit interface, the app bridges the gap between raw data and actionable market insights. Whether you are identifying top-performing stocks, analyzing return volatility, or discovering market patterns, this application empowers users to navigate the complexities of stock market analysis with confidence and precision.
        """
    )


































    


    













