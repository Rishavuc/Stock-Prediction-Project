# Stock-Prediction-Project
                                         
                                         ABSTRACT
Stock price prediction is a critical component of financial analysis and algorithmic trading, enabling investors and institutions to make informed decisions. This project explores the application of machine learning techniques, specifically Long Short-Term Memory (LSTM) neural networks, for predicting future stock prices based on historical data. The system is built using Python and Streamlit for interactive visualization and utilizes data from Yahoo Finance. Key indicators such as Relative Strength Index (RSI), Simple Moving Average (SMA), and trading volume are incorporated as features to improve prediction accuracy.
The model is trained using past closing prices and technical indicators, and evaluated using Mean Squared Error (MSE) as a performance metric. Graphs of actual vs predicted prices, loss curves, and future forecasts are presented to aid in analysis. This solution demonstrates how deep learning can effectively model sequential financial data and provides a foundation for building more complex decision-making systems in quantitative finance.
                                 
             		INTRODUCTION
In the world of finance, the stock market plays a big role in the economy of any country. Every day, millions of people buy and sell shares in the hope of making a profit. But predicting whether a stock’s price will go up or down is not an easy task. There are many factors that affect stock prices, like company news, global events, economic conditions, and even public sentiment on social media. Because of this, there is a need for advanced tools that can help people understand stock trends and make better decisions.
Stock prediction has become a popular topic in recent years, especially with the rise of artificial intelligence (AI) and machine learning. These technologies can study large amounts of past stock data and learn patterns that are not visible to the human eye. With the help of these smart systems, it is now possible to get fairly accurate predictions of future stock prices.
In this project, we have used a type of deep learning model called LSTM (Long Short-Term Memory) to predict stock prices. This model is very good at understanding sequences — like stock prices over time. We have also used the Streamlit framework to build a web-based app where users can type in a stock symbol, choose how many days of past data to use, and get predictions for future prices. The app also displays graphs, actual vs predicted prices, and whether the stock meets swing trading rules.
This project will be helpful for students, beginner investors, and anyone who is interested in the stock market. It combines the power of machine learning, real-time data, and a user-friendly interface to make stock prediction simple and easy to understand.
The idea behind this project is simple: What if a user could just type a stock name, click a button, and see predictions based on the latest available data? What if the app could also tell whether the stock is following any useful trading pattern like swing trading? That’s what we built using Python, Streamlit, LSTM, and Yahoo Finance data.
Problem Statements
•	The stock market is highly dynamic, with prices fluctuating rapidly based on various factors such as global events, economic indicators, and company performance. This makes it challenging for individual investors to make informed decisions regarding buying and selling stocks.
•	Existing stock prediction tools are often either paid services or overly complex for beginners. They require a strong understanding of technical indicators and financial modeling, making them inaccessible to many students and novice traders.
•	Many available platforms do not support real-time data fetching or fail to update stock prices automatically, limiting the reliability and accuracy of predictions.
•	The lack of user-friendly, open-source applications that combine modern machine learning techniques with up-to-date financial data creates a barrier for students and researchers who wish to explore stock market forecasting.
•	There is a clear need for an educational and practical project that not only performs future stock price predictions but also demonstrates data acquisition, preprocessing, model training, and prediction logic in an understandable and interactive way.
•	Furthermore, current basic prediction models do not include strategies such as swing trading logic or stock-specific analytics, which are essential for making more informed investment decisions.
Technologies Used:
1. Programming Language: Python
Why Python? Python is chosen due to its ease of use, extensive libraries, and frameworks for machine learning, data manipulation, and web development. It allows for rapid development and has robust community support.
Libraries and Frameworks Used:
•	pandas: For data manipulation and cleaning.
•	numpy: For numerical operations and matrix calculations.
•	yfinance: To fetch historical stock market data.
•	scikit-learn: For data preprocessing and scaling (e.g., MinMaxScaler).
•	keras/TensorFlow: For building and training the LSTM model.
•	matplotlib: For visualizing data and model predictions.
•	Streamlit: For building the user interface for the web application.
2. Data Collection and Preprocessing
Method:
•	yfinance API: The stock data is fetched using the yfinance Python library. This library allows users to download historical stock price data directly from Yahoo Finance.
Steps:
•	Data is fetched for a specific stock (e.g., Apple, Tesla) by specifying the ticker symbol and date range.
•	The fetched data typically includes columns such as Open, Close, High, Low, and Volume.
Example:
 
Data Preprocessing:
•	Cleaning Missing Data: Data gaps can be filled using interpolation methods or forward-fill.
•	Normalization: Stock data values, especially prices, are normalized to a specific range (0 to 1) to prevent scale issues when training the LSTM model. The MinMaxScaler is used for this purpose.
Example
 


3. Model Development: LSTM (Long Short-Term Memory)
LSTM is a type of Recurrent Neural Network (RNN) that is well-suited for time series forecasting because it is capable of remembering information for long periods, making it ideal for predicting stock prices that depend on historical trends.
LSTM Architecture:
•	Input Layer: The input consists of the previous time steps (e.g., previous 60 days' closing prices) to predict the next day’s closing price.
•	Hidden Layers: Multiple LSTM layers are stacked to capture complex patterns in the data.
•	Dropout Layers: Dropout layers are used for regularization to prevent overfitting.
•	Output Layer: A single unit that outputs the predicted stock price for the next day
Example:
 
Training the Model:
•	The model is trained on the training dataset with a chosen number of epochs (iterations over the full dataset) and a batch size.
•	Training is performed using the Adam optimizer, which is a common choice for gradient descent in deep learning.
Example:
 

User Interface: Streamlit
Streamlit is used for quickly creating interactive web applications with minimal code. It's highly suitable for deploying machine learning models and visualizing data interactively.
Streamlit App:
•	The app allows users to input a stock ticker (e.g., "AAPL" for Apple), select a time period for prediction, and view the stock price trends and predictions.
•	Users can interact with real-time predictions and see the model’s output immediately after entering the stock symbol.
Example:

 
System Architecture
The architecture of the stock price prediction app can be broken down into several key components:
1.	Data Collection: Stock price data is fetched from the yfinance API.
2.	Data Preprocessing: The fetched data undergoes cleaning, normalization, and splitting into training and testing sets.
3.	Model Development: The LSTM model is built, trained, and evaluated using the preprocessed data.
4.	Prediction: The trained LSTM model is used to predict future stock prices.
5.	Streamlit Web Interface: An interactive user interface is built using Streamlit to allow users to enter stock ticker symbols, select date ranges, and visualize the predictions.
6.	Real-Time Prediction: The model predicts stock prices based on the user’s input, and results are displayed graphically.





Flowchart
[Start]
|
[User Inputs Stock Ticker and Date Range]
|
[Fetch Stock Data from yfinance API]
|
[Data Preprocessing]
|-------> [Handle Missing Data (Forward Fill/Interpolation)]
|-------> [Normalization (MinMaxScaler)]
|-------> [Split Data (Train/Test)]
|
[Build and Train LSTM Model]
|
[Model Evaluation (MSE, R²)]
|
[Prediction (Next Day Stock Price)]
|
[Display Results in Streamlit Interface]
|-------> [Visualize Predicted and Actual Stock Prices]
|
[End]




Performance Evaluation
The performance of the stock price prediction model using Long Short-Term Memory (LSTM) is evaluated using two key metrics: Mean Squared Error (MSE) and R-squared (R²). These metrics provide insight into how well the model is predicting the stock prices and its ability to capture the underlying patterns in the data.
•	Mean Squared Error (MSE): MSE quantifies the average squared difference between the predicted and actual stock prices. A lower MSE indicates that the model's predictions are closer to the actual values, which implies better performance. In this project, the model's MSE was calculated for the test dataset, and the result indicates that the model performs reasonably well in terms of minimizing prediction errors.
•	R-squared (R²): The R² value measures the proportion of the variance in the actual stock prices that can be explained by the model. An R² value close to 1 indicates that the model has a high explanatory power, while a value closer to 0 suggests that the model is not capturing the data patterns effectively. For this model, the R² value shows a strong fit, meaning that the LSTM model successfully predicts stock prices based on historical data.
Graphical Representation of Predicted vs. Actual Stock Prices
To visually assess the performance of the model, a graphical comparison between the predicted and actual stock prices is presented. The graph plots the actual stock prices over time alongside the predicted stock prices. This allows us to easily observe how closely the predicted prices follow the actual price movements.
In the graph, the actual stock prices are represented by a solid line, while the predicted prices are shown by a dashed line. A good model will show the predicted values closely following the actual values, indicating accurate predictions. If there are significant deviations between the two lines, it suggests areas where the model may need further refinement.
By comparing the predicted and actual prices visually, we can identify any trends or patterns where the model excels or areas where improvements could be made, such as adjusting the model architecture, feature engineering, or hyperparameters.
Possible Future Improvements:
While the current model performs well, there are several avenues for future improvements:
1.	Data Enhancement:
o	Incorporating additional features, such as technical indicators (e.g., moving averages, Bollinger Bands), market sentiment data (e.g., from news or social media), and macroeconomic variables, could enhance the model's predictive power.
o	Increasing the dataset size and including data from multiple stocks or market sectors may improve the model’s generalization and robustness.
2.	Model Optimization:
o	Experimenting with more advanced LSTM variations, such as Bidirectional LSTM or GRU (Gated Recurrent Unit), may improve model performance by capturing more complex dependencies in the data.
o	Hyperparameter tuning (e.g., adjusting the number of layers, units per layer, or learning rate) could further optimize the model’s performance.
3.	Ensemble Methods:
o	Combining multiple models (e.g., LSTM with ARIMA or XGBoost) in an ensemble approach could help improve accuracy and reduce overfitting, providing more reliable predictions.
4.	Real-Time Prediction:
o	Integrating real-time stock price data and allowing the app to make predictions on the fly would increase the practical utility of the application for real-time stock market analysis.
5.	Model Interpretability:
o	Implementing model interpretability techniques (e.g., SHAP or LIME) could provide more transparency on how the model makes predictions, which could help users understand and trust the app's predictions better.
Data Acquisition
	Data Source - Yahoo Finance:
•	The stock price data is fetched from Yahoo Finance, a reliable and widely used financial data provider.
•	Yahoo Finance offers comprehensive historical and real-time stock market data, including key metrics like stock prices, volume, dividends, and stock splits.
•	The data is publicly available and can be accessed using APIs such as the yFinance API.
	Fetching Data Using yFinance API:
•	The yFinance API simplifies the process of downloading historical stock data into Python for analysis and model training.
•	The API allows access to various financial data points, such as opening, closing, high, low prices, adjusted close prices, and trading volume.
•	Data is fetched by specifying the stock symbol (e.g., "AAPL" for Apple, "TSLA" for Tesla) and a date range (e.g., from January 1, 2012, to the current date).
	Real-Time Data:
•	One of the key features of using the yFinance API is its ability to provide real-time data.
•	The stock data retrieved includes the most up-to-date prices, reflecting the latest trading activity.
•	This real-time feature ensures that the data used for prediction models is current, allowing for accurate and timely stock price forecasts.
	User Input for Stock Symbol:
•	The stock symbol is provided by the user of the application. The user can input any valid stock symbol to fetch data for that specific stock (e.g., "AAPL" for Apple or "TSLA" for Tesla).
•	This gives users the flexibility to analyze the stock data of different companies according to their needs.
	Customizable Date Range:
•	Users can customize the date range for which they want to download stock data, depending on their analysis requirements.
•	Whether the user needs data for the past year, five years, or a specific custom range, they can specify the start and end dates for the data retrieval.
•	This flexibility helps users focus on particular time frames that align with their analysis or prediction goals.
	Ease of Data Retrieval:
•	The process of fetching data is streamlined, making it easy to obtain large amounts of stock market data in a structured format (usually a Pandas DataFrame).
•	The simple integration of the yFinance API with Python makes it easy to incorporate in projects like stock price prediction apps.
	Data Update Frequency:
•	The yFinance API fetches the latest available trading day’s data, ensuring that the stock data reflects the most recent market movements.
•	By running the data fetching command periodically, the application can stay updated with real-time stock price information.
	Data Structure:
•	The data retrieved from Yahoo Finance typically includes multiple columns: Open, High, Low, Close, Adj Close (adjusted for splits/dividends), and Volume.
•	This structure allows for easy manipulation and preprocessing for machine learning tasks, such as stock price prediction.
	Importance for Model Accuracy:
•	Real-time and historical data are crucial for training accurate machine learning models, as they allow the model to learn from actual market behavior and trends.
•	The accuracy of predictions depends on using clean, up-to-date data that reflects the real-world stock price movements.
Interface Features
•	User inputs stock name (e.g., TSLA, SBIN.NS).
•	User selects prediction window (30–100 days).
•	Click 'Train Model & Predict' to begin.
•	Predictions are displayed in table format and line chart.
•	A separate section shows the next day's predicted price.
•	A swing trading strategy result is also provided at the end.

Code snippets:
 
 ![image](https://github.com/user-attachments/assets/10024f9a-62cd-4cef-b10e-a91f8e7fadce)

![image](https://github.com/user-attachments/assets/c8765d37-1c4e-43c2-9744-88c6f03bb0e5)

Our Project Output:

 ![image](https://github.com/user-attachments/assets/96885afa-49ed-43c5-9388-a8fff620aa92)

 ![image](https://github.com/user-attachments/assets/9d0e2dae-7acd-4194-9b3e-6727912d0bed)


CONCLUSION
In this project, we successfully implemented a stock price prediction system using LSTM neural networks, showcasing the potential of deep learning for modeling and forecasting financial time series data. The model was trained on historical stock prices and enhanced with technical indicators like RSI and SMA to capture market trends more effectively.
The results indicate that LSTM models can learn complex temporal dependencies in stock data, providing reasonably accurate predictions. While no model can perfectly forecast stock prices due to market volatility and external factors, this approach demonstrates a valuable tool for preliminary analysis and decision support in trading strategies. Future work may include integrating sentiment analysis, real-time data streams, and hybrid models to further enhance prediction accuracy and adaptability.

