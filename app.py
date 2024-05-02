import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Thêm các thư viện cần thiết cho LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input, Activation
from keras.models import load_model
from keras import optimizers
from keras.losses import MeanSquaredError
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import optimizers
import numpy as np

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), width=800))
colors=px.colors.qualitative.Plotly

# Kết nối tới cơ sở dữ liệu SQLite
engine = create_engine('sqlite:///stock_data.db')

# Tạo giao diện người dùng
st.title('Dự đoán giá cổ phiếu')

# Load dữ liệu từ cơ sở dữ liệu mỗi khi hàm được gọi
def load_data():
    return pd.read_sql('SELECT * FROM stock_data', engine)

def preprocess_data(df):
    # Tính toán chỉ số RSI
    df['RSI'] = ta.rsi(df.close, length=14)

    # Tính toán Trung bình Động Exponential
    df['EMAF'] = ta.ema(df.close, length=20)
    df['EMAM'] = ta.ema(df.close, length=100)
    df['EMAS'] = ta.ema(df.close, length=150)

    # Tính toán mục tiêu và lớp mục tiêu
    df['Target'] = df['close'] - df['open']
    df['Target'] = df['Target'].shift(-1)
    df['TargetClass'] = [1 if df['Target'].iloc[i] > 0 else 0 for i in range(len(df))]

    # Lấy giá đóng cửa tiếp theo
    df['TargetNextClose'] = df['close'].shift(-1)

    # Loại bỏ các cột không cần thiết và hàng chứa giá trị NaN
    df.dropna(inplace=True)
    df.drop(['open', 'high', 'low', 'volume'], axis=1, inplace=True)
    df.reset_index(inplace=True)

    # Chọn các cột cho tập dữ liệu cuối cùng
    data_set = df.iloc[:, 3:11]
    pd.set_option('display.max_columns', None)

    return data_set


# Widget cho phép người dùng tải lên file CSV
csv_file = st.file_uploader("Tải lên file CSV", type=['csv'])

if csv_file is not None:
    # Đọc dữ liệu từ file CSV
    df_new = pd.read_csv(csv_file)

    # Chuyển đổi kiểu dữ liệu của các cột
    df_new['date'] = pd.to_datetime(df_new['date'])  # Chuyển đổi sang kiểu datetime
    df_new['open'] = df_new['open'].astype(float)  # Chuyển đổi sang kiểu float
    df_new['high'] = df_new['high'].astype(float)  # Chuyển đổi sang kiểu float
    df_new['low'] = df_new['low'].astype(float)  # Chuyển đổi sang kiểu float
    df_new['close'] = df_new['close'].astype(float)  # Chuyển đổi sang kiểu float
    df_new['volume'] = df_new['volume'].astype(int)  # Chuyển đổi sang kiểu integer (hoặc bạn có thể sử dụng kiểu float nếu cần)

    # Hiển thị dữ liệu đã được tải lên
    st.subheader("Dữ liệu cổ phiếu từ file CSV:")
    st.write(df_new)

    # Nút để lưu dữ liệu vào cơ sở dữ liệu
    if st.button('Lưu vào cơ sở dữ liệu'):
        df_new.to_sql('stock_data', engine, if_exists='append', index=False)
        st.success('Dữ liệu đã được lưu vào cơ sở dữ liệu!')


# Combo box cho người dùng chọn ticker
if st.checkbox("Nhập dữ liệu"):
    ticker = st.selectbox('Chọn mã cổ phiếu:', ['MSN','FPT', 'VIC', 'PNJ'], key='select_tickerInput')

    # Nhập ngày
    date = st.date_input('Chọn ngày:', pd.Timestamp.now())

    # Nhập các thông tin về cổ phiếu
    open_price = st.number_input('Nhập giá mở cửa:')
    low_price = st.number_input('Nhập giá thấp nhất:')
    high_price = st.number_input('Nhập giá cao nhất:')
    close_price = st.number_input('Nhập giá đóng cửa:')
    volume = st.number_input('Nhập khối lượng giao dịch:')

    # Nút để lưu dữ liệu vào cơ sở dữ liệu
    if st.button('Lưu vào cơ sở dữ liệu', key='input_data'):
        data = {'ticker': ticker, 'date': date, 'open': open_price, 'low': low_price, 'high': high_price, 'close': close_price, 'volume': volume}
        df = pd.DataFrame(data, index=[0])
        df.to_sql('stock_data', engine, if_exists='append', index=False)
        st.success('Dữ liệu đã được lưu vào cơ sở dữ liệu!')


st.markdown("### *Hiển thị dữ liệu đã lưu*")
# Hiển thị dữ liệu
if st.checkbox('Hiển thị dữ liệu đã lưu' ):
    show_all = st.checkbox('Hiện tất cả')
    if show_all:
        df = load_data()
        st.subheader('Dữ liệu đã được lưu:')
        st.write(df)
    else:
        df = load_data()  # Load dữ liệu
        selected_ticker = st.selectbox('Chọn mã cổ phiếu:', ['MSN','FPT', 'VIC', 'PNJ'], key='selectbox_ticker')
        df_filtered = df[df['ticker'] == selected_ticker]
        st.subheader(f'Dữ liệu cổ phiếu cho mã {selected_ticker}:')
        st.write(df_filtered)


    # Chức năng chọn và xóa hàng dữ liệu không phù hợp
    if st.checkbox('chọn hàng để xóa'):
        rows_to_delete = st.multiselect('Chọn hàng để xóa:', df.index)
        if st.button('Xác nhận xóa'):
            df.drop(index=rows_to_delete, inplace=True)
            df.to_sql('stock_data', engine, if_exists='replace', index=False)
            st.success('Các hàng đã được xóa thành công!')

    # Chức năng xóa theo mã cổ phiếu
    if st.checkbox('Xóa theo mã'):
        # Chọn mã cổ phiếu
        selected_ticker_to_delete = st.selectbox('Chọn mã cổ phiếu để xóa dữ liệu:', ['MSN', 'FPT', 'VIC', 'PNJ'])

        # Tạo một bản sao của dữ liệu với mã cổ phiếu đã chọn
        df_to_delete = df[df['ticker'] == selected_ticker_to_delete]

        if st.button('Xác nhận xóa'):
            # Xóa dữ liệu tương ứng với mã cổ phiếu đã chọn
            df.drop(df_to_delete.index, inplace=True)
            df.to_sql('stock_data', engine, if_exists='replace', index=False)
            st.success(f'Dữ liệu cho mã cổ phiếu {selected_ticker_to_delete} đã được xóa thành công!')


    if st.button('Xóa tất cả dữ liệu'):
        df.drop(df.index, inplace=True)  # Drop all rows
        df.to_sql('stock_data', engine, if_exists='replace', index=False)
        st.success('Toàn bộ dữ liệu đã được xóa thành công!')


st.markdown("### *Phân tích thị trường*")
# Thêm mã code trực quan hóa dữ liệu   
if st.checkbox('Market Analysis'):
    df = load_data()  # Load dữ liệu
    selected_ticker = st.selectbox('Chọn mã cổ phiếu:', ['MSN','FPT', 'VIC', 'PNJ'], key='forVizualize')   
    df = df[df['ticker'] == selected_ticker]

    data_date = df.date.unique()
    df['volatility'] = df['close'].shift(-1) - df['close']
    df['volatility'] = df['volatility'].shift(1)

    returns = df.groupby('date')['volatility'].mean().rename('Average Return')
    close_avg = df.groupby('date')['close'].mean().rename('Closing Price')
    vol_avg = df.groupby('date')['volume'].mean().rename('Volume')

    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True)

    for i, j in enumerate([returns, close_avg, vol_avg]):
        fig.add_trace(go.Scatter(x=data_date, y=j, mode='lines',
                                name=j.name, marker_color=colors[i]), row=i+1, col=1)

    fig.update_xaxes(rangeslider_visible=False,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=2, label="2y", step="year", stepmode="backward"),
                            dict(step="all")])),             
                    row=1,col=1)

    fig.update_layout(template=temp, title='MSN Market Average Stock Return, Closing Price, and Shares Traded',
                    hovermode='x unified', height=700, width=900,
                    yaxis1=dict(title='Stock Return'),
                    yaxis2_title='Closing Price', yaxis3_title='Shares Traded',
                    showlegend=False)
    st.plotly_chart(fig)


st.markdown("### *Xây dựng model dự đoán*")
if st.checkbox('Training Model'):
    df = load_data()  # Load dữ liệu
    selected_ticker = st.selectbox('Chọn mã cổ phiếu:', ['MSN','FPT', 'VIC', 'PNJ'], key='for_train')
    df = df[df['ticker'] == selected_ticker]

    data_set = preprocess_data(df)
    st.write(data_set)

    #normalize data using MinMaxScaler (chuẩn hóa dữ liệu)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_set_scaled = scaler.fit_transform(data_set)

    X = []
    backcandles = 30
    for j in range(data_set_scaled[0].size):
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):
            X[j].append(data_set_scaled[i-backcandles:i, j])

    # Di chuyển trục từ vị trí 0 đến vị trí 2
    X=np.moveaxis(X, [0], [2])

    X, yi =np.array(X), np.array(data_set_scaled[backcandles:,-1])
    y=np.reshape(yi,(len(yi),1))

    # chia dữ liệu thành train và test
    splitlimit = int(len(X)*0.8)
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]


    if st.button('Train', key='train_btn'):
        #tf.random.set_seed(20)
        np.random.seed(10)

        lstm_input = Input(shape=(backcandles, 8), name='lstm_input')

        inputs = LSTM(150, name='first_layer')(lstm_input)
        inputs = Dense(1, name='dense_layer')(inputs)
        output = Activation('linear', name='output')(inputs)

        # Compile the model with MeanSquaredError
        model = Model(inputs=lstm_input, outputs=output)
        adam = optimizers.Adam()
        model.compile(optimizer=adam, loss=MeanSquaredError())

        # train model
        model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split=0.1)
        
        # Save the model
        model.save(f'{selected_ticker}_model.h5')

        # Load the model
        model = load_model(f'{selected_ticker}_model.h5')

        # Perform predictions
        y_pred = model.predict(X_test)
        fig = plt.figure(figsize=(12,6))
        plt.plot(y_test, color='black', label='Test')
        plt.plot(y_pred, color='green', label='Prediction')
        plt.legend()
        plt.title("Stock Price Prediction using Neural Network Model")
        st.pyplot(fig)

        # Đánh giá độ chính xác của mô hình
        loss_train = model.evaluate(X_train, y_train, verbose=0)
        loss_test = model.evaluate(X_test, y_test, verbose=0)

        st.write(f"Độ chính xác trên tập huấn luyện: {100-loss_train*100}%")
        st.write(f"Độ chính xác trên tập kiểm tra: {100-loss_test*100}%")

    if st.checkbox('Dự đoán giá cho nhiều ngày tiếp theo'):
        num_days_to_predict = st.number_input('Nhập số ngày dự đoán:', min_value=1, max_value=365, value=10)
        selected_ticker = st.selectbox('Chọn mã cổ phiếu:', ['MSN', 'FPT', 'VIC', 'PNJ'], key='for_pred')

        if st.button('Xác nhận'):
            df = load_data()  # Load dữ liệu
            df = df[df['ticker'] == selected_ticker]

            data_set = preprocess_data(df)
            model = load_model(f'{selected_ticker}_model.h5')
            latest_data = data_set.tail(30)  # Lấy dữ liệu từ 30 ngày trước đó

            # Chuẩn bị dữ liệu đầu vào cho mô hình
            latest_data_normalized = []

            # Khởi tạo MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))

            for column in range(latest_data.shape[1]):
                column_data = latest_data.iloc[:, column].values.reshape(-1, 1)
                # Fit scaler với dữ liệu của cột hiện tại
                scaler.fit(column_data)
                column_data_normalized = scaler.transform(column_data)
                latest_data_normalized.append(column_data_normalized)

            latest_data_normalized = np.array(latest_data_normalized).reshape(1, 30, -1)

            # Dự đoán giá cho số ngày tiếp theo
            predicted_prices = []

            for _ in range(num_days_to_predict):
                next_day_price_normalized = model.predict(latest_data_normalized)
                predicted_prices.append(next_day_price_normalized[0][0])

                # Cập nhật dữ liệu để dự đoán ngày tiếp theo
                latest_data_normalized = np.roll(latest_data_normalized, -1, axis=1)
                latest_data_normalized[0, -1, -1] = next_day_price_normalized

            # Chuyển đổi giá dự đoán về dạng ban đầu
            predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

            # Tạo dataframe chứa giá dự đoán cho số ngày tiếp theo
            df = load_data()
            prediction_dates = pd.date_range(start=df['date'].iloc[-1], periods=num_days_to_predict + 1, tz='UTC')[1:]
            predicted_df = pd.DataFrame({'Date': prediction_dates, 'Predicted_Close': predicted_prices.flatten()})
            predicted_df.set_index('Date', inplace=True)

            # Display the DataFrame containing predicted prices
            st.write(predicted_df)
