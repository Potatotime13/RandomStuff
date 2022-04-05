import tensorflow as tf
from Models.CModels import DTML
from Environments.TradingEnvs import TradeEnv

### create trading environment to get market data and to test the model ###
Tenv = TradeEnv(0.0, 0.75)

### load timeseries data using the environment ###
data, mid, num_stocks = Tenv.load_data()

### build the DTML model described in the paper ###
hidden_representation = 64
attention_heads = 4
drop_out = 0.1
observed_days = 10
model = DTML(num_stocks,hidden_representation,attention_heads,drop_out,observed_days)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.FalsePositives()])

### tranform data using the model function, the output is a normalized vector of 11 features per timestep ###
train_in, train_out, test_in, test_out, prices_trade, prices_full = model.generate_data(data,mid)

### pretrain model before the evaluation in the environment ###
model.fit(train_in,train_out,batch_size=4, epochs=10)

### run trading environment to benchmark the model performance ###
result = Tenv.trading_simulation(model,2, test_in, test_out, prices_trade)
print(result)