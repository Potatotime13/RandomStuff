import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class LSTM_Attention(layers.Layer):
    '''
    -Combining the lstm hidden states h per stock along the time series using the formula (2)
    -INPUT is [batches,model_d,time]
    -OUTPUT is [batches,model_d]
    -Hint: Transpose operations using permutations are needed as we only want to transpors model_d and time
    '''
    def __init__(self):
        super(LSTM_Attention, self).__init__()

    def call(self, inputs):
        a_i = tf.exp(tf.matmul(inputs,tf.transpose(inputs[:,0:1,:], perm=[0,2,1])))
        a_i = tf.divide(a_i,tf.reduce_sum(a_i))
        outputs = tf.matmul(tf.transpose(inputs,perm=[0,2,1]), a_i)
        return outputs

class LSTM_Multipath(layers.Layer):
    '''
    -Multi LSTM models running in parallel througt the timeseries of the stocks
    -First timeseries is always the overall market
    -Before the time states are fed to the lstm models they will be transformed using a dense layer
    -INPUT dim is [batches,stocks,time,state]
    -OUTPUT after the transformation is [batches,stocks,time,model_d]
    -INPUT lstm is [batches,time,features]
    -OUTPUT is [batches,stocks,time,features]
    '''
    def __init__(self, dense_output, num_paths):
        super(LSTM_Multipath, self).__init__()
        self.dense_output = dense_output
        self.num_paths = num_paths

    def build(self, inp_sh):
        self.transformer = layers.Dense(self.dense_output,activation='tanh')
        self.lstm_models = [layers.LSTM(self.dense_output, return_sequences=True, return_state=True) for _ in range(self.num_paths)]
        self.attention = LSTM_Attention()

    def call(self, inputs):
        x = self.transformer(inputs)
        x_list = []
        for i in range(x.shape[1]):
            tmp, _, _= self.lstm_models[i](x[:,i,:,:])
            tmp = self.attention(tmp)
            x_list.append(tmp)
        outputs = tf.stack(x_list, axis=1)
        return outputs

class LSTM_Normalization(layers.Layer):
    '''
    Normalizing each feature dimension across all stocks using a learnable z normalization
    INPUT is [batches,stocks,features]
    OUTPUT is [batches,stocks,features]
    '''
    def __init__(self, lstm_dim):
        super(LSTM_Normalization, self).__init__()
        self.lstm_dim = lstm_dim

    def build(self, inp_sh):
        self.factor_gamma = self.add_weight('gammas',[1,inp_sh[1],inp_sh[2],1])
        self.factor_beta = self.add_weight('betas',[1,inp_sh[1],inp_sh[2],1])

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1])
        mean = tf.expand_dims(mean, axis=1)
        std = tf.expand_dims(tf.sqrt(var), axis=1)
        outputs = tf.math.add(
                    tf.math.multiply(self.factor_gamma,
                    tf.math.divide(tf.math.subtract(inputs,mean),std)),
                    self.factor_beta)
        return outputs

class Multi_Context(layers.Layer):
    '''
    Adding the market representation to each stock representation weighted by a factor
    INPUT is [batches, stocks, features]
    OUTPUT is [batches, stocks -1, features]
    '''
    def __init__(self,market_weight):
        super(Multi_Context, self).__init__()
        self.market_weight = market_weight

    def build(self, inp_sh):
        self.factor_beta = self.market_weight

    def call(self, inputs):
        outputs = tf.add(inputs[:,1:,:,:],tf.multiply(self.factor_beta, inputs[:,0:1,:,:]))
        return outputs

class Context_Transformer(layers.Layer):
    '''
    Transform the multi head attention as described in 'attention is all you need'
    INPUT is []
    OUTPUT is []
    '''
    def __init__(self, rate, num_dense):
        super(Context_Transformer, self).__init__()
        self.rate = rate
        self.num_dense = num_dense

    def build(self, inp_sh):
        self.d1 = layers.Dense(4*self.num_dense, activation='relu')
        self.d2 = layers.Dense(self.num_dense)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerdrop = tf.keras.layers.Dropout(self.rate)

    def call(self, input1, input2):
        inp_sum = tf.squeeze(tf.math.add(input1,input2), [3])
        mlp = self.d1(inp_sum)
        mlp = self.d2(mlp)
        out = tf.math.tanh(tf.math.add(inp_sum, mlp))
        out = self.layerdrop(out)
        outputs = self.layernorm(tf.squeeze(input2, [3]) + out)
        return outputs