import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression


def pate_lamda (x, teacher_models, lamda):

    y_hat = []
        
    for teacher in teacher_models:            
        temp_y = teacher.predict(np.reshape(x, [1,-1]))
        y_hat.append(temp_y)
  
    y_hat = np.asarray(y_hat).flatten()
    n0 = sum(y_hat == 0)
    n1 = sum(y_hat == 1)
  
    lap_noise = np.random.laplace(loc=0.0, scale=lamda)
  
    out = (n1 + lap_noise) / float(n0 + n1)
    out = int(out > 0.5)
        
    return n0, n1, out 


def pategan(x_train, parameters):
  
    n_s = parameters['n_s']
    batch_size = parameters['batch_size']
    k = parameters['k']
    epsilon = parameters['epsilon']
    delta = parameters['delta']
    lamda = parameters['lamda']
  
    L = 20
    alpha = np.zeros([L])
    epsilon_hat = 0
    
    no, dim = x_train.shape
    z_dim = dim
    student_h_dim = dim
    generator_h_dim = 4 * dim  
  
    x_partition = []
    partition_data_no = no // k
    
    idx = np.random.permutation(no)
    
    for i in range(k):
        temp_idx = idx[int(i * partition_data_no):int((i + 1) * partition_data_no)]
        temp_x = x_train[temp_idx, :]      
        x_partition.append(temp_x)    
  
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape=size, stddev=xavier_stddev)    
        
    def sample_Z(m, n):
        return np.random.uniform(-1., 1., size=[m, n])
     
    ## Placeholder
    Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])   
    Z = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim])

    # Student
    S_W1 = tf.Variable(xavier_init([dim, student_h_dim]))
    S_b1 = tf.Variable(tf.zeros(shape=[student_h_dim]))
    
    S_W2 = tf.Variable(xavier_init([student_h_dim, 1]))
    S_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_S = [S_W1, S_W2, S_b1, S_b2]
    
    # Generator

    G_W1 = tf.Variable(xavier_init([z_dim, generator_h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[generator_h_dim]))

    G_W2 = tf.Variable(xavier_init([generator_h_dim, generator_h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[generator_h_dim]))

    G_W3 = tf.Variable(xavier_init([generator_h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))
    
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ## Models
    def generator(z):
        G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
        G_out = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        
        return G_out
    
    def student(x):
        S_h1 = tf.nn.relu(tf.matmul(x, S_W1) + S_b1)
        S_out = tf.matmul(S_h1, S_W2) + S_b2
        
        return S_out
      
    ## Loss  
    G_sample = generator(Z)
    S_fake = student(G_sample)
  
    S_loss = tf.reduce_mean(Y * S_fake) - tf.reduce_mean((1 - Y) * S_fake)
    G_loss = -tf.reduce_mean(S_fake)
  
    # Optimizer
    S_optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4)
    G_optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4)

    S_train_op = S_optimizer.minimize(-S_loss, var_list=theta_S)
    G_train_op = G_optimizer.minimize(G_loss, var_list=theta_G)

    clip_S = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_S]
  
    ## Sessions
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
        
    ## Iterations
    while epsilon_hat < epsilon:      
          
        # 1. Train teacher models
        teacher_models = []
    
        for i in range(k):
                
            Z_mb = sample_Z(partition_data_no, z_dim)
            G_mb = sess.run(G_sample, feed_dict={Z: Z_mb})
                
            temp_x = x_partition[i]
            idx = np.random.permutation(len(temp_x[:, 0]))
            X_mb = temp_x[idx[:partition_data_no], :]
                
            X_comb = np.concatenate((X_mb, G_mb), axis=0)
            Y_comb = np.concatenate((np.ones([partition_data_no,]), np.zeros([partition_data_no,])), axis=0)
                
            model = LogisticRegression()
            model.fit(X_comb, Y_comb)
            teacher_models.append(model)
            
        # 2. Student training
        for _ in range(n_s):
          
            Z_mb = sample_Z(batch_size, z_dim)
            G_mb = sess.run(G_sample, feed_dict={Z: Z_mb})
            Y_mb = []
            
            for j in range(batch_size):                
                n0, n1, r_j = pate_lamda(G_mb[j, :], teacher_models, lamda)
                Y_mb.append(r_j)
       
                # Update moments accountant
                q = np.log(2 + lamda * abs(n0 - n1)) - np.log(4.0) - (lamda * abs(n0 - n1))
                q = np.exp(q)
                
                # Compute alpha
                for l in range(L):
                    temp1 = 2 * (lamda**2) * (l+1) * (l+2)
                    temp2 = (1-q) * ( ((1-q)/(1-q*np.exp(2*lamda)))**(l+1) ) + q * np.exp(2*lamda * (l+1))
                    alpha[l] += np.min([temp1, np.log(temp2)])
        
            # PATE labels for G_mb  
            Y_mb = np.reshape(np.asarray(Y_mb), [-1, 1])
                
            # Update student
            _, D_loss_curr, _ = sess.run([S_train_op, S_loss, clip_S], feed_dict={Z: Z_mb, Y: Y_mb})
    
        # Generator Update        
        Z_mb = sample_Z(batch_size, z_dim)
        _, G_loss_curr = sess.run([G_train_op, G_loss], feed_dict={Z: Z_mb})
        
        # epsilon_hat computation
        curr_list = []        
        for l in range(L):
            temp_alpha = (alpha[l] + np.log(1 / delta)) / float(l + 1)
            curr_list.append(temp_alpha)
        
        epsilon_hat = np.min(curr_list)    

    
    x_train_hat = sess.run(G_sample, feed_dict={Z: sample_Z(no, z_dim)})
    
    return x_train_hat
