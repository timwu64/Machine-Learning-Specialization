import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
#...
#input > weight > hidden layer 1 (activation function)
#feedforward
#compare output to intended output > cost function (cross entropy)
#optimization fucntion (optimizer) > minimize cost (adamOtimizer.. SGD, adaGrad)
#backrpogragation
#feed forward + backprop = epoch
#...

mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

# 10 classes, 0-9
#...
#0=[1,0,0,0,0,0,0,0,0,0,0]
#1=[0,1,0,0,0,0,0,0,0,0,0]
#2=[0,0,1,0,0,0,0,0,0,0,0]
#3=[0,0,0,1,0,0,0,0,0,0,0]
#...

n_modes_hl1 = 500
n_modes_hl2 = 500
n_modes_hl3 = 500

n_classes = 10
batch_size = 100

# height x width
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer ={'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
			 'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}
	hidden_2_layer ={'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
			 'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}
	hidden_3_layer ={'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
			 'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}
	output_layer ={'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
		       'biases':tf.Variable(tf.random_normal([n_classes]))}

	# (input_data * weights) + biases
	l1 = tf.add(tf.matmul(data, hedden_1_layer['weights']) + hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hedden_2_layer['weights']) + hidden_2_layer['biases'])	
	l2 = tf.nn.relu(l2)
	
	l3 = tf.add(tf.matmul(l2, hedden_3_layer['weights']) + hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)
	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']	
	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction.y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	# cycles 
	hm_epochs = 2
	
	with tf.Session() as sess:
	     sess.run(tf.initialize_all_variables())
	
	     for epoch in hm_epochs:
		epoch_loss = 0
		for _ in range(int(mnist.train.num_examples/batch_size)):
			x, y = mnist.train.next_batch(batch_size)
			_, c = see.run([optimizer, cost], feed_dict ={x: x, y: y})
			epoch_loss += c
		print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:' epoch_loss)
	     correct = tf.equal(tf.argmax(predicition, 1), 



