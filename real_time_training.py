import pandas as pd
from sklearn import cross_validation
import tensorflow as tf
import numpy as np
from scipy.stats.stats import pearsonr
from sqlalchemy import create_engine
from pymongo import MongoClient
import pygeoip
import datetime as dt
import time
import ipaddress

gi = pygeoip.GeoIP('GeoIP.dat')
disk_engine = create_engine('sqlite:///score.db')
disk_engine1 = create_engine('sqlite:///score1.db')
disk_engine2 = create_engine('sqlite:///score2.db')
disk_engine3 = create_engine('sqlite:///score3.db')
disk_engine4 = create_engine('sqlite:///score4.db')
disk_engine5 = create_engine('sqlite:///score5.db')
disk_engine6 = create_engine('sqlite:///score6.db')
disk_engine7 = create_engine('sqlite:///score7.db')
disk_engine8 = create_engine('sqlite:///score8.db')
disk_engine9 = create_engine('sqlite:///score9.db')
client = MongoClient()
db = client.frequency
collection = db.frequency

def convert_ip(rawip):
	ip = "invalid"
	try:
		ip = gi.country_name_by_addr(str(ipaddress.IPv4Address(ipaddress.ip_address(rawip))))
	except:
		ip = "invalid"
	return ip

def newip(ip, domain):
	data = {"ip" : ip, "frequency" : 1, "domain" : domain}
	collection.insert_one(data)
	return 1

def oldip(ip, domain):
	res = collection.find_one({"ip" : ip, "domain" : domain})
	freq = res['frequency']
	if(freq < 5):
		freq = freq + 1
		collection.update({"ip": ip, "domain" : domain}, {"$set": {"frequency" : freq}})
	return freq
	
def get_hour(timestamp):
	hour = 0
	try:
		hour = dt.datetime.fromtimestamp(timestamp / 1e3).hour
	except:
		hour = 0
	return hour
	
def process_real_time_data(time_limit):
	query = "select * from adlog.day1 where success='1' and server_time > " + str(time_limit) + " order by server_time desc limit 5000;"
	print(query)
	df = pd.read_gbq( query, project_id = 'sulvo-1075' )
	df.replace('^\s+', '', regex=True, inplace=True) #front
	df.replace('\s+$', '', regex=True, inplace=True) #end
	
	time_limit = df['server_time'].max()
	
	df['floor'] = pd.to_numeric(df['floor'], errors='ignore')
	df['client_time'] = pd.to_numeric(df['client_time'], errors='ignore')
	df['client_time'] = df.apply(lambda row: get_hour(row.client_time), axis=1)
		
	y = df['floor']
	X = df[['ip','size','domain','client_time','device','ad_position','client_size','root']]
	X_back = df[['ip','size','domain','client_time','device','ad_position','client_size','root']]
	
	df['geo'] = df.apply(lambda row: convert_ip(row.ip), axis=1)
	df['frequency'] = df.apply(lambda row: oldip(row.ip, row.domain) if pd.notnull(collection.find_one({"ip" : row.ip, "domain" : row.domain})) else newip(row.ip, row.domain), axis=1)
	
	X['frequency'] = df['frequency']
	X_back['frequency'] = X['frequency']
	X['geo'] = df['geo']
	X_back['geo'] = X['geo']

	for col in X.columns:
		avgs = df.groupby(col, as_index=False)['floor'].aggregate(np.mean)
		for index,row in avgs.iterrows():
			k = row[col]
			v = row['floor']
			X.loc[X[col] == k, col] = v
	
	X.drop('ip', inplace=True, axis=1)
	X_back.drop('ip', inplace=True, axis=1)

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size= 0, random_state=42)
	
	X_train = X_train.astype(float) 
	y_train = y_train.astype(float)
	
	X_train = np.log(X_train + 1)
	y_train = np.log(y_train + 1)
	
	X_train = X_train.as_matrix()
	y_train = y_train.as_matrix()
		
	index = []
	i1 = 0
	processed = 0
	while(1):
		flag = True
		for i in range(X_train.shape[1]):
			if i > processed :
				#print(i1,X_train.shape[1],X.columns[i1])
				i1 = i1 + 1
				corr = pearsonr(X_train[:,i], y_train)
				PEr= .674 * (1- corr[0]*corr[0])/ (len(X_train[:,i])**(1/2.0))
				if corr[0] < PEr:
					X_train = np.delete(X_train,i,1)
					index.append(X.columns[i1-1])
					processed = i - 1 
					flag = False
					break
		if flag:
			break
	
	return y_train, X_train, y, X_back, X, time_limit, index
	
def real_time(time_limit):
	tf.reset_default_graph()
	y_train, X_train, y, X_back, X, time_limit, index = process_real_time_data(time_limit)
	
	learning_rate = 0.0001
	
	y_t = tf.placeholder("float", [None,1])
	x_t = tf.placeholder("float", [None,X_train.shape[1]])
	W = tf.Variable(tf.random_normal([X_train.shape[1],1],stddev=.01))
	b = tf.constant(1.0)
	
	model = tf.matmul(x_t, W) + b
	cost_function = tf.reduce_sum(tf.pow((y_t - model),2))
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)
	
	init = tf.initialize_all_variables()
	
	with tf.Session() as sess:
		sess.run(init)
		w = W.eval(session = sess)
		of = b.eval(session = sess)
		print("Before Training #################################################")
		print(w,of)
		print("#################################################################")
		step = 0
		previous = 0
		while(1):
			step = step + 1
			sess.run(optimizer, feed_dict={x_t: X_train.reshape(X_train.shape[0],X_train.shape[1]), y_t: y_train.reshape(y_train.shape[0],1)})
			cost = sess.run(cost_function, feed_dict={x_t: X_train.reshape(X_train.shape[0],X_train.shape[1]), y_t: y_train.reshape(y_train.shape[0],1)})
			if step%1000 == 0:
				print(cost)
				if(previous == cost):
					break
				previous = cost
		w = W.eval(session = sess)
		of = b.eval(session = sess)
		print("Before Training #################################################")
		print(w,of)
		print("#################################################################")
		
		df2 = pd.DataFrame()
		started = False
		i = 0
		#print(index,X_back.columns)
		for col in X_back.columns:
			if str(col) not in index:
				#print(str(col),i)
				df1 = pd.DataFrame()
				df1['feature_value'] = X_back[col].apply(str).as_matrix()  + "_" + str(col)
				df1['score'] = np.multiply(np.log(X[col].astype(float)+ 1),w[i][0])
				#df1['feature_name'] = str(col)
				df1 = df1.drop_duplicates()
				if started:
					df2 = df2.append(df1)
				else:
					df2 = df1
				started = True
				i = i + 1
		print(df2)	
		df2.to_sql('scores', disk_engine, if_exists='replace')
		df2.to_sql('scores', disk_engine1, if_exists='replace')
		df2.to_sql('scores', disk_engine2, if_exists='replace')
		df2.to_sql('scores', disk_engine3, if_exists='replace')
		df2.to_sql('scores', disk_engine4, if_exists='replace')
		df2.to_sql('scores', disk_engine5, if_exists='replace')
		df2.to_sql('scores', disk_engine6, if_exists='replace')
		df2.to_sql('scores', disk_engine7, if_exists='replace')
		df2.to_sql('scores', disk_engine8, if_exists='replace')
		df2.to_sql('scores', disk_engine9, if_exists='replace')
			
time_limit = 0
while(1):
	try:
		real_time(time_limit)
		time.sleep(1200)
	except:
		time.sleep(2400)

	
