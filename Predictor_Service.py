import falcon
from falcon_cors import CORS
import json
import pygeoip
import json
import datetime as dt
import ipaddress
import math
from concurrent.futures import *
import numpy as np
from google.cloud import datastore

def logit(x):
	return (np.exp(x) / (1 + np.exp(x)))

def is_visible(client_size, ad_position):
        y=height=0
        try:
                height  = int(client_size.split(',')[1])
                y = int(ad_position.split(',')[1])
        except:
                pass
        if y < height:
                return "1"
        else:
                return "0"

class Predictor(object):
	def __init__(self,domain,is_big):
		self.client = datastore.Client('sulvo-europe')
	        self.ctr = 'ctr_' + domain 
		self.ip = "ip_" + domain
		self.scores = "score_" + domain
		self.probabilities = "probability_" + domain
		self.multiplier = 'multiplier_all'
		if is_big:
			self.is_big = "is_big_" + domain
			self.scores_big = "score_big_" + domain
			self.probabilities_big = "probability_big_" + domain
        	self.gi = pygeoip.GeoIP('GeoIP.dat')
		self.big = is_big
		self.domain = domain

	def get_hour(self,timestamp):
        	return dt.datetime.utcfromtimestamp(timestamp / 1e3).hour

	def fetch_score(self, featurename, featurevalue, kind):
		pred = 0
		try:
			key = self.client.key(kind,featurename + "_" + featurevalue)
			res= self.client.get(key)
			if res is not None:
	        		pred = float(res['score'])
		except:
			pass
		return pred

	def get_score(self, featurename, featurevalue):
		with ThreadPoolExecutor(max_workers=8) as pool:
                        future_score = pool.submit(self.fetch_score,featurename, featurevalue,self.scores)
			future_prob = pool.submit(self.fetch_score,featurename, featurevalue,self.probabilities)
			if self.big:
				future_howbig = pool.submit(self.fetch_score,featurename, featurevalue,self.is_big)
				future_predbig = pool.submit(self.fetch_score,featurename, featurevalue,self.scores_big)
				future_probbig = pool.submit(self.fetch_score,featurename, featurevalue,self.probabilities_big)
			pred = future_score.result()
			prob = future_prob.result()
			if not self.big:
				return pred, prob
			howbig = future_howbig.result()
			pred_big = future_predbig.result()
			prob_big = future_probbig.result()
			return howbig, pred, prob, pred_big, prob_big


	def get_value(self, f, value):
		if f == 'visible':
			fields = value.split("_")
			value = is_visible(fields[0], fields[1])		
		if f == 'ip':
			ip = str(ipaddress.IPv4Address(ipaddress.ip_address(value)))
                        geo = self.gi.country_name_by_addr(ip)
			if self.big:
				howbig1,pred1, prob1, pred_big1, prob_big1 = self.get_score('geo', geo)
			else:
				pred1, prob1 = self.get_score('geo', geo)
                	freq = '1'
			key = self.client.key(self.ip,ip)
			res = self.client.get(key)
			if res is not None:
			        freq = res['ip']
			if self.big:
				howbig2, pred2, prob2, pred_big2, prob_big2 = self.get_score('frequency', freq)
			else:
				pred2, prob2 =  self.get_score('frequency', freq)
			if self.big:
				return (howbig1 + howbig2), (pred1 + pred2), (prob1 + prob2), (pred_big1 + pred_big2), (prob_big1 + prob_big2)
			else:
				return (pred1 + pred2), (prob1 + prob2)	
		if f == 'root':
			try:
				res = client.get('root', value)
				if res is not None:
					ctr = res['ctr']
					avt = res['avt']
					avv = res['avv']
					if self.big:
						(howbig1,pred1,prob1,pred_big1,prob_big1) = self.get_score('ctr', str(ctr))
	                                	(howbig2,pred2,prob2,pred_big2,prob_big2) = self.get_score('avt', str(avt))
		                                (howbig3,pred3,prob3,pred_big3,prob_big3) = self.get_score('avv', str(avv))
        		                        (howbig4,pred4,prob4,pred_big4,prob_big4) = self.get_score(f, value)
					else:
						(pred1,prob1) = self.get_score('ctr', str(ctr))
						(pred2,prob2) = self.get_score('avt', str(avt))
						(pred3,prob3) = self.get_score('avv', str(avv))
						(pred4,prob4) = self.get_score(f, value)
				if self.big:
 					return (howbig1 + howbig2 + howbig3 + howbig4), (pred1 + pred2 + pred3 + pred4), (prob1 + prob2 + prob3 + prob4),(pred_big1 + pred_big2 + pred_big3 + pred_big4),(prob_big1 + prob_big2 + prob_big3 + prob_big4)
				else:
	 				return (pred1 + pred2 + pred3 + pred4), (prob1 + prob2 + prob3 + prob4)
			except:
				return 0,0
	        if f == 'client_time':
        	     	value = str(self.get_hour(int(value)))
		return self.get_score(f, value)


	def on_post(self, req, resp):
		try:
			input_json = json.loads(req.stream.read(),encoding='utf-8')
			input_json['visible'] = input_json['client_size'] + "_" + input_json['ad_position']
			del input_json['client_size']
			del input_json['ad_position']
			howbig = 0
			pred = 0 
			prob = 0
			pred_big = 0
			prob_big = 0
			with ThreadPoolExecutor(max_workers=8) as pool:
				future_array = { pool.submit(self.get_value,f,input_json[f]) : f for f in input_json}
				for future in as_completed(future_array):
					if self.big:
		                	        howbig1, pred1, prob1,pred_big1,prob_big1 = future.result()
						pred = pred + pred1
						pred_big = pred_big + pred_big1
						prob = prob + prob1
						prob_big = prob_big + prob_big1
						howbig = howbig + howbig
					else:
	                	        	pred1, prob1 = future.result()
						pred = pred + pred1
						prob = prob + prob1

			if self.big:
				if howbig > .65:
					pred, prob = pred_big, prob_big

			resp.status = falcon.HTTP_200
	
		        res = math.exp(pred)-1
			if res < 0.1:
				res = 0.1
			if prob < 0.1 :
				prob = 0.1
			if prob > 0.9:
				prob = 0.9
		
			row = None
			if self.big:
				if howbig > 0.6:
					key = self.client.key('multiplier_all', self.domain + "_big")
				else:
					key = self.client.key('multiplier_all', self.domain)
			else:
				key = self.client.key('multiplier_all', self.domain)
			res1 = self.client.get(key)
			#print res['high'],res['low']
			high = float(res1['high'])
			low = float(res1['low'])
			multiplier = low + (high -low)*prob
			res = multiplier*res
			resp.body = str(res)
		except Exception,e:
			print(str(e))
			resp.status = falcon.HTTP_200
			resp.body = str("0.1")

cors = CORS(allow_all_origins=True,allow_all_methods=True,allow_all_headers=True)  
wsgi_app = api = falcon.API(middleware=[cors.middleware])

f = open('publishers2.list')
for line in f:
	if "#" not in line:
		fields = line.strip().split('\t')
		domain = fields[0].strip()
		big = (fields[1].strip() == '1')
		p = Predictor(domain, big)
		url = '/predict/' + domain
		api.add_route(url, p)
f.close()


