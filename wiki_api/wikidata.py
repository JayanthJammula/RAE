import json, urllib # Needed libs
import logging
import sys
import time
import urllib.request  as ur
from urllib.error import HTTPError, URLError
from pprint import pprint # Not necessary
from urllib.parse import quote
from wiki_api.Wiki import OnelineSearchEngine
wikisearch = OnelineSearchEngine()
from functools import lru_cache

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {"User-Agent": "RAE/1.0 (+https://github.com/sycny/RAE)"}

def _safe_json_get(url, max_retries=3, initial_delay=0.5):
	"""Fetch JSON from URL with retry and exponential backoff."""
	delay = initial_delay
	last_error = None
	for attempt in range(max_retries):
		try:
			req = ur.Request(url, headers=_DEFAULT_HEADERS)
			with ur.urlopen(req, timeout=10) as resp:
				content = resp.read()
			return json.loads(content)
		except (HTTPError, URLError, TimeoutError) as e:
			last_error = e
			logger.debug(f"Wikidata API attempt {attempt+1}/{max_retries} failed for {url}: {e}")
			if attempt < max_retries - 1:
				time.sleep(delay)
				delay *= 2
		except json.JSONDecodeError as e:
			logger.debug(f"Invalid JSON from {url}: {e}")
			return None
	logger.warning(f"Wikidata API failed after {max_retries} attempts: {url} ({last_error})")
	return None

@lru_cache(None)
def entity2id(q):
	norm_q = wikisearch.normalize(q)
	if norm_q != None:
		q = norm_q
	ans = []
	url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search="+quote(q)+"&language=en" # this quote is so useful!!: quote(Karl Döhler) ---> Karl%20D%C3%B6hler
	response = _safe_json_get(url)
	if response:
		ans += response.get("search", [])
	try:
		if (ans == [] and " " in q):
			url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search="+"+".join(q.split(" ")[::-1])+"&language=en"
			response = _safe_json_get(url)
			if response:
				ans += response.get("search", [])
		if (ans == [] and len(q.split(" ")) > 2):
			url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search="+"+".join([q.split(" ")[0], q.split(" ")[-1]])+"&language=en"
			response = _safe_json_get(url)
			if response:
				ans += response.get("search", [])
		if len(ans) > 0:
			return ans[0]["id"]
		else:
			return None
	except (KeyError, IndexError, TypeError) as e:
		logger.debug(f"entity2id fallback failed for '{q}': {e}")
		return None



@lru_cache(None)
def id2entity(q):
	# Get wikidata id from wikidata api
	ans = []
	url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search="+"+".join(q.split(" "))+"&language=en"
	response = _safe_json_get(url)
	if response:
		ans += response.get("search", [])
	if (ans == [] and " " in q):
		# Reverse Trick : Pan Changjiang
		url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search="+"+".join(q.split(" ")[::-1])+"&language=en"
		response = _safe_json_get(url)
		if response:
			ans += response.get("search", [])
	if (ans == [] and len(q.split(" ")) > 2):
		# Abbreviation Trick
		url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search="+"+".join([q.split(" ")[0], q.split(" ")[-1]])+"&language=en"
		response = _safe_json_get(url)
		if response:
			ans += response.get("search", [])
	try:
		if len(ans) > 0:
			# Returns the first one, most likely one
			return ans[0]["label"]
		else:
			# Some outliers : Salvador Domingo Felipe Jacinto Dali i Domenech - Q5577
			return "Not Applicable"
	except (KeyError, IndexError, TypeError) as e:
		logger.debug(f"id2entity fallback failed for '{q}': {e}")
		return "Not Applicable"

def getp(p):
	# Get property name given property id
	# Initialization required
	return property_dict[p]

def getc(c):
	# Get entity name given entity id
	url = "https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids="+c+"&languages=en&format=json"
	response = _safe_json_get(url)
	if response:
		try:
			return response["entities"][c]["labels"]["en"]["value"]
		except KeyError:
			return "Not Applicable"
	return "Not Applicable"

def Related(name): 
	# Get related property-entity (property id, property name, entity id, entity name) given entity name
	# Return a list of dicts, each dict contains (pid, property, eid, entity)
	# Fail to fetch eid would result in empty list
	query = entity2id(name)
	if query is None: return []
	ans = []
	url = "https://www.wikidata.org/w/api.php?action=wbgetentities&ids="+query+"&format=json&languages=en"
	response = _safe_json_get(url)
	if not response:
		return ans
	for p in response["entities"].get(query, {}).get("claims", {}):
		for c in response["entities"][query]["claims"][p]:
			# Enumerate property & entity (multi-property, multi-entity)
			try:
				# Some properties are not related to entities, thus try & except
				cid = c["mainsnak"]["datavalue"]["value"]["id"]
				ans.append({
					"pid": p,
					"property": getp(p),
					"eid": cid,
					"entity": getc(cid)
					})
				#ans.append("\\property\\"+p+"\t"+getp(p)+"\t\\entity\\"+cid+"\t"+getc(cid))
				# Print in a pid-pname-eid-ename fashion
			except (KeyError, TypeError) as e:
				logger.debug(f"Skipping claim for property {p}: {e}")
				continue
	return ans

def init():
	# WARNING: RUN BEFORE USE GETP
	# Needed for property name fetching
	global property_dict
	property_dict = {}
	url = "https://quarry.wmflabs.org/run/45013/output/1/json"
	# Fetch json from given lib
	res = _safe_json_get(url)
	if not res:
		return
	for w in res["rows"]:
		property_dict[w[0]] = w[1]


#init()
# For test only
'''
entity = input("Please input the entity name\n")
pprint(Related(entity))
pprint('-'*37)
pprint(id2entity(entity))
'''
