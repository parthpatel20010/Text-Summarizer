import nltk
import operator
import networkx as nx
import numpy
import urllib
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import requests
import selenium.webdriver as wdriver
from bs4 import BeautifulSoup as bs
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import tensorflow as tf
textExample = "Hello this is my first sample! Welcome? I am very excited that you're here with me today!"

def summarize():
    url = "https://futurism.com/theories-intelligent-life-fermi-paradox/"
    html = urllib.request.urlopen(url).read()
    soup = bs(html, "lxml")
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text = """World War II (often abbreviated to WWII or WW2), also known as the Second World War, was a global war that lasted from 1939 to 1945, although related conflicts began earlier. The vast majority of the world's countries—including all of the great powers—eventually formed two opposing military alliances: the Allies and the Axis. It was the most global war in history; it directly involved more than 100 million people from over 30 countries. In a state of total war, the major participants threw their entire economic, industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. World War II was the deadliest conflict in human history, marked by 50 to 85 million fatalities, most of which were civilians in the Soviet Union and China. It included massacres, the genocide of the Holocaust, strategic bombing, starvation, disease, and the first use of nuclear weapons in history.[1][2][3][4]

The Empire of Japan aimed to dominate Asia and the Pacific and was already at war with the Republic of China in 1937,[5] but the world war is generally said to have begun on 1 September 1939[6], the day of the invasion of Poland by Nazi Germany and the subsequent declarations of war on Germany by France and the United Kingdom. From late 1939 to early 1941, in a series of campaigns and treaties, Germany conquered or controlled much of continental Europe, and formed the Axis alliance with Italy and Japan. Under the Molotov–Ribbentrop Pact of August 1939, Germany and the Soviet Union partitioned and annexed territories of their European neighbours, Poland, Finland, Romania and the Baltic states. The war continued primarily between the European Axis powers and the coalition of the United Kingdom and the British Commonwealth, with campaigns including the North Africa and East Africa campaigns, the aerial Battle of Britain, the Blitz bombing campaign, and the Balkan Campaign, as well as the long-running Battle of the Atlantic. On 22 June 1941, the European Axis powers launched an invasion of the Soviet Union, opening the largest land theatre of war in history, which trapped the major part of the Axis military forces into a war of attrition. In December 1941, Japan attacked the United States and European colonies in the Pacific Ocean, and quickly conquered much of the Western Pacific. """

    x = sent_tokenize(text)
    nodes = []
    for item in x:
        if(len(item) < 250):
            nodes.append(item)
    edges = connect(nodes)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    weight_dict = sorted(nx.pagerank(graph).items(), key=operator.itemgetter(1))
    sentences = list(weight_dict)
    
    print(sentences[len(sentences)-1][0].encode('ascii', 'ignore'))
    #print(sentences[len(sentences)-2])
    #print(sentences[len(sentences)-3])
    #print(sentences[len(sentences)-4])
def connect(nodes):
    return [(start,end,similarity(start, end)) for start in nodes for end in nodes if start is not end] 


def similarity(s1, s2):
    return len(common_words(s1,s2))/(numpy.log(len(s1)*len(s2)))

def common_words(s1,s2):
     val = []
     words1 = word_tokenize(s1)
     words2 = word_tokenize(s2)
     for x in words1:
         for y in words2:
            if(x==y and x not in set(stopwords.words("english")) or len(wordnet.synsets(x)) !=0  and y in wordnet.synsets(x)[0].lemma_names() and x not in set(stopwords.words("english"))): #and x not in set(stopwords.words("english")) ):
                 val+=x
                 list(filter(lambda a: a != y, words2))
         
     return val

summarize()
