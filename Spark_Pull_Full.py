import pyspark as ps
from collections import Counter
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import random
from data_prep import parse_line
spark = (ps.sql.SparkSession.builder 
        .master("local[4]") 
        .appName("nathanscope") 
        .getOrCreate()
        )


sc = spark.sparkContext

col_names = ['marketplace','customer_id','review_id','product_id','product_parent','product_title','product_category','star_rating','helpful_votes','total_votes','vine','verified_purchase','review_headline','review_body','review_date']
cols = {}
for i in range(len(col_names)):
    cols[col_names[i]] = i 
    
def casting_function(row):
    if row[0] == 'marketplace':
        return ()
    if len(row)!=15:
        return(len(row))
    (marketplace, customer_id, review_id, product_id, product_parent, product_title,product_category,star_rating,helpful_votes,total_votes,vine,verified_purchase,review_headline,review_body,review_date) = row
    
    
    return (marketplace, str(customer_id), str(review_id), str(product_id), product_parent, str(product_title),str(product_category),int(star_rating),int(helpful_votes),int(total_votes),vine,verified_purchase,str(review_headline),str(review_body),review_date)

def time_cast(row):
    pattern = '%Y-%m-%d'
    row =list(row)
    row[14] = int(time.mktime(time.strptime(row[14], pattern)))/86400
    return tuple(row)

def toCSVLine(data):
  return ','.join(str(d) for d in data)

def make_parse(row):
    row = list(row)
    row.append(parse_line(row[13]))
    return tuple(row)

file = open("data/S3_Repos.txt","r+")
repos = []
for line in file.readlines():
    line = line.replace('https://s3.amazonaws.com/amazon-reviews-pds/tsv/','')
    line = line.replace('\n','')
    repos.append('s3://amazon-reviews-pds/tsv/'+line)
     

total_row_counts = []
final_rows = []
parsed_final = []
opt_rows = []
Scale = []
Names = []
tenK_sizes = []
loop_num = 0




for repo in repos:
    loop_num = loop_num + 1
    print(str(loop_num)+'-Begin: '+repo.replace('s3://amazon-reviews-pds/tsv/amazon_reviews_','').replace('.tsv.gz',''))
    rdd_books = sc.textFile(repo).map(lambda rowstr : rowstr.split("\t")).map(casting_function)
    rdd_books =rdd_books.filter(lambda x: len(x)==15).map(time_cast)
    rdd_sample = rdd_books.filter(lambda x: x[7]==1).filter(lambda x: x[9]>5)
    print('Check_1')
    tiny_df = pd.DataFrame(rdd_sample.take(10000))
    print('Check_2')
    tiny_df.to_csv('data/size_check.csv')
    tenK_size = os.path.getsize('data/size_check.csv')
    print('Check_3')
    total_rows = rdd_sample.count()
    total_row_counts.append(total_rows)
    print('Check_4')
    optimal_bytes = 90000000
    
    tenK_scale = optimal_bytes/tenK_size
    tenK_sizes.append(tenK_size)
    
    optimal_rows = 10000*tenK_scale
    opt_rows.append(optimal_rows)
    file_name = 'data/Spark_Pulls/'+repo.replace('s3://amazon-reviews-pds/tsv/amazon_reviews_','').replace('.tsv.gz','')+'.csv'
    print('Check_5')
    if total_rows <optimal_rows:
        print('Check_6a')
        temp = pd.DataFrame(rdd_sample.map(make_parse).collect())
        print('Check_7a')
        temp.to_csv(file_name)
        Scale.append(1)
        final_rows.append(total_rows)
    else:
        scale = optimal_rows/total_rows
        Scale.append(scale)
        final_rows.append(total_rows*scale)
        rdd_sample = rdd_sample.filter(lambda x: random.random() < scale).map(make_parse)
        print('Check_6b')
        temp = pd.DataFrame(rdd_sample.collect())
        print('Check_6b')
        temp.to_csv(file_name)
        print('Check_7b')
    optimal_bytes = 95000000
    parsed_size = os.path.getsize(file_name)
    if parsed_size > optimal_bytes:
        temp = temp.sample(frac=optimal_bytes/parsed_size, random_state=1)
        temp.to_csv(file_name)
    parsed_final.append(len(temp))
    print('finished: '+repo.replace('s3://amazon-reviews-pds/tsv/amazon_reviews_','').replace('.tsv.gz',''))
    Names.append(repo.replace('s3://amazon-reviews-pds/tsv/amazon_reviews_','').replace('.tsv.gz',''))
df = pd.DataFrame()
df['Name'] = Names
df['Scale'] = Scale
df['total_rows']  =total_row_counts
df['final_rows'] = final_rows
df['parsed_final'] = parsed_final
df['opt_rows'] = opt_rows
df['tenK_size'] = tenK_sizes
df.to_csv('data/spark_metrics.csv')
    