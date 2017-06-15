from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.sql import DataFrame
import numpy as np
from pyspark.sql import SparkSession


spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .getOrCreate()
#import numpy as np


dt = [("1", np.int32), ("2", np.float16), ("3", np.uint8), ("_4", np.uint8), ("_", np.float16)]
is2D_array = True
header_len = 80 if is2D_array else 64+16*len(dt)

schema = StructType([StructField('a1', IntegerType(),False),
				   StructField('a2',FloatType(),False), 
				   StructField('a3',ShortType(),False),
				   StructField('a4',ShortType(),False),
				   StructField('a6',FloatType(),False)])

sc = SparkContext.getOrCreate()

filenameRdd = sc.binaryFiles('gs://dataproc-82bfd75b-4f26-406e-bc4c-afc0bd56f646-us/b.npy')

def read_array(rdd):
	#output = zlib.decompress((bytes(rdd[1])),15+32) # in case also zipped
	array = np.frombuffer(bytes(rdd[1])[header_len:], dtype = dt) # remove Header (80 bytes)
	#array = array.newbyteorder().byteswap() # big Endian
	return array.tolist()

def debug_r_a(rdd):
	return len(bytes(rdd[1])), len(bytes(rdd[1])[header_len:])

unzipped = filenameRdd.flatMap(debug_r_a)
print(unzipped.collect())

unzipped = filenameRdd.flatMap(read_array)
bin_df = spark.createDataFrame(unzipped, schema)
print(bin_df.collect())