from pyspark import SparkContext
from pyspark.sql import SQLContext, HiveContext
from geopy.distance import vincenty
from pyspark.sql.functions import udf, expr
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, BooleanType, TimestampType
import csv
from datetime import datetime


sc = SparkContext()
sqlContext = HiveContext(sc)

CITI_PATH = 'hdfs:///tmp/citibike.csv'
YELLOW_PATH = 'hdfs:///tmp/yellow.csv.gz'

# initialize two RDDs
citi = sc.textFile(CITI_PATH)
yellow = sc.textFile(YELLOW_PATH)

# strip header rows
yellow = yellow.filter(lambda x: not x.startswith('tpep'))
citi = citi.filter(lambda x: not x.startswith('cartodb'))


# parse citibike data from csv
def parse_citi(chunk):
	for chunk_row in csv.reader(chunk):
		try:
			# station_id
			yield (int(chunk_row[5]),
				# a unique identifier for the ride (bike ID + starting timestamp)
				chunk_row[13] + chunk_row[3],
				# ride start timestamp
				datetime.strptime(chunk_row[3][:-3], '%Y-%m-%d %H:%M:%S'))
		except ValueError:
			pass

# parse taxi data from csv
def parse_yellow(chunk):
	for chunk_row in csv.reader(chunk):
		try:
			# dropoff timestamp
			yield (datetime.strptime(chunk_row[1][:-2], '%Y-%m-%d %H:%M:%S'),
				# dropoff lat
				float(chunk_row[4]),
				# dropoff long
				float(chunk_row[5]))
		except ValueError:
			pass

# parse the datasets into row tuples
yellow_rows = yellow.mapPartitions(parse_yellow)
citi_rows = citi.mapPartitions(parse_citi)


# define dataframe schemas
yellow_schema = StructType([StructField('dropoff_time', TimestampType(), True),
							StructField('dropoff_lat', FloatType(), True),
							StructField('dropoff_lng', FloatType(), True)])

citi_schema = StructType([StructField('station_id', IntegerType(), True),
							StructField('ride_id', StringType(), True),
							StructField('start_time', TimestampType(), True)])

# instantiate the dataframes
yellow_df = sqlContext.createDataFrame(yellow_rows, yellow_schema)
citi_df = sqlContext.createDataFrame(citi_rows, citi_schema)


# filtering function to check if the taxi dropoff location is within 0.25 miles of citibike station
def is_dropoff_close(lat, lng):
	# greenwich and 8th ave station
	station = (40.73901691, -74.00263761)
	# taxi dropoff location
	dropoff = (lat, lng)
    
	try:
		distance = vincenty(station, dropoff)
		if distance.miles <= 0.25:
			return True
		else:
			return False
	except ValueError:
		return False
# wrap the filtering function in pyspark-friendly UDF
is_dropoff_close_udf = udf(is_dropoff_close, BooleanType())


# filtering function to ensure that a timestamp occurs on february 1st
def is_feb_first(stamp):
	try:
		if ((stamp.month == 2) & (stamp.day == 1)):
			return True
		else:
			return False
	except:
		return False
# wrap the filtering function in pyspark-friendly UDF
is_feb_first_udf = udf(is_feb_first, BooleanType())


# only citibike rides on Feb 1
citi_df = citi_df.filter(is_feb_first_udf(citi_df.start_time))
# only citibike rides starting at greenwich and 8th ave
citi_df = citi_df.filter(citi_df.station_id == 284)
# take only the taxi rides ending within 0.25 miles of station
yellow_df = yellow_df.filter(is_dropoff_close_udf(yellow_df.dropoff_lat, yellow_df.dropoff_lng))


# build a time window in which citibike trips should be considered
yellow_df = yellow_df.withColumn(
	'window_end',
	yellow_df.dropoff_time + expr("INTERVAL 10 MINUTE"))


# join the datasets on two conditions:
# bike start time is after the taxi dropoff
# bike start time is before the 10 minute window's endpoint
cond = [citi_df.start_time >= yellow_df.dropoff_time,
		citi_df.start_time <= yellow_df.window_end]

joined_df = yellow_df.join(citi_df, cond, 'inner')

print '{} citibike rides'.format(joined_df.select('ride_id').distinct().count())


"""
DATA INDICES

citibike:
0 cartodb_id,
1 the_geom,
2 tripduration,
3 starttime,
4 stoptime,
5 start_station_id,
6 start_station_name,
7 start_station_latitude,
8 start_station_longitude,
9 end_station_id,
10 end_station_name,
11 end_station_latitude,
12 end_station_longitude,
13 bikeid,
14 usertype,
15 birth_year,
16 gender


taxi:
0 tpep_pickup_datetime
1 tpep_dropoff_datetime
2 pickup_latitude
3 pickup_longitude
4 dropoff_latitude
5 dropoff_longitude

"""
