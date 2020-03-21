#%% import some libraries...
from math import cos, asin, sqrt
import pandas as pd
import csv
import time
pd.options.display.float_format = '{:.4f}'.format

#%%Define functions
def csv_contents_to_pandas_df(directory_name, file_name):
    '''Function to read and assign csv file contents to pandas df'''
    try:
        with open(directory_name + '/' + file_name + '.csv', 'rb') as file_obj:
            temp_df = pd.read_csv(file_obj)
            file_obj.close()
    except FileNotFoundError:
        print("File not found")
    return temp_df

def distance_between_coordinates(lat1, lon1, lat2, lon2):
    '''Function to Calculate the distance between 2 sets of coordinates'''
    p = 0.017453292519943295  # Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * \
        cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))  # 2*R*asin...

def add_row_to_population_lookup(
    population_lookup_df, cca2, Country, pop2020, dropdownData, area, GrowthRate
    ):
    '''Function to add a row to an existing population lookup table.  First, 
        create the data frame, then do any calculations, say world percentage,
        rank, population density, then create the output df which is the 
        original along with the added row'''
    temp_df_row_to_add = pd.DataFrame({
        'country_populations_cca2': [cca2],
        'country_populations_Country': [Country],
        'country_populations_pop2020':  [pop2020],
        'country_populations_dropdownData': [dropdownData],
        'country_populations_area': [area],
        'country_populations_Density': [0.00],
        'country_populations_GrowthRate': [GrowthRate],
        'country_populations_WorldPercentage': [0.00],
        'country_populations_rank': [0]})
    #calculate density
    temp_df_row_to_add.at[0, 'country_populations_Density'] = temp_df_row_to_add.loc[0]['country_populations_pop2020'] / \
        temp_df_row_to_add.loc[0]['country_populations_area']
    #concatenate dfs
    temp_concatenated_df = pd.concat([
        population_lookup_df,
        temp_df_row_to_add
    ]).reset_index(drop=True)
    #Calculate world percentage
    temp_concatenated_df['country_populations_worldPercentage_updated'] = (temp_concatenated_df['country_populations_pop2020'] / \
        temp_concatenated_df['country_populations_pop2020'].sum()).astype(float)
    #Calculate the rank
    temp_concatenated_df['country_populations_rank_updated'] = temp_concatenated_df.country_populations_pop2020.rank(
        ascending=False, method='dense')

    return temp_concatenated_df


#%% Set params...
#determines how many items to look up...
run_nearest_city_loop = False
num_to_lookup_long_and_lats = None
num_to_lookup_case_coordinates = None

input_directory_name_corona_case = 'csse_covid_19_data/csse_covid_19_time_series'
output_file_name_corona_case = 'corona_daily_long_lat.csv'


#%% This is where we read the daily case files and melt them
input_file_name_corona_case_confirmed = "time_series_19-covid-Confirmed"

corona_daily_by_country_confirmed = csv_contents_to_pandas_df(
    directory_name=input_directory_name_corona_case, 
    file_name=input_file_name_corona_case_confirmed)

#add an id column
corona_daily_by_country_confirmed['case_lat_long_id'] = corona_daily_by_country_confirmed.groupby(
    ['Lat', 'Long']).ngroup()

#melt
corona_daily_by_country_confirmed_melt = pd.melt(
    corona_daily_by_country_confirmed,
    id_vars=('Province/State', 'Country/Region', 'Lat', 'Long', 'case_lat_long_id')).add_prefix('cases_')

#Rename columns and assign zero values to additional columns
corona_daily_by_country_confirmed_melt = corona_daily_by_country_confirmed_melt.rename(columns={
    'cases_variable': 'cases_date',
    'cases_value': 'confirmed'})
corona_daily_by_country_confirmed_melt['deaths'] = 0
corona_daily_by_country_confirmed_melt['recovered'] = 0

corona_daily_by_country_confirmed_melt.dtypes

#%% This is where we read the daily death case files and melt them
input_file_name_corona_case_deaths = "time_series_19-covid-Deaths"

corona_daily_by_country_deaths = csv_contents_to_pandas_df(
    directory_name=input_directory_name_corona_case,
    file_name=input_file_name_corona_case_deaths)

#add an id column
corona_daily_by_country_deaths = corona_daily_by_country_deaths.merge(
    corona_daily_by_country_confirmed[['Lat', 'Long', 'case_lat_long_id']],
     left_on=['Lat', 'Long'], right_on=['Lat', 'Long'],
     how='left')

#melt
corona_daily_long_lat_deaths_melt = pd.melt(
    corona_daily_by_country_deaths,
    id_vars=('Country/Region', 'Lat', 'Long', 'case_lat_long_id')).add_prefix('cases_')

#weird data type issue (results as "object" rather than "int")
corona_daily_long_lat_deaths_melt['cases_value'] = pd.to_numeric(
    corona_daily_long_lat_deaths_melt['cases_value'], errors='coerce')

#Rename columns and assign zero values to additional columns
corona_daily_long_lat_deaths_melt = corona_daily_long_lat_deaths_melt.rename(columns={
    'cases_variable': 'cases_date',
    'cases_value': 'deaths'})
corona_daily_long_lat_deaths_melt['confirmed'] = 0
corona_daily_long_lat_deaths_melt['recovered'] = 0


#%% This is where we read the daily recovered case files and melt them
input_file_name_corona_case_recovered = "time_series_19-covid-Recovered"

corona_daily_by_country_recovered = csv_contents_to_pandas_df(
    directory_name=input_directory_name_corona_case,
    file_name=input_file_name_corona_case_recovered)

#add an id column
corona_daily_by_country_recovered = corona_daily_by_country_recovered.merge(
    corona_daily_by_country_confirmed[['Lat', 'Long', 'case_lat_long_id']],
    left_on=['Lat', 'Long'], right_on=['Lat', 'Long'],
    how='left')

#remove extraneous columns
corona_daily_long_lat_country_recovered = corona_daily_by_country_recovered.drop(
    ["Province/State"], axis=1)

#melt
corona_daily_long_lat_recovered_melt = pd.melt(
    corona_daily_long_lat_country_recovered,
    id_vars=('Country/Region', 'Lat', 'Long', 'case_lat_long_id')).add_prefix('cases_')

#Rename columns and assign zero values to additional columns
corona_daily_long_lat_recovered_melt = corona_daily_long_lat_recovered_melt.rename(columns={
    'cases_variable': 'cases_date',
    'cases_value': 'recovered'})
corona_daily_long_lat_recovered_melt['confirmed'] = 0
corona_daily_long_lat_recovered_melt['deaths'] = 0


#%% concatenate the DataFrames
corona_daily_long_lat_melt = pd.concat([
    corona_daily_long_lat_recovered_melt, 
    corona_daily_long_lat_deaths_melt, 
    corona_daily_by_country_confirmed_melt])

#%% Write the file to csv
corona_daily_long_lat_melt.to_csv(
    'corona_cases_daily_with_populations_postunion.csv', index=False)

#filter any records which have zero confirmed, deaths, recovered...
corona_daily_long_lat_melt['total_observations'] = corona_daily_long_lat_melt[['recovered', 'deaths', 'confirmed']].sum(axis=1)
corona_daily_long_lat_melt = corona_daily_long_lat_melt[
    corona_daily_long_lat_melt['total_observations'] > 0]

#apply data frame
# corona_daily_with_meta_data = corona_daily_long_lat_melt

#%% This is where we read our population file and merge it with the corona cases
# Here we also 1) add rows to our population table and 2) adjust the names of 
#  countries in our population table
input_directory_population_lookup = 'population_data'
input_file_name_population_lookup = "population-data"

#Read in file
population_lookup = csv_contents_to_pandas_df(
    directory_name=input_directory_population_lookup,
    file_name=input_file_name_population_lookup).add_prefix('country_populations_')

#Add a row
population_lookup_adjusted = add_row_to_population_lookup(
    population_lookup_df = population_lookup,
    cca2='n/a',
    Country='Cruise Ship',
    pop2020=3.770,
    dropdownData='n/a',
    area=0.19314,
    GrowthRate = 0,         
)

#Fix some naming disputes
temp_country_replacement_list = {
    'United States': 'US',
    "Macedonia": "North Macedonia",
    "South Korea": 'Korea, South',
    "Czech Republic": 'Czechia',
    "DR Congo": 'Congo (Kirshasa)',
    "Ivory Coast": "Cote d'lvoire"
}
population_lookup_adjusted = population_lookup_adjusted.replace(
    {'country_populations_Country': temp_country_replacement_list})

#%% Merge the two files and write to csv
output_file_name_corona_case_with_meta_and_populations = 'corona_cases_daily_with_populations.csv'
corona_daily_long_lat_melt = corona_daily_long_lat_melt.merge(
    population_lookup_adjusted, left_on='cases_Country/Region', right_on='country_populations_Country')


#%% This is because python isn't so good at identifying date types (for pragith)
corona_daily_long_lat_melt['cases_date'] = pd.to_datetime(
    corona_daily_long_lat_melt['cases_date'])

#find the max date and add it to the dataframe
max_date = max(corona_daily_long_lat_melt['cases_date'])
corona_daily_long_lat_melt['cases_max_date'] = max_date


#%% Write the file to csv
corona_daily_long_lat_melt.to_csv(
    'corona_cases_daily_with_populations_preaggregate.csv', index=False)


#%% aggregate for row reduction
corona_daily_with_meta_data_and_populations_aggregated = corona_daily_long_lat_melt.groupby([
    'cases_Country/Region',
    'cases_Lat',
    'cases_Long',
    'cases_case_lat_long_id',
    'cases_date',
    'country_populations_pop2020',
    'country_populations_dropdownData',
    'country_populations_area',
    'country_populations_Density',
    'country_populations_GrowthRate',
    'country_populations_WorldPercentage',
    'country_populations_rank',
    'country_populations_worldPercentage_updated',
    'country_populations_rank_updated',
    'cases_max_date'
]).agg(recovered=('recovered', 'sum'),
       confirmed=('confirmed', 'sum'),
       deaths=('deaths', 'sum'),
       total_observations=('total_observations', 'sum')
       ).reset_index()

#%% Write the file to csv
corona_daily_long_lat_melt.to_csv(
    output_file_name_corona_case_with_meta_and_populations, index = False)


#%% Add a marker for when total confirmed cases reached N for each country
num_cases_to_start = 10
corona_daily_with_meta_data_and_populations_aggregated.loc[
    corona_daily_with_meta_data_and_populations_aggregated['confirmed'] >= num_cases_to_start] == 'greater_than_10_cases'


#%% Assign a sequential number to each date based on the "confirmed_cases_less_than_or_greater_than_10"
corona_daily_with_meta_data_and_populations_aggregated = corona_daily_with_meta_data_and_populations_aggregated
corona_daily_with_meta_data_and_populations_aggregated['cases_Country/Region']

#add an id column
corona_daily_with_meta_data_and_populations_aggregated['date_count_by_country'] = corona_daily_with_meta_data_and_populations_aggregated.groupby(
    ['cases_Country/Region', 'cases_date']).ngroup()



#Add a column to indicate how many days after N cases have passed
#%% Add a marker for when total confirmed cases reached 10 for each country
