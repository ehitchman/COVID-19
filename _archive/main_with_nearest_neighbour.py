#%% import some libraries...
from math import cos, asin, sqrt
import pandas as pd
import csv
import time
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_rows', 500)

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
run_nearest_city_loop = True
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
corona_daily_long_lat_confirmed_melt = pd.melt(
    corona_daily_by_country_confirmed,
    id_vars=('Country/Region', 'Lat', 'Long', 'case_lat_long_id')).add_prefix('cases_')

#Rename columns
corona_daily_long_lat_confirmed_melt = corona_daily_long_lat_confirmed_melt.rename(columns={
    'cases_variable': 'cases_date',
    'cases_value': 'confirmed'})

#Add placeholder columns
corona_daily_long_lat_confirmed_melt['deaths'] = 0
corona_daily_long_lat_confirmed_melt['recovered'] = 0


#%% This is where we read the daily case files and melt them
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

#Rename columns
corona_daily_long_lat_deaths_melt = corona_daily_long_lat_deaths_melt.rename(columns={
    'cases_variable': 'cases_date',
    'cases_value': 'deaths'})

#Add placeholder columns
corona_daily_long_lat_deaths_melt['confirmed'] = 0
corona_daily_long_lat_deaths_melt['recovered'] = 0


#%% This is where we read the daily case files and melt them
input_file_name_corona_case_recovered = "time_series_19-covid-Recovered"

corona_daily_by_country_recovered = csv_contents_to_pandas_df(
    directory_name=input_directory_name_corona_case,
    file_name=input_file_name_corona_case_recovered)

#add an id column
corona_daily_by_country_recovered = corona_daily_by_country_recovered.merge(
    corona_daily_by_country_confirmed[['Lat', 'Long', 'case_lat_long_id']],
    left_on=['Lat', 'Long'], right_on=['Lat', 'Long'],
    how='left')

#melt
corona_daily_long_lat_recovered_melt = pd.melt(
    corona_daily_by_country_recovered,
    id_vars=('Country/Region', 'Lat', 'Long', 'case_lat_long_id')).add_prefix('cases_')

#Rename columns
corona_daily_long_lat_recovered_melt = corona_daily_long_lat_recovered_melt.rename(columns={
    'cases_variable': 'cases_date',
    'cases_value': 'recovered'})

#Add placeholder columns
corona_daily_long_lat_recovered_melt['confirmed'] = 0
corona_daily_long_lat_recovered_melt['deaths'] = 0


#%% concatenate the DataFrames
corona_daily_long_lat_melt = pd.concat([
    corona_daily_long_lat_recovered_melt, 
    corona_daily_long_lat_deaths_melt, 
    corona_daily_long_lat_confirmed_melt])

#Fill NAs
corona_daily_long_lat_melt[['recovered', 'deaths', 'confirmed']] = corona_daily_long_lat_melt[
    ['recovered', 'deaths', 'confirmed']].fillna(0)

corona_daily_long_lat_melt.dtypes

#filter any records which have zero confirmed, deaths, recovered...
corona_daily_long_lat_melt['total_observations'] = corona_daily_long_lat_melt[['recovered', 'deaths', 'confirmed']].sum(axis=1)
corona_daily_long_lat_melt = corona_daily_long_lat_melt[
    corona_daily_long_lat_melt['total_observations'] > 0]


#%% This is where we read our longitude latitude lookup file
input_directory_long_lat_lookup = 'simplemaps_worldcities_basicv1.6'
input_file_name_long_lat_lookup = "worldcities"


#Read in file, add column name prefix and  filter to desired number of rows
longitude_latitude_lookup = csv_contents_to_pandas_df(
    directory_name=input_directory_long_lat_lookup,
    file_name=input_file_name_long_lat_lookup).add_prefix(
        'mapped_city_')[0:num_to_lookup_long_and_lats]


#%% This is where we loop through all unique coordinates and apply meta data
if run_nearest_city_loop == True:
    corona_daily_with_meta_data = corona_daily_long_lat_melt

    output_dictionary_temp = {}
    unique_case_cordinates = corona_daily_long_lat_melt[[
        'cases_Lat', 'cases_Long', 'cases_case_lat_long_id']].drop_duplicates()[
            0:num_to_lookup_case_coordinates].reset_index()

    absolutestart = time.time()
    iterationstart = time.time()

    for index_cases, row_cases in unique_case_cordinates.iterrows():

        stop = time.time()
        iterationduration = stop-iterationstart
        totalduration = stop-absolutestart
        iterationstart = time.time()

        if index_cases != 0:
            if index_cases % round(len(unique_case_cordinates.index)/20, 0) == 0:
                print('-------------------------------------')
                print('last iteration took:', iterationduration,
                    'total runtime is currently:', totalduration/60, "minutes")
                print("this is location #", index_cases, "of", len(
                    unique_case_cordinates.index), "unique case locations recorded (long/lat)...")
                print("estimated time remaining:", iterationduration *
                    (len(unique_case_cordinates.index)-index_cases)/60, "minutes")
                print('-------------------------------------')

        #Create a placeholder for this iteration
        output_dictionary_temp[index_cases] = [0, 0, 0, 0, 0]

        #add the latitude longitude values for the unique case coordinate
        output_dictionary_temp[index_cases][0:3] = unique_case_cordinates.loc[index_cases][[
            'cases_Lat', 'cases_Long', 'cases_case_lat_long_id']].values.flatten().tolist()

        for index_lookup, row_lookup in longitude_latitude_lookup.iterrows():
            if index_lookup == 0:
                #adds the resulting 'distance' between both sets of lat/long
                output_dictionary_temp[index_cases][3] = distance_between_coordinates(
                    lat1=output_dictionary_temp[index_cases][0],
                    lon1=output_dictionary_temp[index_cases][1],
                    lat2=longitude_latitude_lookup.iloc[index_lookup]['mapped_city_lat'],
                    lon2=longitude_latitude_lookup.iloc[index_lookup]['mapped_city_lng'])

                #add the lookup tables id for thecorresponding item that's being looked up
                output_dictionary_temp[index_cases][4] = longitude_latitude_lookup.iloc[
                    index_lookup]['mapped_city_id']

            if index_lookup != 0:

                #using the current lookup case, calculate the distance
                new_index_lookup_distance = distance_between_coordinates(
                    lat1=output_dictionary_temp[index_cases][0],
                    lon1=output_dictionary_temp[index_cases][1],
                    lat2=longitude_latitude_lookup.iloc[index_lookup]['mapped_city_lat'],
                    lon2=longitude_latitude_lookup.iloc[index_lookup]['mapped_city_lng'])

                #add the lookup tables id for thecorresponding item that's being
                # looked up, but only if it is greater than the value already held
                if output_dictionary_temp[index_cases][3] > new_index_lookup_distance:
                    output_dictionary_temp[index_cases][3] = new_index_lookup_distance
                    output_dictionary_temp[index_cases][4] = longitude_latitude_lookup.iloc[
                        index_lookup]['mapped_city_id']

    #convert dictionary, change column names, drop extraneous columns
    corona_case_lookup_coordinates = pd.DataFrame.from_dict(output_dictionary_temp, orient="index").rename(columns={
        0: 'case_city_latitude',
        1: 'case_city_longitude',
        2: 'cases_case_lat_long_id',  # need it
        3: 'mapped_city_distance_from_original_case',
        4: 'mapped_city_id'})  # need it
    corona_case_lookup_coordinates = corona_case_lookup_coordinates.drop(
        ["case_city_latitude", "case_city_longitude",
        "mapped_city_distance_from_original_case"],
        axis=1)

    #Merge with meta data from coordinates lookup table, change data type
    corona_case_lookup_coordinates_with_meta = corona_case_lookup_coordinates.merge(
        longitude_latitude_lookup, on='mapped_city_id')
    corona_case_lookup_coordinates_with_meta['mapped_city_id'] = corona_case_lookup_coordinates_with_meta['mapped_city_id'].astype(
        'int64')

    # Merge the lookup table with the original case data
    corona_daily_with_meta_data = corona_daily_long_lat_melt.merge(
        corona_case_lookup_coordinates_with_meta, on='cases_case_lat_long_id')

if run_nearest_city_loop == False:
    corona_daily_with_meta_data = corona_daily_long_lat_melt

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
output_file_name_corona_case_with_meta_and_populations = 'corona_daily_long_lat_with_meta_and_populations.csv'
corona_daily_with_meta_data_and_populations = corona_daily_with_meta_data.merge(
    population_lookup_adjusted, left_on='cases_Country/Region', right_on='country_populations_Country'
    )

#This is because python isn't so good at identifying date types (for pragith)
corona_daily_with_meta_data_and_populations['cases_date'] = pd.to_datetime(
    corona_daily_with_meta_data_and_populations['cases_date'])

#find the max date and add it to the dataframe
max_date = max(corona_daily_with_meta_data_and_populations['cases_date'])
corona_daily_with_meta_data_and_populations['cases_max_date'] = max_date




#Fill NAs
temp_str_cols = corona_daily_with_meta_data_and_populations.columns[
    corona_daily_with_meta_data_and_populations.dtypes==object]
corona_daily_with_meta_data_and_populations[temp_str_cols] = corona_daily_with_meta_data_and_populations[
    temp_str_cols].fillna('missing')

temp_floatint_cols = 'mapped_city_population'
corona_daily_with_meta_data_and_populations[temp_floatint_cols] = corona_daily_with_meta_data_and_populations[
    temp_floatint_cols].fillna(0)

corona_daily_with_meta_data_and_populations[corona_daily_with_meta_data_and_populations['cases_Country/Region']=='Vietnam']


#aggregate for row reduction
corona_daily_with_meta_data_and_populations_aggregated = corona_daily_with_meta_data_and_populations.groupby([
    'cases_Country/Region', 
    'cases_Lat', 
    'cases_Long',
    'cases_case_lat_long_id', 
    'cases_date', 
    'mapped_city_id',
    'mapped_city_city',
    'mapped_city_city_ascii',
    'mapped_city_lat',
    'mapped_city_lng',
    'mapped_city_country',
    'mapped_city_iso2',
    'mapped_city_iso3',
    'mapped_city_admin_name',
    'mapped_city_capital',
    'mapped_city_population',
    'country_populations_cca2',
    'country_populations_Country',
    'country_populations_pop2020', 
    'country_populations_dropdownData',
    'country_populations_area', 
    'country_populations_Density',
    'country_populations_GrowthRate', 
    'country_populations_WorldPercentage',
    'country_populations_rank', 
    'country_populations_worldPercentage_updated',
    'country_populations_rank_updated'
    ]).agg(recovered=('recovered', 'sum'),
            confirmed=('confirmed', 'sum'),
            deaths=('deaths', 'sum'),
            total_observations=('total_observations', 'sum')
            ).reset_index()



#%% Write to CSV
corona_daily_with_meta_data_and_populations_aggregated.to_csv(
    output_file_name_corona_case_with_meta_and_populations, index=False)
