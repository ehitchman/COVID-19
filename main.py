#%% import some libraries...
from math import cos, asin, sqrt
import pandas as pd
import numpy as np
import csv
import time
import pycountry_convert
from kaggle.api.kaggle_api_extended import KaggleApi
pd.options.display.float_format = '{:.6f}'.format


#%%Define functions
def csv_contents_to_pandas_df(directory_name, file_name):
    '''Function to read and assign csv file contents to pandas df'''
    try:
        with open(directory_name + '/' + file_name.replace('.csv.csv', '').replace('.csv', '') + '.csv', 'rb') as file_obj:
            temp_df = pd.read_csv(file_obj, keep_default_na=False)
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
        'country_populations_rank': [0]
        })

    #calculate density
    temp_df_row_to_add.at[0, 'country_populations_Density'] = temp_df_row_to_add.loc[0]['country_populations_pop2020'] / temp_df_row_to_add.loc[0]['country_populations_area']

    #concatenate dfs
    temp_concatenated_df = pd.concat([
        population_lookup_df,
        temp_df_row_to_add
    ]).reset_index(drop=True)

    #Calculate world percentage
    temp_concatenated_df['country_populations_worldPercentage'] = (temp_concatenated_df['country_populations_pop2020'] / \
        temp_concatenated_df['country_populations_pop2020'].sum()).astype(float)

    #Calculate the updated rank
    temp_concatenated_df['country_populations_rank'] = temp_concatenated_df.country_populations_pop2020.rank(
        ascending=False, method='dense').astype(int)

    return temp_concatenated_df


def update_country_name_in_population_lookup(
    population_lookup_df, 
    original_to_new_country_name_and_code_df):

    print('------------------------------------------------------')
    print("this is population_lookup_df")
    print(population_lookup_df, '\n')
    print('------------------------------------------------------')
    print("this is original_to_new_country_name_and_code_df")
    print(original_to_new_country_name_and_code_df, '\n')

    final_df = population_lookup_df.merge(
        original_to_new_country_name_and_code_df,
        left_on='country_populations_Country',
        right_on='orig_countryName',
        how="left")

    print('------------------------------------------------------')
    print('\n', 'this is final DF after the merge...')
    print(final_df)

    #Here is a statement similar to coalesce which prioritizes the lookup value,
    # and if no lookup value is found, the original value is used in its place
    final_df['country_populations_countryName_final'] = final_df[['orig_countryName', 'upd_countryName', 'country_populations_Country']].apply(
        lambda x: x['upd_countryName'] if pd.isnull(x[
            'orig_countryName']) else x['country_populations_Country'], axis=1)

    final_df['country_populations_countryCode_final'] = final_df[['orig_countryName', 'upd_countryCode', 'country_populations_cca2']].apply(
        lambda x: x['upd_countryCode'] if pd.isnull(x[
            'orig_countryName']) else x['country_populations_cca2'], axis=1)

    #Aggregate adjusted population table to ensure there are no duplicates
    final_df_aggregated = final_df.groupby([
        'country_populations_countryCode_final',
        'country_populations_countryName_final'
    ]).agg(country_populations_pop2020=('country_populations_pop2020', 'sum'),
           country_populations_area=('country_populations_area', 'sum')
           ).reset_index()

    print('------------------------------------------------------')
    print('\n', 'this is final DF after the aggregation...')
    print(final_df_aggregated)

    #calculate density
    final_df_aggregated['country_populations_Density'] = final_df_aggregated['country_populations_pop2020'] / final_df_aggregated['country_populations_area']

    #Calculate world percentage
    final_df_aggregated['country_populations_worldPercentage'] = (final_df_aggregated['country_populations_pop2020'] / final_df_aggregated['country_populations_pop2020'].sum()).astype(float)

    #Calculate the updated rank
    final_df_aggregated['country_populations_rank'] = final_df_aggregated.country_populations_pop2020.rank(
        ascending=False, 
        method='dense').astype(int)

    print('------------------------------------------------------')
    print('this is final DF after the updated calculations...')
    print(final_df_aggregated)

    return final_df_aggregated


def add_row_to_continent_lookup(
    continent_lookup_df, 
    CountryName, 
    CapitalName, 
    CaptialLatitude, 
    CapitalLongitude, 
    CountryCode, 
    ContinentName):
    '''Function to add a row to an existing population lookup table.  First,
        create the data frame, then do any calculations, say world percentage,
        rank, population density, then create the output df which is the
        original along with the added row'''

    #Creat the df to add to the original file
    temp_df_row_to_add = pd.DataFrame({
        'continent_CountryName': [CountryName],
        'continent_CapitalName': [CapitalName],
        'continent_CaptialLatitude':  [CaptialLatitude],
        'continent_CapitalLongitude': [CapitalLongitude],
        'continent_CountryCode': [CountryCode],
        'continent_ContinentName': [ContinentName]
        })

    #concatenate dfs
    temp_concatenated_df = pd.concat([
        continent_lookup_df,
        temp_df_row_to_add
    ]).reset_index(drop=True)

    return temp_concatenated_df


def download_csv_from_kaggle(dataset, filename, path, force):
    api = KaggleApi()
    api.authenticate()

    print('filename being checked:', filename)
    filename = filename.replace('.csv.csv', '').replace('.csv','') + '.csv'

    print('directory being checked:', path)    
    print('filename being checked', filename)

    temp_dataset = api.dataset_download_file(
        dataset = dataset, 
        file_name = filename,
        path=path,
        force=force, 
        quiet=False)

    if temp_dataset == False:
        temp_dataset = csv_contents_to_pandas_df(
            directory_name=path, file_name=filename)

    return temp_dataset


#%% Set some params...
#primary parameters for script...
input_directory_name_corona_cases = 'csse_covid_19_data/csse_covid_19_time_series'
output_final_directory = 'output'
population_data_directory = 'country_population_data'
country_to_continent_file = 'concap.csv'
country_to_continent_directory = 'country_to_continent_mapping'
run_nearest_city_loop = False

cases_column_order = [
    'cases_date',
    'cases_Province/State', 
    'cases_Country/Region', 
    'cases_Lat', 
    'cases_Long', 
    'cases_case_lat_long_id', 
    'deaths', 
    'confirmed', 
    'recovered',
    ]

#secondary parameters related to QA
output_qc_directory = 'qc'
num_to_lookup_long_and_lats = None
num_to_lookup_case_coordinates = None
run_script_printouts_and_write_qc_files = True


#%% This is where we read the daily case files and melt them
input_file_name_corona_case_confirmed = "time_series_19-covid-Confirmed.csv"
corona_daily_by_country_confirmed = csv_contents_to_pandas_df(
    directory_name=input_directory_name_corona_cases, 
    file_name=input_file_name_corona_case_confirmed)

#add an id column
corona_daily_by_country_confirmed['case_lat_long_id'] = corona_daily_by_country_confirmed.groupby(
    ['Lat', 'Long']).ngroup()

#melt
corona_daily_by_country_confirmed_melt = pd.melt(
    corona_daily_by_country_confirmed,
    id_vars=('Province/State', 'Country/Region', 'Lat', 'Long', 'case_lat_long_id')).add_prefix('cases_')

#Rename columns and assign zero values to additional columns, order columns
corona_daily_by_country_confirmed_melt = corona_daily_by_country_confirmed_melt.rename(columns={
    'cases_variable': 'cases_date',
    'cases_value': 'confirmed'})
corona_daily_by_country_confirmed_melt['deaths'] = 0
corona_daily_by_country_confirmed_melt['recovered'] = 0
corona_daily_by_country_confirmed_melt = corona_daily_by_country_confirmed_melt[
    cases_column_order]

#print details about the file read in.
if run_script_printouts_and_write_qc_files == True:
    print('------------------------------------------------------') 
    print('corona_daily_by_country_confirmed_melt', 'dtypes', '\n')
    print(corona_daily_by_country_confirmed_melt.dtypes)


#%% This is where we read the daily death case files and melt them
input_file_name_corona_case_deaths = "time_series_19-covid-Deaths"
corona_daily_by_country_deaths = csv_contents_to_pandas_df(
    directory_name=input_directory_name_corona_cases,
    file_name=input_file_name_corona_case_deaths)

#add an id column
corona_daily_by_country_deaths = corona_daily_by_country_deaths.merge(
    corona_daily_by_country_confirmed[['Lat', 'Long', 'case_lat_long_id']],
     left_on=['Lat', 'Long'], right_on=['Lat', 'Long'],
     how='left')

#melt
corona_daily_by_country_deaths_melt = pd.melt(
    corona_daily_by_country_deaths,
    id_vars=('Province/State', 'Country/Region', 'Lat', 'Long', 'case_lat_long_id')).add_prefix('cases_')

#weird data type issue (results as "object" rather than "int")
corona_daily_by_country_deaths_melt['cases_value'] = pd.to_numeric(
    corona_daily_by_country_deaths_melt['cases_value'], errors='coerce')

#Rename columns and assign zero values to additional columns
corona_daily_by_country_deaths_melt = corona_daily_by_country_deaths_melt.rename(columns={
    'cases_variable': 'cases_date',
    'cases_value': 'deaths'})
corona_daily_by_country_deaths_melt['confirmed'] = 0
corona_daily_by_country_deaths_melt['recovered'] = 0
corona_daily_by_country_deaths_melt = corona_daily_by_country_deaths_melt[
    cases_column_order]

#print details about the file read in.
if run_script_printouts_and_write_qc_files == True:
    print('------------------------------------------------------')
    print('corona_daily_by_country_deaths_melt', 'dtypes', '\n')
    print(corona_daily_by_country_deaths_melt.dtypes)


#%% This is where we read the daily recovered case files and melt them
input_file_name_corona_case_recovered = "time_series_19-covid-Recovered"

corona_daily_by_country_recovered = csv_contents_to_pandas_df(
    directory_name=input_directory_name_corona_cases,
    file_name=input_file_name_corona_case_recovered)

#add an id column
corona_daily_by_country_recovered = corona_daily_by_country_recovered.merge(
    corona_daily_by_country_confirmed[['Lat', 'Long', 'case_lat_long_id']],
    left_on=['Lat', 'Long'], right_on=['Lat', 'Long'],
    how='left')

#melt
corona_daily_by_country_recovered_melt = pd.melt(
    corona_daily_by_country_recovered,
    id_vars=('Province/State', 'Country/Region', 'Lat', 'Long', 'case_lat_long_id')).add_prefix('cases_')

#Rename columns and assign zero values to additional columns
corona_daily_by_country_recovered_melt = corona_daily_by_country_recovered_melt.rename(columns={
    'cases_variable': 'cases_date',
    'cases_value': 'recovered'})
corona_daily_by_country_recovered_melt['confirmed'] = 0
corona_daily_by_country_recovered_melt['deaths'] = 0
corona_daily_by_country_recovered_melt = corona_daily_by_country_recovered_melt[
    cases_column_order]

#print details about the file read in.
if run_script_printouts_and_write_qc_files == True:
    print('------------------------------------------------------')
    print('corona_daily_by_country_recovered_melt', 'dtypes', '\n')
    print(corona_daily_by_country_recovered_melt.dtypes)


#%% concatenate the DataFrames
corona_daily_by_country_totals = pd.concat([
    corona_daily_by_country_recovered_melt, 
    corona_daily_by_country_deaths_melt, 
    corona_daily_by_country_confirmed_melt])

#filter any records which have zero confirmed, deaths, recovered...
corona_daily_by_country_totals['total_observations'] = corona_daily_by_country_totals[
    ['recovered', 'deaths', 'confirmed']].sum(axis=1)
corona_daily_by_country_totals = corona_daily_by_country_totals[
    corona_daily_by_country_totals['total_observations'] > 0]

#print details about the file to be written to sv.
if run_script_printouts_and_write_qc_files == True:
    print('------------------------------------------------------')
    print('corona_daily_by_country_totals', 'dtypes', '\n')
    print(corona_daily_by_country_totals.dtypes)


#%% This is where we read our population file and merge it with the corona cases
# Here we also 1) add rows to our population table and 2) adjust the names of 
#  countries in our population table
input_directory_population_lookup = 'population_data'
input_file_name_population_lookup = "country-population-data"

#Read in file
population_data = csv_contents_to_pandas_df(
    directory_name=input_directory_population_lookup,
    file_name=input_file_name_population_lookup).add_prefix('country_populations_')

#Add a row to the country population file for any recorded cases which may not 
# have a respective 'country'
population_data = add_row_to_population_lookup(
    population_lookup_df = population_data,
    cca2='n/a',
    Country='Cruise Ship',
    pop2020=3.770,
    dropdownData='n/a',
    area=0.19314,
    GrowthRate = 0,         
)

#Write the qc file to csv
if run_script_printouts_and_write_qc_files == True:
    population_data.to_csv(
        output_qc_directory + '/' + 'population_data_with_added_countries_qc1.csv',
        index=False)

#build the data frame for joining and reconciliing country names
data = {'orig_countryName': ['United States', 'Macedonia', 'South Korea',
                             "Czech Republic", "DR Congo", "Ivory Coast",
                             "Reunion", "Guadeloupe", "Martinique",
                             "French Guiana", "Mayotte"],
        'upd_countryName': ['US', 'North Macedonia', 'Korea, South', 'Czechia',
                            'Congo (Kirshasa)', "Cote d'lvoire", 'France',
                            'France', 'France', 'France', 'France'],
        'upd_countryCode': ['US', 'MK', 'KR', 'CZ', 'CD', 'CI',
                            'FR', 'FR', 'FR', 'FR', 'FR']}

temp_country_replacement_df = pd.DataFrame(data)


#%% Update incorrect country labels (i.e. some labels are regions/provinces instead
# of countries)
population_data_with_added_countries_updated_country_names = update_country_name_in_population_lookup(
    population_lookup_df = population_data,
    original_to_new_country_name_and_code_df = temp_country_replacement_df)

    
#%% print details about the file to be written to csv.
if run_script_printouts_and_write_qc_files == True:
    print('------------------------------------------------------')
    print('\n','population_data_with_added_countries_updated_country_names',
        'dtypes')
    print(population_data_with_added_countries_updated_country_names.dtypes)

if run_script_printouts_and_write_qc_files == True:
    population_data_with_added_countries_updated_country_names.to_csv(
        output_qc_directory + '/' + 'population_data_with_added_countries_updated_country_names_qc1.csv',
        index=False)

print('------------------------------------------------------')
print('population_data_with_added_countries_updated_country_names number of countries:', 
len(population_data_with_added_countries_updated_country_names.index))


#%%Here we identify the continent name for each country and merge it with the population data set
country_to_continent_mapping = download_csv_from_kaggle(
    dataset='nikitagrec/world-capitals-gps', 
    filename=country_to_continent_file,
    path=country_to_continent_directory,
    force=False).add_prefix("continent_")


#%% Add missing rows to country_to_continent_mapping file
# data = {'CountryName': ['a','1'],
#         'CapitalName': ['b','2'],
#         'CapitalLatitude': ['c','3'],
#         'CapitalLongitude': ['d','4'],
#         'CountryCode': ['e','5'],
#         'ContinentName':['f','6']}

# country_to_continent_mapping_updated = apply(
#     lambda a,b,c,d,e,f: )

# add_row_to_continent_lookup(
#     continent_lookup_df=country_to_continent_mapping,
#     CountryName, 
#     CapitalName, 
#     CaptialLatitude, 
#     CapitalLongitude, 
#     CountryCode, 
#     ContinentName)

print('------------------------------------------------------')
population_data_with_added_countries_updated_country_names_with_continent = population_data_with_added_countries_updated_country_names.merge(
    country_to_continent_mapping,
    left_on='country_populations_countryCode_final',
    right_on='continent_CountryCode')
print('population_data_with_added_countries_updated_country_names_with_continent number of countries:',
    len(population_data_with_added_countries_updated_country_names_with_continent.index))
    
len(population_data_with_added_countries_updated_country_names_with_continent.index)


print('------------------------------------------------------')
population_data_with_added_countries_updated_country_names_with_continent_left = population_data_with_added_countries_updated_country_names.merge(
    country_to_continent_mapping,
    how='left',
    left_on='country_populations_countryCode_final',
    right_on='continent_CountryCode')
print('population_data_with_added_countries_updated_country_names_with_continent_left number of countries:', len(population_data_with_added_countries_updated_country_names_with_continent_left.index))

population_data_with_added_countries_updated_country_names_with_continent_left.to_csv(
    output_qc_directory + '/' + 'population_data_with_added_countries_updated_country_names_with_continent_qc1a_left.csv',
    index=False)

print('------------------------------------------------------')
population_data_with_added_countries_updated_country_names_with_continent_right = population_data_with_added_countries_updated_country_names.merge(
    country_to_continent_mapping,
    how='right',
    left_on='country_populations_countryCode_final',
    right_on='continent_CountryCode')
print('population_data_with_added_countries_updated_country_names_with_continent_right number of countries:', (population_data_with_added_countries_updated_country_names_with_continent_right.index))

population_data_with_added_countries_updated_country_names_with_continent_right.to_csv(
    output_qc_directory + '/' + 'population_data_with_added_countries_updated_country_names_with_continent_qc1b_right.csv',
    index=False)


#%% Merge the daily case totals and country population files and write to csv
output_file_name_corona_case_with_meta_and_populations = 'corona_cases_daily_with_populations.csv'
corona_daily_by_country_totals_and_populations = corona_daily_by_country_totals.merge(
    population_data_with_added_countries_updated_country_names, 
    left_on='cases_Country/Region', right_on='country_populations_countryName_final')

#This is to help identify the date type (for pp)
corona_daily_by_country_totals_and_populations['cases_date'] = pd.to_datetime(
    corona_daily_by_country_totals_and_populations['cases_date'])

#find the max date and add it to the dataframe
max_date = max(corona_daily_by_country_totals_and_populations['cases_date'])
corona_daily_by_country_totals_and_populations['cases_max_date'] = max_date

#Write the file to csv
# TODO This is a temporary step
if run_script_printouts_and_write_qc_files == True:
    corona_daily_by_country_totals_and_populations.to_csv(
        output_qc_directory + '/' + 'corona_cases_daily_with_populations_qc2_preaggregate.csv', 
        index=False)

#print details about the file to be written to sv.
if run_script_printouts_and_write_qc_files == True:
    print('------------------------------------------------------') 
    print('\n', 'corona_daily_by_country_totals_and_populations', 'dtypes')
    print(corona_daily_by_country_totals_and_populations.dtypes)


#%% Identify columns with nulls, and Fill 'NA', 'missing' or 'zero'
null_str_fill_value = 'missing'
null_float_fill_value = 0

#identify str and float columns with nulls
temp_columns_with_nulls = corona_daily_by_country_totals_and_populations.columns[
    corona_daily_by_country_totals_and_populations.isna().any()].tolist()
temp_columns_with_nulls_float = corona_daily_by_country_totals_and_populations[temp_columns_with_nulls].columns[
    corona_daily_by_country_totals_and_populations[temp_columns_with_nulls].dtypes == float]
temp_columns_with_nulls_str = corona_daily_by_country_totals_and_populations[temp_columns_with_nulls].columns[
    corona_daily_by_country_totals_and_populations[temp_columns_with_nulls].dtypes == object]
    
#apply default value for str/float columns with nulls from list above
corona_daily_by_country_totals_and_populations[temp_columns_with_nulls_str] = corona_daily_by_country_totals_and_populations[
    temp_columns_with_nulls_str].fillna(null_str_fill_value)
corona_daily_by_country_totals_and_populations[temp_columns_with_nulls_float] = corona_daily_by_country_totals_and_populations[
    temp_columns_with_nulls_float].fillna(null_float_fill_value)

#Write the file to csv
# TODO This is a temporary step
if run_script_printouts_and_write_qc_files == True:
    corona_daily_by_country_totals_and_populations.to_csv(
        output_qc_directory + '/' + 'corona_cases_daily_with_populations_qc3_preaggregate_filled_nas.csv',
        index=False)
        

#%%aggregate for row reduction
corona_daily_by_country_totals_and_populations_aggregated = corona_daily_by_country_totals_and_populations.groupby([
    'cases_date',
    'cases_max_date',
    'cases_Lat',
    'cases_Long',
    'cases_case_lat_long_id',
    'cases_Country/Region',
    'cases_Province/State',
    'country_populations_pop2020',
    'country_populations_area',
    'country_populations_Density',
    #'country_populations_GrowthRate',
    'country_populations_worldPercentage_updated',
    'country_populations_rank_updated'
]).agg(recovered=('recovered', 'sum'),
       confirmed=('confirmed', 'sum'),
       deaths=('deaths', 'sum')
       ).reset_index()

#Write the file to csv
# TODO This is a temporary step
if run_script_printouts_and_write_qc_files == True:
    corona_daily_by_country_totals_and_populations_aggregated.to_csv(
        output_qc_directory + '/' + 'corona_cases_daily_with_populations_qc4_postaggregate.csv', 
        index=False)


#%% Add a flag for when total confirmed cases reached N for each country
output_file_name_corona_case_with_meta_and_populations_and_flags = 'corona_cases_daily_with_populations_and_flags.csv'
num_cases_to_start = 10
corona_daily_by_country_totals_and_populations_aggregated['10_or_more_confirmed_cases'] = np.where(
    corona_daily_by_country_totals_and_populations_aggregated.confirmed >= num_cases_to_start, 
    True, False)

#Assign a sequential number to each date based on the 
# "confirmed_cases_less_than_or_greater_than_10"
corona_daily_by_country_totals_and_populations_aggregated['10_or_more_confirmed_cases_day_count'] = corona_daily_by_country_totals_and_populations_aggregated.groupby(
    ['cases_Country/Region', '10_or_more_confirmed_cases'])['cases_date'].rank(method='dense')

#Write the file to csv
# TODO This is a temporary step
if run_script_printouts_and_write_qc_files == True:
    corona_daily_by_country_totals_and_populations_aggregated.to_csv(
        output_qc_directory + '/' + 'corona_cases_daily_with_populations_qc5_postaggregate_with_flags.csv',
        index=False)

#%%Write the final aggregated dataframe to csv
corona_daily_by_country_totals_and_populations_aggregated.to_csv(
    output_final_directory + '/' + output_file_name_corona_case_with_meta_and_populations_and_flags,
    index=False)

#%%
print('all done here...')
