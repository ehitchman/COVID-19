#%% import some libraries and the local functions file...
from math import cos, asin, sqrt
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import csv, time, pycountry_convert, requests, json, pprint
import functions
import datetime

################################################################################
#US or Global Pull? TODO -- adjust for 'both' ##################################
input_option_us_or_global = 'global'
################################################################################
################################################################################

#%% Set some params for directires, options for core scripts & QA printouts,
# starting with generic options to help with data type printouts
pd.options.display.float_format = '{:.6f}'.format

#directories
input_directory_corona_cases = 'csse_covid_19_data\csse_covid_19_time_series'
input_directory_population_data = 'population_data'
input_directory_country_to_continent = 'country_to_continent_mapping'
output_directory_final_data = 'output'

#File Names
input_file_name_corona_case_confirmed = "time_series_covid19_confirmed_global.csv"
output_file_name_corona_case_with_meta_and_populations_and_flags = 'corona_cases_daily_with_populations_and_flags.csv'
input_file_name_population_lookup = "country-population-data"

#Script Options
run_nearest_city_loop = False
input_case_types = ['deaths','confirmed','recovered']

#QA Options
output_qc_directory = 'qc'
num_to_lookup_long_and_lats = None
num_to_lookup_case_coordinates = None
run_script_printouts_and_write_qc_files = False

#Read data in from the forked repo, process it and write it to the output folder
cases_data_daily_by_country = functions.read_cases_data(
    input_directory_corona_cases=input_directory_corona_cases,
    us_or_global=input_option_us_or_global,
    case_types=input_case_types)

#%% This is where we read our population file and merge it with the corona cases
# Here we also 1) add rows to our population table and 2) adjust the names of 
#  countries in our population table
#Read in file
population_data = functions.csv_contents_to_pandas_df(
    directory_name=input_directory_population_data,
    file_name=input_file_name_population_lookup).add_prefix('country_populations_')

#Add a row to the country population file for any recorded cases which may not 
# have a respective 'country'
population_data = functions.add_row_to_population_lookup(
    population_lookup_df = population_data,
    cca2='n/a',
    Country='Cruise Ship',
    pop2020=3.770,
    dropdownData='n/a',
    area=0.19314,
    GrowthRate = 0,)

#Update incorrect country labels (i.e. some labels are regions/provinces instead
# of countries, or their names neeed to be updated to match the corona cases file)
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

#Update country names in the population data file using the df created from the dict
population_data_with_added_countries_updated_country_names = functions.update_country_name_in_country_population_lookup(
    population_lookup_df = population_data,
    original_to_new_country_name_and_code_df = temp_country_replacement_df)

#Here we identify the continent name for each country and merge it with the 
# population data set
kaggle_country_to_continent_dataset = 'andradaolteanu/country-mapping-iso-continent-region'
kaggle_country_to_continent_file = 'continents2'
country_to_continent_mapping = functions.download_csv_from_kaggle(
    dataset=kaggle_country_to_continent_dataset,
    filename=kaggle_country_to_continent_file,
    path=input_directory_country_to_continent,
    force=False).add_prefix("continent_")

#build the data frame for joining and reconciliing country names
data = {'CountryName': ['Cruise Ship'],
        'CapitalName': ['None'],
        'CapitalLongitude': [None],
        'CaptialLatitude': [None],
        'CountryCode': ['None'],
        'ContinentName': ['None']
        }
temp_country_to_continent_rows_to_add = pd.DataFrame(data)

#Update country names in the population data file
country_to_continent_mapping_added_rows = functions.add_row_to_continent_lookup(
    continent_lookup_df=country_to_continent_mapping,
    CountryName='Cruise Ship',
    CapitalName='None',
    CapitalLongitude=None,
    CaptialLatitude=None,
    CountryCode='None',
    ContinentName='None')

#Add missing rows to country_to_continent_mapping file
population_data_with_added_countries_updated_country_names_with_continent = population_data_with_added_countries_updated_country_names.merge(
    country_to_continent_mapping,
    left_on='country_populations_countryCode_final',
    right_on='continent_alpha-2')

#%% Merge the daily case totals and country population files and write to csv
#TODO -- clean up the 'global' vs. 'us' files prior to transforming (in functions.py)
output_file_name_corona_case_with_meta_and_populations = 'corona_cases_daily_with_populations.csv'
if input_option_us_or_global.lower() == 'us':
    cases_data_daily_by_country_and_populations = cases_data_daily_by_country.merge(
        population_data_with_added_countries_updated_country_names_with_continent,
        left_on='cases_Country_Region',
        right_on='country_populations_countryName_final')
elif input_option_us_or_global.lower() == 'global':
    cases_data_daily_by_country_and_populations = cases_data_daily_by_country.merge(
        population_data_with_added_countries_updated_country_names_with_continent,
        left_on='cases_Country/Region', 
        right_on='country_populations_countryName_final')

#This is to help identify the date type (for pp)
cases_data_daily_by_country_and_populations['cases_date'] = pd.to_datetime(
    cases_data_daily_by_country_and_populations['cases_date'])

#find the max date and add it to the dataframe
cases_data_daily_by_country_and_populations['cases_max_date'] = max(
    cases_data_daily_by_country_and_populations['cases_date'])

#%%aggregate for row reduction
#TODO -- clean up the 'global' vs. 'us' files prior to transforming (in functions.py)
if input_option_us_or_global.lower() == 'us':
    cases_data_daily_by_country_and_populations_aggregated = cases_data_daily_by_country_and_populations.groupby([
        'cases_date',
        'cases_max_date',
        'cases_Lat',
        'cases_Long_',
        'cases_lat_long_id',
        'cases_Country_Region',
        'cases_Province_State',
        'country_populations_pop2020',
        'country_populations_area',
        'country_populations_Density',
        'country_populations_worldPercentage',
        'country_populations_rank',
        'continent_sub-region',
        'continent_continentName'
    ]).agg(
        cases_confirmed=('cases_confirmed', 'sum'),
        cases_deaths=('cases_deaths', 'sum')
        ).reset_index()
elif input_option_us_or_global.lower() == 'global':
    cases_data_daily_by_country_and_populations_aggregated = cases_data_daily_by_country_and_populations.groupby([
        'cases_date',
        'cases_max_date',
        'cases_Lat',
        'cases_Long',
        'cases_lat_long_id',
        'cases_Country/Region',
        'cases_Province/State',
        'country_populations_pop2020',
        'country_populations_area',
        'country_populations_Density',
        'country_populations_worldPercentage',
        'country_populations_rank',
        'continent_sub-region',
        'continent_continentName'
    ]).agg(
        cases_recovered=('cases_recovered', 'sum'),
        cases_confirmed=('cases_confirmed', 'sum'),
        cases_deaths=('cases_deaths', 'sum')
        ).reset_index()

#%% Download and merge the Canada Province/Territory populations
canada_province_territory_populations = functions.download_stats_canada_provincial_populations(
    output_file_name='canada_province_population_data.csv',
    input_directory_population_data=input_directory_population_data,
    stats_canada_data_year='2016').add_prefix('canada_')

#TODO -- clean up the 'global' vs. 'us' files prior to transforming (in functions.py)
if input_option_us_or_global.lower() == 'us':
    cases_data_daily_by_country_and_populations_aggregated_with_canada_province_populations = cases_data_daily_by_country_and_populations_aggregated.merge(
        canada_province_territory_populations,
        how='left',
        left_on='cases_Province_State',
        right_on='canada_PROV_TERR_NAME_NOM')
elif input_option_us_or_global.lower() == 'global':
    cases_data_daily_by_country_and_populations_aggregated_with_canada_province_populations = cases_data_daily_by_country_and_populations_aggregated.merge(
        canada_province_territory_populations,
        how='left',
        left_on='cases_Province/State',
        right_on='canada_PROV_TERR_NAME_NOM')

print('finished joining provincial population data')
print('-----------------------------------------')

#%% Add a flag for when total confirmed cases reached N for each country and
# then Assign a sequential number to each date based on the
#Name of output file
cases_data_daily_by_country_and_populations_aggregated_with_canada_province_populations_and_flags = functions.add_flag_for_n_cases_date(
    cases_dataframe=cases_data_daily_by_country_and_populations_aggregated_with_canada_province_populations,
     n_list=[10, 25, 100])

#%%Write the final aggregated dataframe to csv
cases_data_daily_by_country_and_populations_aggregated_with_canada_province_populations_and_flags.to_csv(
    output_directory_final_data + '/' + input_option_us_or_global + "_" + output_file_name_corona_case_with_meta_and_populations_and_flags,
    index=False)

#%% indicates completion
print('all done here...')

#%%
