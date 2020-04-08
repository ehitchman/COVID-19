
#%% import some libraries and the local functions file...
from math import cos, asin, sqrt
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import csv
import time
import datetime
import pycountry_convert
import requests
import json
import pprint

#%%Define functions
def csv_contents_to_pandas_df(directory_name, file_name):
    '''Function to read and assign csv file contents to pandas df'''

    print("directory name:", directory_name)
    print("file name:", file_name)
    print("file object:", directory_name + '/' +
          file_name.replace('.csv.csv', '').replace('.csv', '') + '.csv')
    try:
        with open(directory_name + '/' + file_name.replace('.csv.csv', '').replace('.csv', '') + '.csv', 'rb') as file_obj:
            temp_df = pd.read_csv(file_obj, keep_default_na=False)
            file_obj.close()
    except FileNotFoundError:
        print("File not found")
    return temp_df


def cleanse_data(data):
    return None

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
    temp_df_row_to_add.at[0, 'country_populations_Density'] = temp_df_row_to_add.loc[0]['country_populations_pop2020'] / \
        temp_df_row_to_add.loc[0]['country_populations_area']

    #concatenate dfs
    temp_concatenated_df = pd.concat([
        population_lookup_df,
        temp_df_row_to_add
    ]).reset_index(drop=True)

    #Calculate world percentage
    temp_concatenated_df['country_populations_worldPercentage'] = (temp_concatenated_df['country_populations_pop2020'] /
                                                                   temp_concatenated_df['country_populations_pop2020'].sum()).astype(float)

    #Calculate the updated rank
    temp_concatenated_df['country_populations_rank'] = temp_concatenated_df.country_populations_pop2020.rank(
        ascending=False, method='dense').astype(int)

    return temp_concatenated_df


def update_country_name_in_country_population_lookup(
        population_lookup_df,
        original_to_new_country_name_and_code_df,
        run_script_printouts_and_write_qc_files=True):

    final_df = population_lookup_df.merge(
        original_to_new_country_name_and_code_df,
        left_on='country_populations_Country',
        right_on='orig_countryName',
        how="left")

    #Here is a statement similar to coalesce which prioritizes the lookup value,
    # and if no lookup value is found, the original value is used in its place
    final_df['country_populations_countryName_final'] = np.where(
        final_df["upd_countryName"].isnull(), final_df["country_populations_Country"], final_df["upd_countryName"])
    final_df['country_populations_countryCode_final'] = np.where(
        final_df["upd_countryName"].isnull(), final_df["country_populations_cca2"], final_df["upd_countryCode"])

    #Aggregate adjusted population table to ensure there are no duplicates
    # Note, that we use sum() because there are multiple provinces/states per
    # country.
    final_df_aggregated = final_df.groupby([
        'country_populations_countryCode_final',
        'country_populations_countryName_final'
    ]).agg(country_populations_pop2020=('country_populations_pop2020', 'sum'),
           country_populations_area=('country_populations_area', 'sum')
           ).reset_index()

    #calculate the density
    final_df_aggregated['country_populations_Density'] = final_df_aggregated['country_populations_pop2020'] / \
        final_df_aggregated['country_populations_area']

    #Calculate the updated world percentage
    final_df_aggregated['country_populations_worldPercentage'] = (
        final_df_aggregated['country_populations_pop2020'] / final_df_aggregated['country_populations_pop2020'].sum()).astype(float)

    #Calculate the updated rank
    final_df_aggregated['country_populations_rank'] = final_df_aggregated.country_populations_pop2020.rank(
        ascending=False,
        method='dense').astype(int)

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
        'continent_alpha-2': [CountryCode],
        'continent_name': [ContinentName]
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
    filename = filename.replace('.csv.csv', '').replace('.csv', '') + '.csv'

    temp_dataset = api.dataset_download_file(
        dataset=dataset,
        file_name=filename,
        path=path,
        force=force,
        quiet=False)

    if temp_dataset == False:
        temp_dataset = csv_contents_to_pandas_df(
            directory_name=path, file_name=filename)

    return temp_dataset


def download_stats_canada_provincial_populations(
        output_file_name='canada_province_population_data.csv',
        input_directory_population_data='population_data',
        stats_canada_data_year='2016'):
    '''
    Example URL: https://www12.statcan.gc.ca/rest/census-recensement/CPR2016.json?lang=E&dguid=2016A000224&topic=13&stat=0
    '''

    #Get the geouid from statscan
    basestatscanurl = 'https://www12.statcan.gc.ca/rest/census-recensement/'
    finalurl = basestatscanurl + 'CR2016Geo.json' + '?' + \
        'lang=E' + '&' + 'geos=PR' + '&' + 'cpt=00'
    response_text = requests.get(finalurl).text.replace('//', '')
    response_json = json.loads(response_text)
    df_geouid = pd.DataFrame(
        response_json['DATA'], columns=response_json['COLUMNS'])

    df_geouid_noncanada = df_geouid[df_geouid['PROV_TERR_NAME_NOM'] != 'Canada']
    list_geoid_noncanada = df_geouid_noncanada['GEO_UID']

    #TODO -- may be redundant 
    def pull_populations_from_stats_canada_based_on_dguid(dguid):
        print('---------')
        basestatscanurl = 'https://www12.statcan.gc.ca/rest/census-recensement/'
        finalurl = basestatscanurl + 'CPR2016.json' + '?' + 'lang=E' + \
            '&' + 'dguid=' + dguid + '&' + 'topic=13' + '&' + 'stat=0'
        print(finalurl)
        response = requests.get(finalurl)
        response_text = response.text.replace('//', '')
        response_json = json.loads(response_text)

        df = pd.DataFrame(response_json['DATA'],
                          columns=response_json['COLUMNS'])
        df = df[df['TEXT_NAME_NOM'] == 'Population, '+stats_canada_data_year]

        return(df)

    #Use the list of geouids to grab the population data
    list_of_dfs = [pull_populations_from_stats_canada_based_on_dguid(
        x) for x in list_geoid_noncanada]

    #concatenate the list of dfs
    df = pd.concat(list_of_dfs, ignore_index=True)
    df.to_csv(input_directory_population_data + '/' + output_file_name)
    print(df)

    return(df)


#%% This is where we read the daily case files and melt them
def read_cases_data(
        input_directory_corona_cases,
        us_or_global='global',
        case_types = ['deaths','confirmed','recovered']):
    
    #test
    # input_directory_corona_cases = 'csse_covid_19_data\csse_covid_19_time_series'
    # us_or_global = 'US'
    # case_types = ['deaths','confirmed','recovered']

    path = input_directory_corona_cases

    try:
        content = us_or_global.lower()
        if ('global' and 'us') not in content:
            raise ValueError("error, you must use 'global' or 'us' as an input")
    except (ValueError, Exception) as e:
        print(e)
    else:
        if us_or_global.lower() == 'global':
            us_or_global = us_or_global.lower()
            case_types = case_types
            pass
        elif us_or_global.lower() == 'us':
            us_or_global = us_or_global.upper()
            case_types = ['deaths'] + list(set(case_types) - set(['deaths']))
            case_types.remove('recovered')
            pass                    

    print('pulling data for:', us_or_global)
    for i in range(len(case_types)):

        #read in the data from the forked directory
        file_name = 'time_series_covid19_' + case_types[i] + '_' + us_or_global + '.csv'
        temp_df = pd.read_csv(input_directory_corona_cases + '\\' + file_name)

        #Grab and merge the populations from the 'deaths' file, use it to 
        # populate populations in other files
        if case_types[i] == 'deaths' and us_or_global.lower() == 'us':
            temp_population_lookup = temp_df[[
                'Combined_Key', 'Population']].drop_duplicates()
        elif us_or_global.lower() == 'us':
            temp_df = temp_df.merge(
                temp_population_lookup,
                how='left',
                on='Combined_Key',
                validate="many_to_one")

        #identify non date columns by iterating through each
        temp_column_names = list(temp_df.columns)
        temp_column_names_is_not_date_df = pd.DataFrame(
            {'columnName': [None], 'columnIsDate': [None]})

        for j in range(len(temp_column_names)):
            isValidDate = None
            temp_column_names_is_not_date_df.at[j, 'columnName'] = temp_column_names[j]

            try:
                if us_or_global.lower() == 'us':
                    result = datetime.datetime.strptime( temp_column_names[j], '%m/%d/%y')
                elif us_or_global.lower() == 'global':
                    result = datetime.datetime.strptime( temp_column_names[j], '%m/%d/%y')
            except Exception as e:
                temp_column_names_is_not_date_df.at[j, 'columnIsDate'] = False
            else:                
                temp_column_names_is_not_date_df.at[j, 'columnIsDate'] = True

        #Filter list of columns to non-date column names to assist with melt
        temp_column_names_is_not_date_list = temp_column_names_is_not_date_df['columnName'][
            temp_column_names_is_not_date_df['columnIsDate'] == False].to_list()

        #melt based on non-date columns and rename the 'value' column, fill NAs 
        # for dimensions, and update date column to datetime
        temp_df_melted = temp_df.melt(
            id_vars=temp_column_names_is_not_date_list).fillna('None')
        temp_df_melted.rename(
            columns={
                'value': case_types[i],
                'variable': 'date'},
            inplace=True)

        #update data type to datetime for date columns
        temp_df_melted['date'] = pd.to_datetime(temp_df_melted['date'])

        #add case types to list of non-date columns
        temp_column_names_is_not_date_list.append(case_types[i])

        #rename case type column from value to [case_type] and then check for
        # missing columns
        temp_missing_columns = list(
            set(case_types) - set(temp_column_names_is_not_date_list))

        #Fill NAs for metrics. can't make reindex work, so using indexing and
        # fillna() to attempt to fill NA's for metrics with the input value
        temp_df_melted = temp_df_melted.reindex(
            columns=temp_df_melted.columns.tolist() + temp_missing_columns,
            fill_value=0)

        #union dataframes
        if i == 0:
            temp_dfs_melted_unioned = temp_df_melted
        else:
            temp_dfs_melted_unioned = pd.concat(
                [temp_dfs_melted_unioned, temp_df_melted], ignore_index=True)

    #identify the aggregation columns
    temp_dfs_melted_uniioned_aggregation_columns = list(set(
        temp_column_names_is_not_date_list) - set(case_types)) + ['date']

    #aggregate case types based on the aggregation columns idetnifided
    #TODO -- clean up the 'global' vs. 'us' files prior to transforming or 
    # automate the building of the agg() statement based on the set() of case 
    # types available
    if us_or_global.lower() == 'global':
        temp_dfs_melted_unioned_aggregated = temp_dfs_melted_unioned.groupby(
            temp_dfs_melted_uniioned_aggregation_columns
        ).agg(recovered=('recovered', 'sum'),
                confirmed=('confirmed', 'sum'),
                deaths=('deaths', 'sum')
                ).reset_index()
    if us_or_global.lower() == 'us':
        temp_dfs_melted_unioned_aggregated = temp_dfs_melted_unioned.groupby(
            temp_dfs_melted_uniioned_aggregation_columns
        ).agg(confirmed=('confirmed', 'sum'),
              deaths=('deaths', 'sum')
              ).reset_index()

    #prior to adding a prefix and generating a lat_long_id, fix column names
    # based on known discrepancies
    column_rename = {
        'Long_': 'Long',
        'Country/Region': 'Country_Region',
        'Province/State': 'Province_State'
        }
    temp_dfs_melted_unioned_aggregated.rename(
        columns=column_rename,
        inplace=True)

    #add a unique id for each unique combination of latitude/longitude
    temp_dfs_melted_unioned_aggregated['lat_long_id'] = temp_dfs_melted_unioned_aggregated.groupby(
        ['Lat', 'Long']).ngroup()

    #add cases_ prefix to make it esaier to determine where each variable comes from
    temp_dfs_melted_unioned_aggregated = temp_dfs_melted_unioned_aggregated.add_prefix(
        'cases_')

    return temp_dfs_melted_unioned_aggregated


#%% Add a flag for when total confirmed cases reached N for each country and
# then Assign a sequential number to each date based on the
#Name of output file
def add_flag_for_n_cases_date(
        cases_dataframe, 
        n_list, 
        case_types=['deaths', 'confirmed', 'recovered']):

    for i in range(len(n_list)):

        while True:
            try:
                if isinstance(n_list[i], int):
                    i_int = n_list[i]
                    i_str = str(n_list[i])
                    break
                else:
                    raise TypeError
                break
            except TypeError:
                print("must be a list of integers")
                break

        #apply a true/false flag based on whether n cases have been reached
        column_n_or_more_confirmed_cases_by_country = i_str + \
            '_or_more_confirmed_cases_by_country'
        cases_dataframe[column_n_or_more_confirmed_cases_by_country] = np.where(
            cases_dataframe.cases_confirmed >= i_int,
            True, False)

        #Apply a count for dates partitioned based on once n cases have been 
        # reached for every combination of dimensions apart from date and
        # case_types
        temp_groupby_columns_for_counts_lat_long = list(
            set(cases_dataframe.columns)-set(case_types)-set(['cases_date']))

        temp_groupby_columns_for_counts_lat_long = temp_groupby_columns_for_counts_lat_long + [column_n_or_more_confirmed_cases_by_country]

        column_n_or_more_confirmed_cases_by_country_day_count = i_str + \
            '_or_more_confirmed_cases_by_country_day_count'
            
        cases_dataframe[column_n_or_more_confirmed_cases_by_country_day_count] = cases_dataframe.groupby(
            temp_groupby_columns_for_counts_lat_long)['cases_date'].rank(method='dense')

        #if n cases have not been reached, remove any counts created from the previous statement
        cases_dataframe.loc[getattr(cases_dataframe, column_n_or_more_confirmed_cases_by_country) == False,
                            column_n_or_more_confirmed_cases_by_country_day_count] = None

    return cases_dataframe
