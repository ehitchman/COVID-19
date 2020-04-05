
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


def update_country_name_in_population_lookup(
        population_lookup_df,
        original_to_new_country_name_and_code_df,
        run_script_printouts_and_write_qc_files=True):

    if run_script_printouts_and_write_qc_files == True:
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

    #Here is a statement similar to coalesce which prioritizes the lookup value,
    # and if no lookup value is found, the original value is used in its place
    final_df['country_populations_countryName_final'] = np.where(
        final_df["upd_countryName"].isnull(), final_df["country_populations_Country"], final_df["upd_countryName"])
    final_df['country_populations_countryCode_final'] = np.where(
        final_df["upd_countryName"].isnull(), final_df["country_populations_cca2"], final_df["upd_countryCode"])

    #Aggregate adjusted population table to ensure there are no duplicates
    final_df_aggregated = final_df.groupby([
        'country_populations_countryCode_final',
        'country_populations_countryName_final'
    ]).agg(country_populations_pop2020=('country_populations_pop2020', 'sum'),
           country_populations_area=('country_populations_area', 'sum')
           ).reset_index()

    #calculate density
    final_df_aggregated['country_populations_Density'] = final_df_aggregated['country_populations_pop2020'] / \
        final_df_aggregated['country_populations_area']

    #Calculate world percentage
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

    print('directory being checked:', path)
    filename = filename.replace('.csv.csv', '').replace('.csv', '') + '.csv'
    print('filename being checked:', filename)

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
        cases_column_order,
        us_or_global,
        case_types = ['deaths','confirmed','recovered']):

    path = input_directory_corona_cases

    while True:
        try:
            content = us_or_global.lower()
            if content.lower() == 'global' or content.lower() == 'us':
                if us_or_global.lower() == 'global':
                    us_or_global = 'global'
                    case_types = case_types
                elif us_or_global.lower() == 'us':
                    us_or_global = 'US'
                    case_types = case_types.remove('recovered')
                    case_types = list(['deaths']) + list(
                        set(case_types) - set(['deaths']))
                break
            else:
                raise TypeError
        except TypeError:
            print("error, you must use 'global' or 'us' as an input")
            break

    print('pulling data for:', us_or_global)

    for i in range(len(case_types)):

        print("this is the case_type for this iteration:", case_types[i])

        #read in the data from the forked directory
        file_name = 'time_series_covid19_' + \
            case_types[i] + '_' + us_or_global + '.csv'
        temp_df = pd.read_csv(input_directory_corona_cases + '\\' + file_name)

        #Grab the populations from the deaths file, use it to populate
        # populations in other files
        if case_types[i] == 'deaths' and us_or_global.lower() == 'us':
            temp_population_lookup = temp_df[[
                'Combined_Key', 'Population']].drop_duplicates()

        elif us_or_global.lower() == 'us':

            #Merge the population from the 'deaths' file with every other file
            temp_df = temp_df.merge(
                temp_population_lookup,
                how='left',
                on='Combined_Key',
                validate="many_to_one")

        #identify non date columns and iterate through each to identify non-date
        # columns
        temp_column_names = list(temp_df.columns)
        temp_column_names_is_not_date_df = pd.DataFrame(
            {'columnName': [None], 'columnIsDate': [None]})
        for j in range(len(temp_column_names)):

            print("this is the column for this iteration", j)

            isValidDate = None
            temp_column_names_is_not_date_df.at[j,
                                                'columnName'] = temp_column_names[j]
            try:
                if us_or_global.lower() == 'us':
                    datetime.datetime.strptime(
                        temp_column_names[j], '%m/%d/%Y')
                elif us_or_global.lower() == 'global':
                    datetime.datetime.strptime(
                        temp_column_names[j], '%m/%d/%y')
            except (TypeError, ValueError):
                isValidDate = False
            if isValidDate == False:
                temp_column_names_is_not_date_df.at[j, 'columnIsDate'] = False
            else:
                temp_column_names_is_not_date_df.at[j, 'columnIsDate'] = True

        #Filter to non-date column names
        temp_column_names_is_not_date_list = temp_column_names_is_not_date_df['columnName'][
            temp_column_names_is_not_date_df['columnIsDate'] == False].to_list()
        print('print temp_column_names_is_not_date_list before append')
        print(temp_column_names_is_not_date_list)
        print('----------------------------------------------')

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
        #print(unique(temp_df_melted['date']))

        #add case types to list of non-date columns
        temp_column_names_is_not_date_list.append(case_types[i])



        print('print temp_column_names_is_not_date_list after append')
        print(temp_column_names_is_not_date_list)
        print('----------------------------------------------')

        print('print all column names')
        print(temp_df_melted.dtypes)
        print('----------------------------------------------')

        #rename case type column from value to [case_type]and then check for
        # missing columns
        temp_missing_columns = list(
            set(case_types) - set(temp_column_names_is_not_date_list))

        #Fill NAs for metrics. can't make .reindex work, so using indexing and
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

    #identify the aggregation columns & then aggregate case types
    temp_dfs_melted_uniioned_aggregation_columns = list(set(
        temp_column_names_is_not_date_list) - set(case_types)) + ['date']

    print("these are the dimensions we will aggregate for")
    print(temp_dfs_melted_uniioned_aggregation_columns)

    temp_dfs_melted_unioned_aggregated = temp_dfs_melted_unioned.groupby(
        temp_dfs_melted_uniioned_aggregation_columns
    ).agg(recovered=('recovered', 'sum'),
            confirmed=('confirmed', 'sum'),
            deaths=('deaths', 'sum')
            ).reset_index()

    #add a unique id for each unique combination of latitude/longitude
    temp_dfs_melted_unioned_aggregated['lat_long_id'] = temp_dfs_melted_unioned_aggregated.groupby(
        ['Lat', 'Long']).ngroup()

    #add cases_ prefix to make it esaier to determine where each variable comes from
    temp_dfs_melted_unioned_aggregated = temp_dfs_melted_unioned_aggregated.add_prefix(
        'cases_')
    print('end of function... added prefix... print column names')
    print(temp_dfs_melted_unioned_aggregated.dtypes)

    return temp_dfs_melted_unioned_aggregated


#%% Add a flag for when total confirmed cases reached N for each country and
# then Assign a sequential number to each date based on the
#Name of output file
def add_flag_for_n_cases_date(
        cases_dataframe, 
        n_list, 
        case_types=['deaths', 'confirmed', 'recovered']):

    for i in range(len(n_list)):
        print('add flag for n cases:', n_list[i], ': these are the columns in the dataframe')
        print(cases_dataframe.dtypes)

        while True:
            try:
                if isinstance(n_list[i], int):
                    i_int = n_list[i]
                    print('this is i_int for the current iteration', i_int)
                    i_str = str(n_list[i])
                    print('this is i_str for the current iteration', i_str)
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
        print('this is column_n_or_more_confirmed_cases_by_country_day_count:', i_str)
        print('---------------------------------------------------')

        #Apply a count for dates partitioned based on once n cases have been 
        # reached for every combination of dimensions apart from date and
        # case_types
        temp_groupby_columns_for_counts_lat_long = list(
            set(cases_dataframe.columns)-set(case_types)-set(['cases_date']))
        print('temp_groupby_columns_for_counts_lat_long after removing case_types and dates')
        print(temp_groupby_columns_for_counts_lat_long)
        print('---------------------------------------------------')

        print([column_n_or_more_confirmed_cases_by_country])
        temp_groupby_columns_for_counts_lat_long = temp_groupby_columns_for_counts_lat_long + [column_n_or_more_confirmed_cases_by_country]
        print('temp_groupby_columns_for_counts_lat_long after appending the n_list confirmed_cases_by_country column')
        print(temp_groupby_columns_for_counts_lat_long)
        print('---------------------------------------------------')

        print('these will be the group by columns at the finest detail')
        print(temp_groupby_columns_for_counts_lat_long)
        print('---------------------------------------------------')

        column_n_or_more_confirmed_cases_by_country_day_count = i_str + \
            '_or_more_confirmed_cases_by_country_day_count'
            
        cases_dataframe[column_n_or_more_confirmed_cases_by_country_day_count] = cases_dataframe.groupby(
            temp_groupby_columns_for_counts_lat_long)['cases_date'].rank(method='dense')
        print('this is column_n_or_more_confirmed_cases_by_country_day_count:', i_str)
        print('---------------------------------------------------')

        #if n cases have not been reached, remove any counts created from the previous statement
        cases_dataframe.loc[getattr(cases_dataframe, column_n_or_more_confirmed_cases_by_country) == False,
                            column_n_or_more_confirmed_cases_by_country_day_count] = None

    return cases_dataframe
