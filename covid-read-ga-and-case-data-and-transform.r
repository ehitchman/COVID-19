
##############
#Load packages
require(rjson)
require(googleAuthR)
require(googleAnalyticsR)
require(tidyverse)
require(data.table)
library(readxl)

#########################################################
#set some parameters for folder names, view ids, QC, etc.
run_script_printouts_and_write_qc_files = FALSE
gaStartDate <- '2020-01-01'
gaEndDate <- as.character(Sys.Date()-1)
folderNameMetricsDimensions <- "metrics-and-dimensions"
output_qc_directory <- 'output'
forderNameLookupTables <- 'lookup_tables'

###################
##Read in functions
source(paste0(getwd(),"/","covid-read-ga-and-case-data-and-transform-functions.r"))

######################################
#Authorize using the existing json file
ga_auth(email = Sys.getenv('GA_AUTH_EMAIL'))

############################
#get details about all views
ga_views_meta <- ga_account_list()
ga_views_meta_filtered = ga_views_meta[which(ga_views_meta['level']=='PREMIUM'),]
ga_views_meta_filtered_starred = ga_views_meta_filtered[which(ga_views_meta_filtered['starred']==TRUE),]
ga_views_meta_filtered_starred_subset = ga_views_meta_filtered_starred[c(1:30),]

###########################################
#Get the GA Account and ViewId Lookup Table
ga_account_list_lookup_filename =  'ga_account_list_for_lookup.xlsx'
ga_account_list_lookup = read_account_list_lookup_file_and_transform(filename = paste0(getwd(),"/", 
                                                                                       forderNameLookupTables,"/",
                                                                                       ga_account_list_lookup_filename),
                                                                     sheetname = 'ga_account_list_for_lookup')

################
#get the GA data
df_unionsed = funcGA_listOfAccounts(ga_account_list = ga_views_meta_filtered_starred_subset,
                                    max_rows = -1,
                                    date_range = c(gaStartDate, gaEndDate),
                                    dimensions = c("ga:date",
                                                   "ga:channelGrouping",
                                                   "ga:country",
                                                   #"ga:region",
                                                   "ga:deviceCategory"),
                                    metrics = c("ga:pageviews",
                                                "ga:sessions",
                                                "ga:timeOnPage",
                                                "ga:bounces"))

####################
#Get the Corona Data
covid = read_covid_file_and_transform()

#####################
#QA - unique countries
if (run_script_printouts_and_write_qc_files == TRUE) {
  qc_unique_countries_covid = "unique_countries_covid.csv"
  unique_countries_covid = unique(covid[c('cases_Country.Region', 'country_populations_pop2020')])
  rownames(unique_countries_covid) <- 1:nrow(unique_countries_covid)
  write.csv(unique_countries_covid, 
            file = qc_unique_countries_covid, 
            row.names = FALSE)
  
  qc_unique_countries_ga = "unique_countries_ga.csv"
  unique_countries_ga = df_unionsed %>%
    group_by(country) %>%
    summarize('sum of pageviews' = sum(pageviews)) %>%
    select('country', 'sum of pageviews')
  rownames(unique_countries_ga) <- 1:nrow(unique_countries_ga)
  write.csv(unique_countries_ga, 
            file = qc_unique_countries_ga, 
            row.names = FALSE)
  
  qc_unique_countries_merged_covid_left = "unique_countries_merged_covid_left.csv"
  unique_countries_merged = merge(unique_countries_covid, unique_countries_ga, 
                                  by.x='cases_Country.Region', by.y='country',
                                  all.x = TRUE)
  rownames(unique_countries_merged) <- 1:nrow(unique_countries_merged)
  write.csv(unique_countries_merged, 
            file = qc_unique_countries_merged_covid_left, 
            row.names = FALSE)
  
  qc_unique_countries_merged_ga_left = "unique_countries_merged_ga_left.csv"
  unique_countries_merged = merge(unique_countries_ga, unique_countries_covid, 
                                  by.x='country', by.y='cases_Country.Region',
                                  all.x = TRUE)
  rownames(unique_countries_merged) <- 1:nrow(unique_countries_merged)
  write.csv(unique_countries_merged, 
            file = qc_unique_countries_merged_ga_left, 
            row.names = FALSE)  


  
  ###########################################
  #merge and then write Write final data file
  qc_ga_and_covid = 'output_ga_and_covid.csv'
  final_merge = merge(df_unionsed, covid, 
                      by.x = c('date', 'country'),
                      by.y = c('cases_date', 'cases_Country.Region'))
  write.csv(final_merge, file = qc_ga_and_covid, row.names = FALSE)
}

#####################################################
#merge final data file with the custom GA lookup file
output_final_merge_with_lookup = 'output_ga_and_covid_and_ga_account_list_lookup.csv'
final_merge_with_lookup = merge(final_merge, 
                                ga_account_list_lookup, 
                                on = c(viewId))

#############
#Write to CSV 
write.csv(final_merge_with_lookup, 
          file = paste0(output_qc_directory, '/', output_final_merge_with_lookup), 
          row.names = FALSE)


