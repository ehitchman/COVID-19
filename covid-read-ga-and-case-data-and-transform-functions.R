
##################################################
##################################################
#Basic GA Reuqest
funcGA_listOfAccounts <- function(ga_account_list, 
                                  max_rows = 100,
                                  date_range = c(Sys.Date()-31, Sys.Date()-1),
                                  dimensions = c("ga:date",
                                                 "ga:channelGrouping",
                                                 "ga:country",
                                                 #"ga:region",
                                                 "ga:deviceCategory"),
                                  metrics = c("ga:pageviews",
                                              "ga:sessions",
                                              "ga:timeOnPage",
                                              "ga:bounces"
                                  )
) {
  
  # This uses GoogleAnalyticsR to pull the CM360 model of Sessions, Clicks, Cost, 
  #  and Impressions by default
  #
  # Args:
  #  @param view_id_x: Google Analytics viewId
  #  @param date_range: Start and end date in a character vector of format
  #    c(startDate, endDate), where each are of format "yyyy-mm-dd"
  #  @param dimensions: Default dimensions for the analytics api request.  Includes
  #    floodlight configuration id, advertiser id and campaign id
  #  @param metrics: Default metrics for the analytics api request.  Includes clicks,
  #    impressions, cost, sessions
  #  @examples: 
  #     #not run 
  #     funcCM360SessionsClicksCost(1234567890)
  #  
  # Returns:
  #  @returns: dataframe of the google analytics response from viewId input
  #
  # Examples:
  #     #not run 
  #     funcCM360SessionsClicksCost(1234567890)
  #
  # Test:
  # ga_views_meta <- ga_account_list()
  # ga_views_meta_filtered = ga_views_meta[which(ga_views_meta['level']=='PREMIUM'),]
  # ga_views_meta_filtered_starred = ga_views_meta_filtered[which(ga_views_meta_filtered['starred']==TRUE),]
  # ga_views_meta_filtered_starred_subset = ga_views_meta_filtered_starred[c(1,3),]
  # ga_account_list = ga_views_meta_filtered_starred_subset
  # max_rows = 100
  # date_range = c(Sys.Date()-31, Sys.Date()-1)
  # dimensions = c("ga:date",
  #                "ga:channelGrouping",
  #                "ga:country",
  #                "ga:region",
  #                "ga:deviceCategory")
  # metrics = c("ga:pageviews",
  #             "ga:sessions",
  #             "ga:timeOnPage",
  #             "ga:bounces"
  # )
  #
  # Require:
  require(dplyr)
  require(googleAnalyticsR)
  
  print("Basic GA Request")
  
  for (i in seq_along(1:nrow(ga_account_list))) {
    
    viewId_column = 'viewId'
    
    print("viewid number:")
    print(i)
    print('out of total number of views:')
    print(nrow(ga_account_list))
    print('-----')
    
    ga_viewId = ga_account_list[i,'viewId']
    print("ga_viewId")    
    print(ga_viewId)
    
    tempDF <- google_analytics(viewId = ga_viewId, 
                               date_range = date_range,
                               dimensions = dimensions, 
                               metrics = metrics,
                               anti_sample = TRUE,
                               max = max_rows)
    
    tempDF <- merge(tempDF, 
                    ga_account_list,
                    on.x = viewId_column)
    
    if (i == 1) {
      finalDF = tempDF
      
    } else {
      finalDF = rbind(finalDF, tempDF)
    }
  }
  
  return(finalDF)
  
}



read_covid_file_and_transform = function(filename = 'C:/Users/erich/OneDrive/Desktop/OneDrive Documents/repos/COVID-19_CSSEGISandData/output/corona_cases_daily_with_populations_and_flags.csv') {
  library(dplyr)
  tempdf = read.csv(filename)
  
  final_df = tempdf %>%
    group_by(cases_date, cases_Country.Region) %>%
    summarize(country_populations_pop2020 = min(country_populations_pop2020),
              recovered = sum(recovered),
              confirmed = sum(confirmed),
              deaths = sum(deaths)) %>%
    select(cases_date,
           cases_Country.Region,
           country_populations_pop2020,
           recovered,
           confirmed,
           deaths) %>%
    arrange(cases_Country.Region, cases_date)
  
  return(final_df)
}


read_account_list_lookup_file_and_transform = function(filename = 'C:/Users/erich/OneDrive/Desktop/OneDrive Documents/repos/COVID-19-GA-Integration/ga_account_list_for_lookup.xlsx',
                                                       sheetname = 'ga_account_list_for_lookup') {
  library(dplyr)
  tempdf = read_excel(path = filename,
                      sheet = sheetname)
  
  return(tempdf)
}