#------------Load Libraries
library(dplyr)
library(ggplot2)
library(rjson)
library(jsonlite)
library(purrr)
library(rpart)
library(ggmap)
library(party)
library(randomForest)
library(xgboost)

setwd("F:\\Sid\\Learnings\\Data Scientist\\Machine Learning With Kaggle\\Two Sigma Connect Rental Listing Inquiries")

loadData <- function(path){
     data <- fromJSON(path)
     vars <- setdiff(names(data), c("photos", "features"))
     data <- map_at(data, vars, unlist) %>% tibble::as_tibble(.)
     
     #---------Convert data to proper types
     if(contains(colnames(data),"interest_level")){
          data$interest_level <- as.factor(data$interest_level)
     }
     #-----Convert managerid to int
     mgrIds <-seq_len(length(unique(data$manager_id)))
     managers <- data.frame(MgrId = mgrIds,manager_id = unique(data$manager_id))
     
     data <- merge(data,managers,by = "manager_id")
     
     #data$MgrId <- as.integer(data$MgrId)
     data$created <- as.POSIXct(data$created)
     
     return(data)
}
train_Data <- loadData("train.json")
test_Data <- loadData("test.json")

featureDictionary <- data.frame(Feature = tolower(unlist(train_Data$features))) %>%
     group_by(Feature) %>% summarise(Total = n()) %>%
     arrange(desc(Total)) %>% filter(Total >= 50)

#-----------Get LAtitude and longitude based on address, as there are some address with 
#-----------0 as latitude and longitude

getGeoDetails <- function(address){   
    
     #use the gecode function to query google servers
     geo_reply = geocode(address, output='all', messaging=TRUE, override_limit=TRUE)
     
     #now extract the bits that we need from the returned list
     answer <- data.frame(lat=NA, long=NA, accuracy=NA, formatted_address=NA, address_type=NA, status=NA)
     answer$status <- geo_reply$status
     
     #if we are over the query limit - want to pause for an hour
     while(geo_reply$status == "OVER_QUERY_LIMIT"){
          print("OVER QUERY LIMIT - Pausing for 1 hour at:") 
          time <- Sys.time()
          print(as.character(time))
          Sys.sleep(60*60)
          geo_reply = geocode(address, output='all', messaging=TRUE, override_limit=TRUE)
          answer$status <- geo_reply$status
     }
     
     #return Na's if we didn't get a match:
     if (geo_reply$status != "OK"){
          return(answer)
     }   
     #else, extract what we need from the Google server reply into a dataframe:
     answer$lat <- geo_reply$results[[1]]$geometry$location$lat
     answer$long <- geo_reply$results[[1]]$geometry$location$lng   
     if (length(geo_reply$results[[1]]$types) > 0){
          answer$accuracy <- geo_reply$results[[1]]$types[[1]]
     }
     answer$address_type <- paste(geo_reply$results[[1]]$types, collapse=',')
     answer$formatted_address <- geo_reply$results[[1]]$formatted_address
     
     return(answer)
}


#-------Explore data set
#train_Data %>% group_by(latitude,longitude) %>% summarise(Total = n()) %>% filter(Total > 1)

#---------Generate full address to get lat and long
train_Data$FullAddress <- paste(train_Data$street_address,"USA",sep = ",")
test_Data$FullAddress <- paste(test_Data$street_address,"USA",sep = ",")

test_Data %>% filter(latitude < 1) %>% select(c(listing_id,latitude,longitude,street_address,display_address,FullAddress)) %>%
     head(100) %>% write.csv("Addresses_test.csv")

train_Data %>% filter(latitude < 1) %>% select(c(listing_id,latitude,longitude,street_address,display_address,FullAddress)) %>%
     head(100) %>% write.csv("Addresses_train.csv")

getLatLong <- function(data){
     addresses <- data #train_Data[train_Data$longitude == 0.0000,c("listing_id","FullAddress")]
     geocoded <- data.frame()
     # Start the geocoding process - address by address. geocode() function takes care of query speed limit.
     for (ii in seq(1, length(addresses$FullAddress))){
          #query the google geocoder - this will pause here if we are over the limit.
               result = getGeoDetails(addresses$FullAddress[ii])
               #print(result$status)     
               result$index <- ii
               #append the answer to the results file.
               geocoded <- rbind(geocoded, result)
               #save temporary results as we are going along
               #saveRDS(geocoded, tempfilename)

     }
     addresses$Latitude = geocoded$lat
     addresses$Longitude = geocoded$long
     
     #-------Remove those addresses where there is no lat and long
     addresses <- addresses[!is.na(addresses$Latitude),]
     return(addresses)
}

#----------Update train data's latitude and longitude
addresses <- getLatLong(train_Data[train_Data$latitude < 1,c("listing_id","FullAddress")])
#-------Update lat and long in train_data
train_Data[train_Data$listing_id %in% addresses$listing_id,"latitude"] <- addresses$Latitude
train_Data[train_Data$listing_id %in% addresses$listing_id,"longitude"] <- addresses$Longitude
#-----------Remove rows where there is no lat and long
train_Data <- train_Data[train_Data$longitude != 0.0000,]

#----------Update test data's latitude and longitude
addresses <- getLatLong(test_Data[test_Data$latitude < 1,c("listing_id","FullAddress")])
#-------Update lat and long in train_data
test_Data[test_Data$listing_id %in% addresses$listing_id,"latitude"] <- addresses$Latitude
test_Data[test_Data$listing_id %in% addresses$listing_id,"longitude"] <- addresses$Longitude
#-----------Remove rows where there is no lat and long
test_Data <- test_Data[test_Data$longitude != 0.0000,]

#---------Check plots
#-----------Plots
#------------There are some 0 latitude and longitude in test and train data. Update those
ggplot(test_Data,aes(x=test_Data$latitude,y=test_Data$longitude)) +
     geom_point() + 
     geom_text(aes(label=paste(test_Data$latitude,y=test_Data$longitude,sep = ",")),hjust=0, vjust=0)


#----------Check managers
temp <- as.data.frame(table(train_Data$manager_id))
temp <- temp[with(temp,order(-Freq)),]

#---------Decision tree
model_DT <- ctree(formula = interest_level ~ bathrooms 
                  + bedrooms + latitude + longitude 
                  + price + MgrId
                  ,data = train_Data)

train_Data$PredictedValue <- predict(model_DT,train_Data)
resultSet <- train_Data[,"PredictedValue"] == train_Data[,"interest_level"]

table(resultSet)

model_RF <- randomForest(formula = interest_level ~ bathrooms 
                  + bedrooms + latitude + longitude 
                  + price
                  ,data = train_Data
                  ,ntree = 500
                  ,mtry = 3)

#--------Check random forest model on train data
train_Data$PredictedValue_RF <- predict(model_RF,train_Data)
resultSet <- train_Data[,"PredictedValue_RF"] == train_Data[,"interest_level"]

#--------Check model on test data
p1 <- predict(model_RF,test_Data,type = "prob")
resultset <- cbind(test_Data$listing_id,p1)

write.csv(resultset,"Result.csv")

#-----------xgboost
input_dataset <- train_Data %>% mutate(Cost = (bathrooms + bedrooms) * price) %>% 
                    select(bathrooms,bedrooms,latitude,longitude,price,Cost,created,interest_level) 

test_dataset <- test_Data %>% mutate(Cost = (bathrooms + bedrooms) * price) %>% 
                    select(bathrooms,bedrooms,latitude,longitude,price,Cost,created) 

#------high: 1, low: 2, medium: 3
#------we need to consider the labels from 0 instead of 1
numeric_labels <- unclass(input_dataset[,"interest_level"]) %>% as.numeric() - 1

params <- list(nthread = 2, num_class = "3",
               objective = "multi:softprob")

model_xgb <- xgboost(data = data.matrix(subset(input_dataset,select = -c(interest_level)))
                         ,label = numeric_labels
                         ,nrounds = 300
                         ,early_stopping_rounds = 3
                         ,params = params)

result_xgb <- predict(model_xgb,data.matrix(subset(input_dataset,select = -c(interest_level))))
#result_xgb <- predict(model_xgb,data.matrix(test_dataset))
#--------reshape data
result_xgb <- matrix(result_xgb,ncol = 3,byrow = TRUE)
colnames(result_xgb) <- c("high","low","medium")

write.csv(data.frame(input_dataset$listing_id,result_xgb),"Result.csv")

importance_matrix <- xgb.importance(names(train_Data),model = model_xgb)


