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
library(e1071) #----------Naive Bayes

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

featuresList <- union(train_Data$features,test_Data$features)
featureDictionary <- data.frame(Feature = tolower(unlist(featuresList))) %>%
     group_by(Feature) %>% summarise(Total = n()) %>%
     arrange(desc(Total)) %>% filter(Total >= 50)

#-------------Read Features manipulated data
featureDictionary <- read.csv("Feature.csv",header = TRUE,stringsAsFactors = FALSE)
updatedFeatureDictionary <- unique(featureDictionary$UpdatedFeatureName)

#---------Modify features in train and test data
createFeatures <- function(data){
     #---------Add columns to data frame
     for(featureName in updatedFeatureDictionary){
          data[,featureName] <- factor(rep(0,nrow(data)),levels = c(0,1))
     }
     #--------Modify value of features in each data set
     for(index in 1:nrow(data)){
          rowData <- data[index,updatedFeatureDictionary]
          rowWiseFeatures <- as.character(tolower(unlist(data[index,"features"])))
          matchingList <- unique(featureDictionary[featureDictionary$Feature %in% rowWiseFeatures,4])
          data[index,matchingList] <- 1
          #print(rowData[,matchingList])
     }
     return(data)
}

train_Data <- createFeatures(train_Data)
test_Data <- createFeatures(test_Data)


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
train_Data$FullAddress <- paste(train_Data$street_address,"USA",sep = ",")#,train_Data$display_address,
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
train_Data <- train_Data[train_Data$latitude != 0.0000,]

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


#-----------xgboost
featureColumnsList <- union(updatedFeatureDictionary
                         ,c("listing_id","bathrooms","bedrooms","latitude","longitude","price","ModPrice","Cost","created","interest_level")
                         )

input_dataset <- train_Data %>% mutate(Cost = (bathrooms + bedrooms) / price, ModPrice = log(price)) 
input_dataset <- input_dataset %>% select(which(colnames(input_dataset) %in% featureColumnsList)) 

#---------Remove outliers
input_dataset <- input_dataset %>% filter(price <= 90000)

test_dataset <- test_Data %>% mutate(Cost = (bathrooms + bedrooms) / price, ModPrice = log(price))
test_dataset <- test_dataset %>% select(which(colnames(test_dataset) %in% featureColumnsList)) 

#------high: 1, low: 2, medium: 3
#------we need to consider the labels from 0 instead of 1
numeric_labels <- unclass(input_dataset[,"interest_level"]) %>% as.numeric() - 1

params <- list(nthread = 2, num_class = "3",
               objective = "multi:softprob")

getModel <- function(data,totalRounds)
{
     bestAccuracy <- 0
     bestModel <- NULL
     for(index in 1:20)
     {
          set.seed(index)
          totalRows <- nrow(data)
          sampleRowsRange <- sample.int(totalRows,40000)
          
          sampleDF <- data[sampleRowsRange,]
          crossValidateDF <- data[setdiff(1:totalRows,sampleRowsRange),]

          numeric_labels <- unclass(sampleDF[,"interest_level"]) %>% as.numeric() - 1 
          
          model_xgb <- xgboost(data = data.matrix(subset(sampleDF,select = -c(interest_level,listing_id)))
                         ,label = numeric_labels
                         ,nrounds = totalRounds
                         ,early_stopping_rounds = 3
                         ,params = params)
          
          result_xgb <- predict(model_xgb,data.matrix(subset(crossValidateDF,select = -c(interest_level,listing_id))))
          
          #--------reshape data
          result_xgb <- data.frame(crossValidateDF$listing_id, matrix(result_xgb,ncol = 3,byrow = TRUE))
          colnames(result_xgb) <- c("listing_id","high","low","medium")
          result_xgb$predictedVal <- ifelse(result_xgb$low > result_xgb$high,"low",ifelse(result_xgb$high>result_xgb$medium,"high","medium"))
          
          accuracyTable <- table(result_xgb$predictedVal, crossValidateDF$interest_level)
          accuracy = (accuracyTable[1,1] + accuracyTable[2,2] + accuracyTable[3,3]) / nrow(crossValidateDF)
          
          if(accuracy > bestAccuracy){
               print(paste("index: ",index))
               print(paste("Accuracy:" ,accuracy))
               bestModel <- model_xgb
               bestAccuracy <- accuracy
          }
          
     }
     return(bestModel)
}

model_xgb <- xgboost(data = data.matrix(subset(input_dataset,select = -c(interest_level,listing_id)))
                     ,label = numeric_labels
                     ,nrounds = 300
                     ,early_stopping_rounds = 3
                     ,params = params)

#---------model_xgb <- getModel(input_dataset,300)

#-----Refer best model and predict result on whole data set
result_xgb <- predict(model_xgb,data.matrix(subset(input_dataset,select = -c(interest_level,listing_id))))
#result_xgb <- predict(model_xgb,data.matrix(subset(test_dataset,select = -c(listing_id))))

#--------reshape data
result_xgb <- data.frame(input_dataset$listing_id, matrix(result_xgb,ncol = 3,byrow = TRUE))
#result_xgb <- data.frame(test_dataset$listing_id, matrix(result_xgb,ncol = 3,byrow = TRUE))
colnames(result_xgb) <- c("listing_id","high","low","medium")
result_xgb$predictedVal <- ifelse(result_xgb$low > result_xgb$high,"low",ifelse(result_xgb$high>result_xgb$medium,"high","medium"))

table(result_xgb$predictedVal, input_dataset$interest_level)

write.csv(data.frame(result_xgb),"Result.csv")

xgb.importance(names(subset(input_dataset,select = -c(interest_level,listing_id))),model = model_xgb)

#-----------------plots

ggplot(input_dataset,aes(x=input_dataset$price,y=input_dataset$bedrooms))+
     geom_point(aes(color = input_dataset$interest_level))

ggplot(input_dataset[input_dataset$interest_level == "medium",],aes(x=interest_level,y=price))+
     geom_boxplot()


ggplot(input_dataset,aes(x=interest_level,y=price))+
     geom_boxplot()

#-------------Density plot for price with interest level as low
ggplot(input_dataset[input_dataset$interest_level == "medium",],aes(price))+
     geom_density()

#-------------Density plot for price     
ggplot(input_dataset,aes(input_dataset$price,color = interest_level))+
     geom_density()

input_dataset %>% group_by(interest_level) %>% summarise(n=n()
                                                         ,minPrice = min(price)
                                                         ,maxPrice=max(price)
                                                         ,minBedrooms = min(bedrooms)
                                                         ,maxBedrooms = max(bedrooms))

input_dataset %>% filter(interest_level == "high") %>% arrange(desc(price)) %>% head(10) %>% select(price)
input_dataset %>% filter(price == 1150000) %>% summarise(n=n())
View(input_dataset[input_dataset$price == 4490000,])



#----BEst result
#1. ----------2709 + 34105 + 1200 (Price less than 90000)

#--------------------------Try naive bayes algorithm

model_naive <- naiveBayes(formula = interest_level ~ . -listing_id - created
                          ,data = input_dataset
                          )

result_nb <- predict(model_naive
              ,newdata = subset(input_dataset,select = -c(interest_level,listing_id,created))
              ,type = c("raw"))

result_nb <- data.frame(result_nb)
result_nb$predictedVal <- ifelse(result_nb$low > result_nb$high
                                      ,"low"
                                      ,ifelse(result_nb$high>result_nb$medium
                                              ,"high"
                                              ,"medium"
                                      )
)
#table(result_nb,input_dataset$interest_level)
table(result_nb$predictedVal,input_dataset$interest_level)

colnames(result_nb) <- c("high_nb","low_nb","medium_nb","predictedVal_nb")
colnames(result_xgb) <- c("listing_id","high_xgb","low_xgb","medium_xgb","predictedVal_xgb")

finalResultSet <- cbind(result_nb,result_xgb)

finalResultSet <- finalResultSet %>% mutate(
                                        high = high_xgb
                                        ,low = low_xgb
                                        , medium = (.4 * medium_xgb + .6 * medium_nb)/2)

finalResultSet$predictedVal <- ifelse(finalResultSet$low > finalResultSet$high
                                      ,"low"
                                      ,ifelse(finalResultSet$high>finalResultSet$medium
                                              ,"high"
                                              ,"medium"
                                              )
                                      )

table(finalResultSet$predictedVal, input_dataset$interest_level)