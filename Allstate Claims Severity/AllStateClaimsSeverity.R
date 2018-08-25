library(dplyr)
library(ggplot2)
#library(e1071)
library(corrplot)

setwd("F:/Sid/Learnings/Machine Learning With Kaggle/Allstate Claims Severity")

dat <- read.csv("train.csv",header = TRUE,sep = ",",stringsAsFactors = TRUE)

#---------Create categorical and continuous variables
cont.var <- paste("cont",1:14,sep = "")
cat.var <- paste("cat",1:116,sep = "")

#check plots of multiple 
corrplot(cor(dat[cont.var]),method = "number")

#-------REmove outlier
dat <- dat[dat$loss != 121012.25,]

#------cont11 and cont12 are highly corelated. We can remove this by their average
dat$cont11 <- (dat$cont11 + dat$cont12) / 2

#----------Remove 12th index
dat <- dat[-1 * which(colnames(dat) == "cont12")]

#Merge levels in factor where total count of a level is less that 20
for(i in 1:length(cat.var)){
  catName <- cat.var[i]
  Other <- data.frame(ftable(dat[,catName]))
  var<- as.character(Other[Other$Freq <= 20,1])
  if(length(var) > 0)
  {
    x <- dat[,catName]
    index <- which(levels(x) %in% var,TRUE)
    levels(dat[,catName])[index] <- c("Other")
  }
}

#----------Relationship
#-----------Cont1 (random variation)
#--------cont2 (need to look)
#--------cont3 (need to look)
#--------cont4 (need to look)
#--------cont5 (random variation)
#--------cont6 (random variation)
#--------cont7 (need to look) take log(x)
#--------cont8 (random variation)
#--------cont9 (random variation)
#--------cont10 (random variation)
#--------cont11 (need to look)
#--------cont13 (need to look)
#--------cont14 (need to look)

#--cont1 + cont2 + cont3 + cont7 + cont8 + cont9 + cont10

ggplot(dat,aes(x = log(loss)))+
  geom_density()
ggplot(dat,aes(y=dat$loss,x=log(dat$cont2)))+
  geom_point()
screen(2)
ggplot(dat,aes(y=dat$loss,x=dat$cont10))+
  geom_point()

formula <- as.formula(paste(c("loss",paste(c(paste(cat.var,collapse = "+"),paste(cont.var,collapse = "+")),collapse = "+")),collapse = " ~ "))
#---------lm1 ---normal
#---------lm2 ----log(loss)
lm1 <- lm(data = dat,loss ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + 
            cat9 + cat10 + cat11 + cat12 + cat13 + cat14 + cat15 + cat16 + 
            cat17 + cat18 + cat19 + cat20 + cat21 + cat22 + cat23 + cat24 + 
            cat25 + cat26 + cat27 + cat28 + cat29 + cat30 + cat31 + cat32 + 
            cat33 + cat34 + cat35 + cat36 + cat37 + cat38 + cat39 + cat40 + 
            cat41 + cat42 + cat43 + cat44 + cat45 + cat46 + cat47 + cat48 + 
            cat49 + cat50 + cat51 + cat52 + cat53 + cat54 + cat55 + cat56 + 
            cat57 + cat58 + cat59 + cat60 + cat61 + cat62 + cat63 + cat64 + 
            cat65 + cat66 + cat67 + cat68 + cat69 + cat70 + cat71 + cat72 + 
            cat73 + cat74 + cat75 + cat76 + cat77 + cat78 + cat79 + cat80 + 
            cat81 + cat82 + cat83 + cat84 + cat85 + cat86 + cat87 + cat88 + 
            cat89 + cat90 + cat91 + cat92 + cat93 + cat94 + cat95 + cat96 + 
            cat97 + cat98 + cat99 + cat100 + cat101 + cat102 + cat103 + 
            cat104 + cat105 + cat106 + cat107 + cat108 + cat109 + cat110 + 
            cat111 + cat112 + cat113 + cat114 + cat115 + cat116 + cont1 + 
            cont2 + cont3 + cont4 + cont5 + cont6 + cont7 + cont8 + cont9 + 
            cont10 + cont11 + cont13 + cont14)

lm2 <- lm(data = dat,loss ~ cat1 + cat2 + cat5 + 
            cat9 + cat10 + cat11 + cat12 + cat13 + cat14 + cat16 + 
            cat17 + cat19 + cat22 + cat23 + 
            cat25 + cat26 + cat27 + cat28 + cat29 + cat31 + cat32 + 
            cat33 + cat34 + cat35 + cat36 + cat37 + cat39 + cat40 + 
            cat41 + cat43 + cat44 + 
            cat49 + cat51 + cat52 + cat53 + cat57 +  cat67 + cat71 + cat72 + 
            cat73 + cat75 + cat79 +
            cat81 + cat82 + cat83 + cat84 + cat85 + cat86 + 
            cat91 + cat92 + cat93 + cat97 + cat100 + cat103 + 
            cont1 * cont2 * cont3 * cont7 * cont8 * cont9 * cont10
          )

summary(lm2)

#----------Y on Predicted Y
ggplot(dat,aes(x=predict(lm2),y=dat$loss))+
  geom_point()+
  geom_smooth(method = lm)

#----------Residual on predicted Y
ggplot(dat,aes(y=resid(lm2),x=predict(lm2)))+
  geom_point()+
  geom_smooth(method = lm)

plot(lm2)


