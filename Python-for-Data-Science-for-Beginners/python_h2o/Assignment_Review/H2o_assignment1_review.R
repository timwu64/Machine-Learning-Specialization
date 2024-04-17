# Assignment #1
#install.packages("h2o")
library(h2o)
h2o.init()

set.seed(11)

# data generator
# house price list: zone, building_age, n_rooms, total_size (sq meters), price

N <- 1000

df <- data.frame(id=1:N)

zones <- c("zone-a", "zone-a", "zone-a", "zone-c", "zone-d", "zone-e")
df$zone <- as.factor( zones[(df$id %% length(zones)+1)] )  # (df$id %% length(zone)+1)  zone pointer

df$n_rooms <- round( runif(df$id, min=2, max=6), 0)

df$size <- df$n_rooms * 20 
df$size <- df$size + runif(df$id, min=1, max=5)

df$building_age <- round(runif(df$id, min=5, max=70), 0)

df$price <- round( df$size * (as.numeric(df$zone)*2000)
                   + df$building_age * 1000, 0)

plot(df$zone, df$price)

h_pl <- as.h2o(df, destination_frame="house_price_list")

# split dataset

parts <- h2o.splitFrame(h_pl, c(0.8, 0.1))

sapply(parts, nrow)

train <- parts[[1]]
valid <- parts[[2]]
test  <- parts[[3]]

# models creation

y <- "price"
x <- setdiff(names(train), c("id",y))

# -----------------------------------------------
m1 <- h2o.randomForest(x, y, train,
                       validation_frame = valid,
                       model_id = "hpl_rf_model_1")

h2o.performance(m1, train=T)
h2o.performance(m1, valid=T)
h2o.performance(m1, test)

# -----------------------------------------------
m2 <- h2o.gbm(x, y, train,
              validation_frame = valid,
              model_id = "hpl_gbm_model_2")

h2o.performance(m2, train=T)
h2o.performance(m2, valid=T)
h2o.performance(m2, test)  # better 

# -----------------------------------------------
m3 <- h2o.gbm(x, y, train,
              validation_frame = valid,
              model_id = "hpl_gbm_model_3",
              max_depth = 3,
              ntree=150)

h2o.performance(m3, train=T)
h2o.performance(m3, valid=T)
h2o.performance(m3, test)  # it begins to overfit 
