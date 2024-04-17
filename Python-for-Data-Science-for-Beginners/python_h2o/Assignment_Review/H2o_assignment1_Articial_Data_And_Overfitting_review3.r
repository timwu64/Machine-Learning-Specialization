#step 0: clear memory and load h2o

rm(list = ls())


library(h2o)

h2o.init()

#step 1: create artificial data

set.seed(123)

N <- 1000

bloodTypes <- c('A','O','AB','B')

d <- data.frame(id = 1:N)
d$bloodTypes <- bloodTypes[d$id %% length(bloodTypes) + 1]

head(d)

bloodTypes <- c('A','A','A','O','O','O','AB','B')

d <- data.frame(id = 1:N)
d$bloodTypes <- bloodTypes[d$id %% length(bloodTypes) + 1]

d$bloodTypes <- as.factor(d$bloodTypes)

d$age <- runif(N, min = 18, max = 65)

v <- round(rnorm(N, mean = 5, sd = 2))
v <- pmax(v,0)
v <- pmin(v,9)
table(v)
d$healthyEating <- v

v <- round(rnorm(N, mean = 5, sd = 2))
v <- v + ifelse(d$age < 30, 1, 0)
v <- pmax(v,0)
v <- pmin(v,9)
table(v)
d$activeLifestyle <- v

v <- 20000 + ((d$age*3)^2)
v <- v + d$healthyEating * 500
v <- v + d$activeLifestyle * 300
v <- v + runif(N, 0, 5000) #noise
d$income <- round(v, -2) #noise

head(d)

#step2: import data to h2o
as.h2o(d, destination_frame = 'people')

people <- h2o.getFrame('people')
summary(people)
summary(d)

#step 3: data partition 

parts <- h2o.splitFrame(
  people, 
  c(0.8,0.1), 
  destination_frames = c('people_train','people_valid', 'people_test'),
  seed = 123
)

sapply(parts, nrow)

train <- parts[[1]]
valid <- parts[[2]]
test <- parts[[3]]


#step 4: fit GBM to fake data
y <- 'income'
x <- setdiff(names(train), c('id', y))

m1 <- h2o.gbm(x, y, train, model_id = 'default_r', validation_frame = valid)

h2o.performance(m1, train = TRUE)
h2o.performance(m1, valid = TRUE)
h2o.performance(m1, test)

#step 5: overfit GBM 

m2 <- h2o.gbm(x, y, train, model_id = 'overfit_r', validation_frame = valid,
              ntrees = 1000, max_depth = 10)


h2o.performance(m2, train = TRUE)
h2o.performance(m2, valid = TRUE)
h2o.performance(m2, test)
