# Coursera Machine Learning Submission for H20

# Need to run install.packages("random") and install.packages("h2o")

# H2o library loading and initalization
library(h2o)
h2o.init

# Artificial Data Creation
set.seed(123)
N <- 1000
data <- data.frame(id= 1:N)

# Name generation Random ; Trying another package
library(random)
string_5 <- as.vector(randomStrings(n=N, len=5, digits=FALSE, upperalpha=TRUE,
                                    loweralpha=FALSE, unique=TRUE, check=TRUE))
data$name = string_5

# Language of the Movie , HollyWood , Nollywood etc :D
languages <-c('English','English', 'Malayalam', 'Yourba','Yourba', 'Japanese', 'Hindi', 'Spanish')
data$language <- as.factor(languages[data$id %% length(languages) +1])

# No of Fights
v = round(rnorm(N, mean=3, sd=1))
v = pmax(v,0)
v = pmin(v,4)
table(v)
data$noOfFights = v

# No of Songs
v = round(rnorm(N, mean=2, sd=1))
v = pmax(v,0)
v = pmin(v,5)
table(v)
data$noOfSongs = v

# Gross collection depends on no of fights, songs and if english or not
v = 200000 +((data$noOfFights * 300) ^ 2) +((data$noOfSongs * 200) ^ 2)
v = v * 100 *(if( c('English') %in% data$language) 0.9 else 0.3)
data$grossCollection =v

# Convert to H2o Data Frame
as.h2o(data, destination_frame = "movies")
movies <- h2o.getFrame("movies")

# Summary
summary(movies)

parts <- h2o.splitFrame(data = movies, 
                        ratios = c(0.8,0.1),
                        destination_frames = c("movies_train","movies_valid","movies_test"),
                        seed=123)

train <-h2o.getFrame("movies_train")
valid <-h2o.getFrame("movies_valid")
test  <-h2o.getFrame("movies_test")

y<- "grossCollection"

x<-setdiff(names(train), c("id",y))
# Purposefully leaving the name column in


m1 <- h2o.gbm(x, y, train,
              model_id = "movies_r",
              validation_frame = valid)

# The model will kick out name as it is bad / constanct column
# Message : 
 # In .h2o.startModelJob(algo, params, h2oRestApiVersion) :
 # Dropping bad and constant columns: [name].

h2o.performance(m1, train = TRUE)
h2o.performance(m1, valid = TRUE)
h2o.performance(m1, test)

# Overfitting by giving more trees and more depths

m2 <- h2o.gbm(x, y, training_frame = train,
              model_id = "movies_overfit_r",
              validation_frame = valid,
              ntrees = 1000,
              max_depth = 10)


h2o.performance(m2, train = TRUE)
h2o.performance(m2, valid = TRUE)
h2o.performance(m2, test)