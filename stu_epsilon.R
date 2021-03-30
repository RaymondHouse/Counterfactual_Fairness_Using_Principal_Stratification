# adult data

rm(list=ls())
library(rpart)
library(Rglpk)
library(pcalg)
library(RBGL)
library(graph)
# library(CAM)
library(CondIndTests)
library(DMwR)
library(bnlearn)
library(lattice)
library(MASS)
library(nnet)
library(mice) 
library(infotheo)
library(klaR)
library(e1071)
library(Rsolnp)


fairness_epsilon <- function(pred){
  n = dim(pred)[1]
  # coding:
  # 0000 - 1    0001 - 2    0010 - 3    0011 - 4
  # 0100 - 5    0101 - 6    0110 - 7    0111 - 8
  # 1000 - 9    1001 - 10   1010 - 11   1011 - 12
  # 1100 - 13   1101 - 14   1110 - 15   1111 - 16
  # pred real attribute
  outcome<-numeric(12)
  coefficient <- matrix(numeric(32), ncol = 4)
  ## P(0,0 |a=0)
  # a1<-c(1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0)
  #pred[, 1]是S, 2是Y, 3是A
  a1<-numeric(16)
  a1[c(1,2,5,6)]<-1
  outcome[1] <- sum(pred[, 1] == 0 & pred[, 2] == 0 & pred[, 3] == 0)/sum(pred[, 3] == 0)
  coefficient[1,2] <- outcome[1]
  coefficient[1,1] <- outcome[1] - sum(pred[, 2] == 0 & pred[, 3] == 0)/sum(pred[, 3] == 0)
  ## p(0,1| a=0)
  a2<-numeric(16)
  a2[c(3,4,7,8)]<-1
  outcome[2] <- sum(pred[, 1] == 0 & pred[, 2] == 1 & pred[, 3] == 0)/sum(pred[, 3] == 0)
  coefficient[2,2] <- outcome[2]
  coefficient[2,1] <- (outcome[2] - sum(pred[, 2] == 1 & pred[, 3] == 0)/sum(pred[, 3] == 0))
  ## p(1,0| a=0)
  a3<-numeric(16)
  a3[c(9,10,13,14)]<-1
  outcome[3] <- sum(pred[, 1] == 1 & pred[, 2] == 0 & pred[, 3] == 0)/sum(pred[, 3] == 0)
  coefficient[3,2] <- (outcome[3] - sum(pred[, 2] == 0 & pred[, 3] == 0)/sum(pred[, 3] == 0))
  coefficient[3,1] <- outcome[3]
  ## p(1,1| a=0)
  a4<-numeric(16)
  a4[c(11,12,15,16)]<-1
  outcome[4] <- sum(pred[, 1] == 1 & pred[, 2] == 1 & pred[, 3] == 0)/sum(pred[, 3] == 0)
  coefficient[4,2] <- (outcome[4] - sum(pred[, 2] == 1 & pred[, 3] == 0)/sum(pred[, 3] == 0))
  coefficient[4,1] <- outcome[4]
  
  ## P(0,0 |a=1)
  a5<-numeric(16)
  a5[c(1,3,9,11)]<-1
  outcome[5] <- sum(pred[, 1] == 0 & pred[, 2] == 0 & pred[, 3] == 1)/sum(pred[, 3] == 1)
  coefficient[5,4] <- outcome[5]
  coefficient[5,3] <- (outcome[5] - sum(pred[, 2] == 0 & pred[, 3] == 1)/sum(pred[, 3] == 1))
  ## p(0,1| a=1)
  a6<-numeric(16)
  a6[c(2,4,10,12)]<-1
  outcome[6] <- sum(pred[, 1] == 0 & pred[, 2] == 1 & pred[, 3] == 1)/sum(pred[, 3] == 1)
  coefficient[6,4] <- outcome[6]
  coefficient[6,3] <- (outcome[6] - sum(pred[, 2] == 1 & pred[, 3] == 1)/sum(pred[, 3] == 1))
  ## p(1,0| a=1)
  a7<-numeric(16)
  a7[c(5,7,13,15)]<-1
  outcome[7] <- sum(pred[, 1] == 1 & pred[, 2] == 0 & pred[, 3] == 1)/sum(pred[, 3] == 1)
  coefficient[7,4] <- (outcome[7] - sum(pred[, 2] == 0 & pred[, 3] == 1)/sum(pred[, 3] == 1))
  coefficient[7,3] <- outcome[7]
  ## p(1,1| a=1)
  a8<-numeric(16)
  a8[c(6,8,14,16)]<-1
  outcome[8] <- sum(pred[, 1] == 1 & pred[, 2] == 1 & pred[, 3] == 1)/sum(pred[, 3] == 1)
  coefficient[8,4] <- (outcome[8] - sum(pred[, 2] == 1 & pred[, 3] == 1)/sum(pred[, 3] == 1))
  coefficient[8,3] <- outcome[8]
  ## tau' == 0
  a9<-numeric(20)
  a9[c(5, 8)] <- 1
  a9[c(9, 12)] <- -1
  
  outcome[9:10] = rep(1,2)
  
  pf <- numeric(20)
  # pf[c(5, 8)] <- 1
  # pf[c(9, 12)] <- -1
  pf[17:20] = rep(1,4)
  pf2 <- numeric(16)
  
  tpf1 <- numeric(20)
  tpf2 <- numeric(20)
  tpf1[5] <- 1
  tpf1[9] <- -1
  tpf2[8] <- 1
  tpf2[12] <- -1
  
  # potential
  mat.weak.potential <- rbind(a1,a2,a3,a4,a5,a6,a7,a8)
  mat.weak.potential_epsilon <- rbind(cbind(mat.weak.potential, coefficient), cbind(matrix(0, nrow = 2, ncol = 16), rbind(c(1,1,0,0), c(0,0,1,1))), a9)
  mat.weak.potential <- as.matrix(mat.weak.potential)
  mat.weak.potential_epsilon <- as.matrix(mat.weak.potential_epsilon)
  
  # total potential
  mat.potential_8_12 <- rbind(cbind(mat.weak.potential, coefficient), cbind(matrix(0, nrow = 2, ncol = 16), rbind(c(1,1,0,0), c(0,0,1,1))), tpf1, tpf2)
  mat.potential_8_12 <- as.matrix(mat.potential_8_12)
  
  mat.potential_5_9 <- rbind(cbind(mat.weak.potential, coefficient), cbind(matrix(0, nrow = 2, ncol = 16), rbind(c(1,1,0,0), c(0,0,1,1))), tpf1, tpf2)
  mat.potential_5_9 <- as.matrix(mat.potential_5_9)
  
  dir <- numeric(11)
  for (i in c(1: 8)){dir[i]<-'=='}
  for (i in c(9:10)){dir[i]<-"<="}
  for (i in c(11:12)){dir[i]<-"=="}
  
  obj_weak <- pf
  # obj_8_12 <- tpf2
  # obj_5_9 <- tpf1
  
  obj_8_12 <- pf
  obj_5_9 <- pf
  
  info <- numeric(10)
  
  # test outcome
  # soutcome = c(0.55,0.05,0.05,0.35,0.85,0.05,0.05,0.05,0,0)
  
  res.weak.potential_1 <- Rglpk_solve_LP(obj_weak, mat.weak.potential_epsilon, dir[1:11], outcome[1:11], types='C', max = F)
  res.weak.potential_2 <- Rglpk_solve_LP(obj_weak, mat.weak.potential_epsilon, dir[1:11], outcome[1:11], types='C', max = T)
  info[1:2] <- c(res.weak.potential_1$status, res.weak.potential_2$status)
  
  res.potential_8_12_1 <- Rglpk_solve_LP(obj_8_12, mat.potential_8_12, dir[1:12], outcome[1:12], types='C', max = F)
  res.potential_8_12_2 <- Rglpk_solve_LP(obj_8_12, mat.potential_8_12, dir[1:12], outcome[1:12], types='C', max = T)
  info[3:4] <- c(res.potential_8_12_1$status, res.potential_8_12_2$status)
  
  # if (res.potential_8_12_1$status == 1){
  #   res.potential_8_12_1 <- Rglpk_solve_LP(obj_8_12, mat.weak.potential, dir[1:8], outcome[1:8], types='C', max = F)
  #   res.potential_8_12_2 <- Rglpk_solve_LP(obj_8_12, mat.weak.potential, dir[1:8], outcome[1:8], types='C', max = T)
  #   info[5:6] <- c(res.potential_8_12_1$status, res.potential_8_12_2$status)
  # }
  
  #res.potential_5_9_1 <- Rglpk_solve_LP(obj_5_9, mat.potential_5_9, dir, outcome[1:9], types='C', max = F)
  #res.potential_5_9_2 <- Rglpk_solve_LP(obj_5_9, mat.potential_5_9, dir, outcome[1:9], types='C', max = T)
  #info[7:8] <- c(res.potential_5_9_1$status, res.potential_5_9_2$status)
  
  if (TRUE){
    res.potential_5_9_1 <- Rglpk_solve_LP(obj_5_9, mat.potential_5_9, dir[1:12], outcome[1:12], types='C', max = F)
    res.potential_5_9_2 <- Rglpk_solve_LP(obj_5_9, mat.potential_5_9, dir[1:12], outcome[1:12], types='C', max = T)
    # print(c(res.potential_5_9_1$optimum, res.potential_5_9_2$optimum))
    info[9:10] <- c(res.potential_5_9_1$status, res.potential_5_9_2$status)
  }
  
  res <- c(min(res.weak.potential_1$optimum, res.weak.potential_2$optimum), max(res.weak.potential_1$optimum, res.weak.potential_2$optimum),
           min(res.potential_8_12_1$optimum, res.potential_8_12_2$optimum), max(res.potential_8_12_1$optimum, res.potential_8_12_2$optimum),
           min(res.potential_5_9_1$optimum, res.potential_5_9_2$optimum), max(res.potential_5_9_1$optimum, res.potential_5_9_2$optimum))
  param <- res.weak.potential_1$solution[17:20]
  
  # res <- c(res.weak.potential_1$optimum, res.weak.potential_2$optimum,
  #          res.potential_8_12_1$optimum, res.potential_8_12_2$optimum,
  #          res.potential_5_9_1$optimum, res.potential_5_9_2$optimum)
  # 
  
  return(list(res = res, info = info, param = param))
}

## ===== processing data =====

data_mat <- read.csv("student-mat.csv", header = TRUE)
data_por <- read.csv("student-por.csv", header = TRUE, sep = ';')

for (i in 1:29){
  data_mat[, i] <- as.numeric(as.factor(data_mat[,i]))
  data_por[, i] <- as.numeric(as.factor(data_por[,i]))
}
# age
data_mat[data_mat[,3] > 18 ,3] <- 19
data_mat[,3] <- data_mat[,3] - 14
data_por[data_por[,3] > 18 ,3] <- 19
data_por[,3] <- data_por[,3] - 14
# travel time
data_mat[data_mat[,13] == 3 ,13] <- 2
data_mat[data_mat[,13] == 4 ,13] <- 3
data_por[data_por[,13] == 3 ,13] <- 2
data_por[data_por[,13] == 4 ,13] <- 3
# medu and fedu
data_mat[data_mat[,7] == 0 ,7] <- 1
data_mat[data_mat[,8] == 0 ,8] <- 1
data_por[data_por[,7] == 0 ,7] <- 1
data_por[data_por[,8] == 0 ,8] <- 1
# failure
data_mat[, 15] <- data_mat[, 15] + 1
data_por[, 15] <- data_por[, 15] + 1
# absence
data_mat[data_mat[, 30] > 0 & data_mat[, 30] <= 5, 30] <- 1
data_mat[data_mat[, 30] > 3 & data_mat[, 30] <= 5, 30] <- 2
data_mat[data_mat[, 30] > 5, 30] <- 3
data_por[data_por[, 30] > 0 & data_por[, 30] <= 5, 30] <- 1
data_por[data_por[, 30] > 3 & data_por[, 30] <= 5, 30] <- 2
data_por[data_por[, 30] > 5, 30] <- 3
data_mat[, 30] <- data_mat[, 30] + 1
data_por[, 30] <- data_por[, 30] + 1

# score
# data_mat[, 31] <- discretize(data_mat[, 31],"equalfreq",3)
# data_por[, 31] <- discretize(data_por[, 31],"equalfreq",3)
# data_mat[, 32] <- discretize(data_mat[, 32],"equalfreq",3)
# data_por[, 32] <- discretize(data_por[, 32],"equalfreq",3)

# data_mat[, 33] <- discretize(data_mat[, 33],"equalfreq",3)
# data_por[, 33] <- discretize(data_por[, 33],"equalfreq",3)

data_por[data_por[, 33]<10, 33] <- 1
data_por[data_por[, 33]>=10, 33] <- 2
data_mat[data_mat[, 33]<10, 33] <- 1
data_mat[data_mat[, 33]>=10, 33] <- 2

data_por[data_por[, 32]<10, 32] <- 1
data_por[data_por[, 32]>=10, 32] <- 2
data_mat[data_mat[, 32]<10, 32] <- 1
data_mat[data_mat[, 32]>=10, 32] <- 2


data_por[data_por[, 31]<10, 31] <- 1
data_por[data_por[, 31]>=10, 31] <- 2
data_mat[data_mat[, 31]<10, 31] <- 1
data_mat[data_mat[, 31]>=10, 31] <- 2


# data_por[data_por[, 32]<3, 32] <- 1 
# data_por[data_por[, 32]==3, 32] <- 2
# data_mat[data_mat[, 32]<3, 32] <- 1 
# data_mat[data_mat[, 32]==3, 32] <- 2 
# 
# data_por[data_por[, 31]<3, 31] <- 1 
# data_por[data_por[, 31]==3, 31] <- 2
# data_mat[data_mat[, 31]<3, 31] <- 1 
# data_mat[data_mat[, 31]==3, 31] <- 2 

# data_mat[, 33] <- (data_mat[, 33])>(data_mat[, 31])
# data_por[, 33] <- (data_por[, 33])>(data_por[, 31])

for (i in 1:33){
  data_mat[, i] <- as.factor(data_mat[,i])
  data_por[, i] <- as.factor(data_por[,i])
}




## epsilon  math=============================================================================
target <- 33

model <- naiveBayes(data_mat[, c(1:30)], data_mat[, target])
pred <- predict(model, data_mat[, c(1:30)], type = "class")

bound_mat <- matrix(0, 7, 6)
info_mat <- matrix(0, 7, 10)

tb <- table(pred, data_mat[,target])
(tb[1,1] + tb[2,2])/length(pred)

# sex
i = 2
input.data <- cbind(as.numeric(pred), as.numeric(data_mat[,target]), as.numeric(data_mat[, i]))-1
result <- fairness_epsilon(input.data)
bound_mat[1, ] <- c(result$res)
info_mat[1, ] <- c(result$info)

# address | sex

i = 4
input.data <- cbind(as.numeric(pred), as.numeric(data_mat[,target]), as.numeric(data_mat[, i]), as.numeric(data_mat[, 2]))-1

# sex = 0
result <- fairness_epsilon(input.data[input.data[, 4] == 0, 1:3])
bound_mat[2, ] <- c(result$res)
info_mat[2, ] <- c(result$info)

# sex = 1
result <- fairness_epsilon(input.data[input.data[, 4] == 1, 1:3])
bound_mat[3, ] <- c(result$res)
info_mat[3, ] <- c(result$info)

# higher | sex, address
i = 21
input.data <- cbind(as.numeric(pred), as.numeric(data_mat[,target]), as.numeric(data_mat[, i]),
                    as.numeric(data_mat[, 2]), as.numeric(data_mat[, 4]))-1
# sex = 0
result <- fairness_epsilon(input.data[(input.data[, 4] == 0), 1:3])
bound_mat[4, ] <- c(result$res)
info_mat[4, ] <- c(result$info)
# sex = 1
result <- fairness_epsilon(input.data[(input.data[, 4] == 1), 1:3])
bound_mat[5, ] <- c(result$res)
info_mat[5, ] <- c(result$info)
# address = 0
result <- fairness_epsilon(input.data[ input.data[, 5] == 0, 1:3])
bound_mat[6, ] <- c(result$res)
info_mat[6, ] <- c(result$info)
# address = 1
result <- fairness_epsilon(input.data[input.data[, 5] == 1, 1:3])
bound_mat[7, ] <- c(result$res)
info_mat[7, ] <- c(result$info)

#por========================================================================


model <- naiveBayes(data_por[, c(1:30)], data_por[, target])
pred <- predict(model, data_por[, c(1:30)], type = "class")

tb <- table(pred, data_por[,target])
(tb[1,1] + tb[2,2])/length(pred)
bound_por <- matrix(0, 7, 6)
info_por <- matrix(0, 7, 10)
param_por <- matrix(0, 7, 4)

# sex
i = 2
input.data <- cbind(as.numeric(pred), as.numeric(data_por[,target]), as.numeric(data_por[, i]))-1
result <- fairness_epsilon(input.data)
bound_por[1, ] <- c(result$res)
info_por[1, ] <- c(result$info)
param_por[1,] <- c(result$param)

# address | sex
i = 4
input.data <- cbind(as.numeric(pred), as.numeric(data_por[,target]), as.numeric(data_por[, i]), as.numeric(data_por[, 2]))-1
# sex = 0
result <- fairness_epsilon(input.data[input.data[, 4] == 0, 1:3])
bound_por[2, ] <- c(result$res)
info_por[2, ] <- c(result$info)
param_por[2,] <- c(result$param)
# sex = 1
result <- fairness_epsilon(input.data[input.data[, 4] == 1, 1:3])
bound_por[3, ] <- c(result$res)
info_por[3, ] <- c(result$info)
param_por[3,] <- c(result$param)

# edu-supp | sex, address
i = 21
input.data <- cbind(as.numeric(pred), as.numeric(data_por[,target]), as.numeric(data_por[, i]),
                    as.numeric(data_por[, 2]), as.numeric(data_por[, 4]))-1

# sex = 0
result <- fairness_epsilon(input.data[(input.data[, 4] == 0), 1:3])
bound_por[4, ] <- c(result$res)
info_por[4, ] <- c(result$info)
param_por[4,] <- c(result$param)
# sex = 1
result <- fairness_epsilon(input.data[(input.data[, 4] == 1), 1:3])
bound_por[5, ] <- c(result$res)
info_por[5, ] <- c(result$info)
param_por[5,] <- c(result$param)
# address = 0
result <- fairness_epsilon(input.data[ input.data[, 5] == 0, 1:3])
bound_por[6, ] <- c(result$res)
info_por[6, ] <- c(result$info)
param_por[6,] <- c(result$param)
# address = 1
result <- fairness_epsilon(input.data[ input.data[, 5] == 1, 1:3])
bound_por[7, ] <- c(result$res)
info_por[7, ] <- c(result$info)
param_por[7,] <- c(result$param)


round(bound_mat, 3)

round(bound_por, 3)

round(param_por, 3)



