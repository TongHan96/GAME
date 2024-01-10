library(Matrix)
library(readr)
library(dplyr)
library(pROC)
library(knitr)
# load('dat.origin.Rdata')


write_auc <- function(auc_value, filename, test=FALSE) {
  if (test){
    filename = paste0(filename, '_TEST', ".csv")
  } else {
    filename = paste0(filename, ".csv")
  }
  # Check if the file already exists
  if (file.exists(filename)) {
    # Append the auc_value to the file
    write(auc_value, file = filename, append = TRUE, sep = ",")
  } else {
    # Create a new file and write the header
    write("AUC", file = filename)
    write(auc_value, file = filename, append = TRUE, sep = ",")
  }
}


get_grad = function(M, Other, LOINC, id, coef){
  options(warn=-1)
  alpha = coef[1]
  beta = coef[2]
  lambda = coef[3]
  
  id$eMe = unlist(lapply(1:nrow(id), function(i){
    return(c(t(LOINC[id$idL[i],]) %*% M %*% t(t(Other[id$idO[i],]))))
  }))
  id$eMe = exp(id$eMe * c(-alpha, beta)[2-id$ans])
  idj = id %>%
    group_by(idL, ans) %>%
    summarise(sumj = sum(eMe), .groups = "drop")
  idj$sumj = idj$sumj  + exp(lambda*c(-alpha,beta)[2-idj$ans])
  
  gradM = lapply(1:nrow(idj), function(i){
    idi = which(id$idL==idj$idL[i] & id$ans==idj$ans[i])
    if(length(idi)==1){
      ej = t(t(Other[id$idO[idi],])) * id$eMe[idi]
    }else{
      ej = t(Other[id$idO[idi],]) %*% t(t(id$eMe[idi]))
    }
    gMsub = (-1)^(idj$ans[i]) * t(t(LOINC[idj$idL[i],])) %*% t(ej) / idj$sumj[i]
    return(gMsub)
  })
  
  gradM = Reduce("+", gradM)
  return(gradM)
}


Procrustes <- function(X1,X2){
  #Return Omega = arg min||X1 - X2 Omgea||_F
  H = t(X2)%*%X1
  mod = svd(H)
  return(mod$u%*%t(mod$v))
}



get_supervied = function(Other, LOINC, dict_label, pair_val_all,
                         pair_test_all, type = 1, coef = c(2,50,0.5),
                         maxstep = 30, epsilon = 1, stepsize = 1e-3){
    ## obtain supervised Other_embed
    ## dict_label: three columns: 'other', 'loinc' and 'ans'
    #### ans = 1 indicates positive pairs while ans = 0 indicates negative pairs.
    ## type: 
    #### type = 0: use positive pairs only; 
    #### type = 1: use both positive pairs and negative pairs.
    ## coef = c(alpha, beta, lambda)
    ## L = 1/alpha \sum log(1+\sum exp(-alpha(S_{ij}-lambda))) + 
    #### 1/beta \sum log(1+\sum exp(beta(S_{ij}-lambda)))
    set.seed(64)
    options(warn=-1)
    maxstep = max(1, maxstep)
    epsilon = max(epsilon, 1e-10)
    stepsize = max(stepsize, 1e-5)

    alpha = coef[1]
    beta = coef[2]
    lambda = coef[3]

    stopifnot(c("other","loinc","ans")%in%colnames(dict_label))
    dict_label = dict_label %>%
    filter(other%in%rownames(Other) & loinc%in%rownames(LOINC))
    if(type == 0){
        dict_label = dict_label[which(dict_label$ans==1),]
    }else if(type == 1){
        dict_label = dict_label[which(dict_label$ans%in%c(0,1)),]
    }
    stopifnot(nrow(dict_label)>2)

    olist = unique(dict_label$other)
    llist = unique(dict_label$loinc)
    Other1 = Other[match(olist,rownames(Other)),]
    LOINC = LOINC[match(llist,rownames(LOINC)),]
    id = data.frame(idO = match(dict_label$other, olist),
                  idL = match(dict_label$loinc, llist),
                  ans = dict_label$ans)

    M = diag(1, nrow = ncol(LOINC), ncol = ncol(Other))
    delta = numeric(maxstep)
    old_auc = 0
    counter <- 0
    patience <- 5
    for(step in 1:maxstep){
        if(counter >= patience)
            break
        print(paste0('--------step ',step,'--------'))
        flush.console()
        gradM = get_grad(M, Other1, LOINC, id, coef)
        M = M - stepsize * gradM
        delta[step] = norm(gradM)
        print(delta[step])
        flush.console()
        if(delta[step] < epsilon) break
        newOther = Other %*% t(M)


        if (step %% 10 == 1) {
            options(warn = -1)
            similarity_scores <- numeric(nrow(pair_val_all))
            for (i in 1:nrow(pair_val_all)) {
                # Map the names to the correct rows in R_MGB and P_MGB
                A <- suppressWarnings(newOther[rownames(Other) == pair_val_all$other[i],])
                B <- suppressWarnings(LOINC[rownames(LOINC) == pair_val_all$loinc[i],])
                similarity_scores[i] <- suppressWarnings(cosine_similarity(as.numeric(A), as.numeric(B)))
            }
            roc_obj <- roc(response = pair_val_all$ans, predictor = similarity_scores, levels = c(0, 1), direction = "<")
            write_auc(auc(roc_obj), filename)
            if (auc(roc_obj) > old_auc) {
                old_auc <- auc(roc_obj)
                counter <- 0
            } else {
                counter <- counter + 1
            }     
        }
        if (step %% 10 == 1) {
            options(warn = -1)
            similarity_scores <- numeric(nrow(pair_test_all))
            for (i in 1:nrow(pair_test_all)) {
                # Map the names to the correct rows in R_MGB and P_MGB
                A <- suppressWarnings(newOther[rownames(Other) == pair_test_all$other[i],])
                B <- suppressWarnings(LOINC[rownames(LOINC) == pair_test_all$loinc[i],])
                similarity_scores[i] <- suppressWarnings(cosine_similarity(as.numeric(A), as.numeric(B)))
            }
            roc_obj <- roc(response = pair_test_all$ans, predictor = similarity_scores, levels = c(0, 1), direction = "<")
            write_auc(auc(roc_obj), filename, test=TRUE)     
        }
  }
    delta = delta[1:step]
    if(step == maxstep) cat("not converge!","\n")
    newOther = Other %*% t(M)
    return(list(Other = newOther, M=M, delta=delta))
}


cosine_similarity <- function(A, B){
  sum(A * B) / (sqrt(sum(A^2)) * sqrt(sum(B^2)))
}

supervise_MGB <- function(pairs, name_all, datax, MGB_index, scale=c(0.01,0.01), coef=c(1,1,0.1), maxstep=1, epsilon = 1, stepsize = 1e-3)
{
    type = sapply(strsplit(name_all[MGB_index], split=':'), function(x) x[1])
    R_MGB = datax[MGB_index,][which(type == "RXNORM"),]
    P_MGB = datax[MGB_index,][which(type == "PheCode"),]
    pairs = gen_pairs(pairs, MGB_index, name_all, scale)
    pair_train_all = pairs$train
    pair_val_all = pairs$val
    pair_test_all = pairs$test
    sc_origin = c()
    for (i in 1:nrow(pair_val_all)){
        C <- R_MGB[rownames(R_MGB) == pair_val_all$other[i], ]
        B <- P_MGB[rownames(P_MGB) == pair_val_all$loinc[i], ]
        sc_origin[i] <- cosine_similarity(as.numeric(C), as.numeric(B))
    }
    roc_ori <- roc(pair_val_all$ans, sc_origin, levels = c(0, 1), direction = "<")
    print(paste('AUC_origin_val:', auc(roc_ori)))
    flush.console()
    write_auc(auc(roc_ori), filename)
    sc_origin = c()
    for (i in 1:nrow(pair_test_all)){
        C <- R_MGB[rownames(R_MGB) == pair_test_all$other[i], ]
        B <- P_MGB[rownames(P_MGB) == pair_test_all$loinc[i], ]
        sc_origin[i] <- cosine_similarity(as.numeric(C), as.numeric(B))
    }
    roc_ori <- roc(pair_test_all$ans, sc_origin, levels = c(0, 1), direction = "<")
    print(paste('AUC_origin_test:', auc(roc_ori)))
    flush.console()
    write_auc(auc(roc_ori), filename, test=TRUE)
    
    results <- get_supervied(R_MGB,P_MGB,pair_train_all,coef = coef, maxstep = maxstep, epsilon = epsilon, stepsize = stepsize, pair_val_all=pair_val_all, pair_test_all = pair_test_all)
    similarity_scores = c()
    for (i in 1:nrow(pair_test_all)){
        # Map the names to the correct rows in R_MGB and P_MGB
        A <- results$Other[rownames(results$Other) == pair_test_all$other[i], ]
        B <- P_MGB[rownames(P_MGB) == pair_test_all$loinc[i], ]
        similarity_scores[i] <- cosine_similarity(as.numeric(A), as.numeric(B))
    }
    roc_obj <- roc(pair_test_all$ans, similarity_scores, levels = c(0, 1), direction = "<")
    print(paste("AUC:", auc(roc_obj)))
    write_auc(auc(roc_ori), filename, test=TRUE)
    return(list(R_MGB=results$Other, P_MGB=P_MGB, auc=roc_obj))
}

gen_list <- function(pairs, MGB_index, name_all){
    set.seed(1)
    # pairs_MGB = pairs[which((pairs[,1] %in% name_all[MGB_index]) & (pairs[,2] %in% name_all[MGB_index])),]

    type = sapply(strsplit(name_all[MGB_index],split=':'), function(x) x[1])

    rx_ph = list()
    for(i in 1:length(which(type == 'RXNORM'))){
        rx = name_all[MGB_index][which(type == 'RXNORM')][i]
        rx_ph[[i]] = pairs[which(pairs[,1] == rx),2]
    }
    names(rx_ph) = name_all[MGB_index][which(type == 'RXNORM')]

    neg_rx_ph = list()
    for(i in 1:length(which(type == 'RXNORM'))){
        neg_rx_ph[[i]] = sample(setdiff(name_all[MGB_index][which(type == 'PheCode')], rx_ph[[i]]), length(rx_ph[[i]]), replace=FALSE)
    }

    names(neg_rx_ph) = name_all[MGB_index][which(type == 'RXNORM')]
    return(list(pos_list = rx_ph, neg_list = neg_rx_ph))
}

list_to_dataframe <- function(lst) {
  dfs <- list()  # Initialize an empty list to store temporary data frames
  
  for(i in seq_along(lst)) {
    if(length(lst[[i]]) > 0) {
      temp_df <- data.frame(Key = rep(names(lst)[i], length(lst[[i]])),
                            Value = lst[[i]], 
                            stringsAsFactors = FALSE)
      dfs[[i]] <- temp_df
    }
  }
  
  # Combine all dataframes in the list
  df <- do.call(rbind, dfs)
  return(df)
}

gen_pairs <- function(pairs, MGB_index, name_all, scale=c(0.01,0.01)){
    set.seed(1)
    MGB_lis = gen_list(pairs, MGB_index, name_all)
    MGB_lis_pos = list_to_dataframe(MGB_lis$pos_list)
    MGB_lis_pos$ans = 1
    MGB_lis_neg = list_to_dataframe(MGB_lis$neg_list)
    MGB_lis_neg$ans = 0
    colnames(MGB_lis_pos) = colnames(MGB_lis_neg) = c('other','loinc','ans')
    train_name = sample(names(MGB_lis$pos_list), scale[1] * length(names(MGB_lis$pos_list)), replace=FALSE)
    val_name = sample(setdiff(names(MGB_lis$pos_list), train_name), scale[2] * length(names(MGB_lis$pos_list)), replace=FALSE)
    test_name = setdiff(names(MGB_lis$pos_list), union(train_name, val_name))
    all_table = rbind(MGB_lis_pos, MGB_lis_neg)
    return(list(train = all_table[which(all_table[,1] %in% train_name),], val = all_table[which(all_table[,1] %in% val_name),], test = all_table[which(all_table[,1] %in% test_name),]))
}

get_auc_all <- function(pairs, datax, scale=c(0.01,0.01), M=200, step_size=1e-4, coef=c(2,50,0.5)){
    MGB_ALL <- supervise_MGB(pairs, name_all, datax, MGB_index=MGB_index, coef=coef, scale=scale, maxstep = M, stepsize = step_size, epsilon = 1e-1)
    VA_ALL <- supervise_MGB(pairs, name_all, datax, MGB_index=VA_index, coef=coef, scale=scale, maxstep = M, stepsize = step_size, epsilon = 1e-1)
    UP_ALL <- supervise_MGB(pairs, name_all, datax, MGB_index=UP_index, coef=coef, scale=scale, maxstep = M, stepsize = step_size, epsilon = 1e-1)
}

#---------------------load data----------------------
setwd('/home/ec2-user/SageMaker/drug_side')

datax <- read.csv(paste0('https://han-attention.s3.amazonaws.com/input/drug_side/drug_side_',now_method,'.csv'))

name_all <- datax[,1]
datax <- t(apply(datax[,2:769], 1, as.numeric))
rownames(datax) = name_all

pairs_all <- read.csv('input/Drug_Indi_Side_From_Jun_0803.csv')
pairs_all = pairs_all[,1:2]

MGB_index = 1:3007
VA_index = 3008:6252
UP_index = 6253:10080

now_time <- format(Sys.time(), "%Y%m%d_%H%M%S")
filename <- paste0("output/AUC_drug_side_effect_", now_method, now_time)
get_auc_all(pairs_all, datax, scale = c(0.01,0.01), M=3000, step_size=now_step)