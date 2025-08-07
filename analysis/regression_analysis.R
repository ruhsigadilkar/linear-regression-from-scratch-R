#--- Set-up ---
setwd("C:/Users/franc/Desktop/UNI/2 - Data science/homework")  # usato per cambiare la directory di lavoro
dataset <- read.csv("Regression2024.csv") # nella variabile dataset ora è contenuto il file .csv specificato come parametro
head(dataset)  # Mostra le prime righe
names(dataset) # Restituisce i nomi delle voci contenute nel dataset
sink(file='raw_output_analysis.txt')

if (!dir.exists("graphs")) {
  dir.create("graphs")
}


set.seed(180103)
n <- nrow(dataset)
train_size <- floor(0.8 * n)
# scegliamo un numero train_size casuale di numeri da 1 a n
train_indices <- sample(seq_len(n), size = train_size)

train_set <- dataset[train_indices, ]
test_set <- dataset[-train_indices, ]

x_test = model.matrix(Y~., test_set)[,-1]
y_test = test_set$Y

cat("============================================================\n")
cat("            REGRESSION ANALYSIS REPORT\n")
cat("------------------------------------------------------------\n")
cat("Working Directory:", getwd(), "\n")
cat("Dataset: Regression2024.csv\n")
cat("============================================================\n\n")


# ----------------------------------------------------------------PUNTO A -> concluso-----------------------------------------------------------
cat("\n------------------ A. CORRELATION ANALYSIS ------------------\n\n")

# --- Calcolo e valutazione della Correlazione ---
# Correlazione: misura l'associazione lineare tra due variabili e varia tra -1, massima correlazione negativa, ed 1, massima correlazione positiva

# le istruzioni successive calcolano e visualizzano la matrice di correlazione
corDataset <- round(cor(dataset), digits =2)
View(corDataset)

# per leggere le correlazioni installo l'apposito pacchetto
#install.packages("corrplot")
library(corrplot) # istruzione per importare il package scaricato 

#invisible(dev.new()) # corrisponde al figure in Matlab, apre una nuova finestra di lavoro
png("graphs/corrplot_matrice.png", width = 800, height = 600)

# Il comando successivo visualizza la matrice di correlazione con un grafico che utilizza ellissi per rappresentare la forza e la direzione della correlazione:
# Ellissi strette e inclinate verso destra: correlazione positiva forte.
# Ellissi strette e inclinate verso sinistra: correlazione negativa forte.
# Cerchio perfetto: correlazione vicina a 0 (nessuna relazione lineare).
# Colori:
# Blu scuro: correlazioni positive.
# Rosso scuro: correlazioni negative.
# In sostanza in questo grafico il colore oppure la forma + la direzione ci danno la stessa informazione
corrplot(corDataset, method = 'ellipse')  # va bene che ci sia anche la Y perchè la matrice è di CORRELAZIONE, non di collinearità, 
#e quindi può avere senso visualizzare anche la Y
dev.off()

# DAI VARI VALORI DI CORRELAZIONE, SCRIVIAMO IL VALORE MEDIO DELLE CORRELAZIONI, IL MAX E QUELLI CHE SONO PROBLEMATICI QUINDI il cui valore assoluto è >0.5
# La riga successiva calcola il valore medio delle correlazioni: richiamo sulla matrice triangolare inferiore, escludendo la diagonale, la funzione di media
average_correlation <- mean(corDataset[lower.tri(corDataset, diag = FALSE)])
cat("Average absolute correlation (excluding diagonal):\n")
cat("  → ", round(average_correlation, 3), "\n\n")
# la correlazione media non sembra problematica

#install.packages("reshape2")
library(reshape2)

mat_change_diag <- corDataset # copio la matrice così da non modificare l'originale
mat_change_diag <- abs(mat_change_diag) # calcolo il valore assoluto dei valori nella matrice copiata così da calcolare agevolmente i valori max
mat_change_diag[lower.tri(mat_change_diag, diag = TRUE)] <- NA # sulla matrice triangolare inferiore e sulla diagonale setta come valori NA poichè 
# sulla diagonale principale i valori sono sempre 1 e la matrice in questione è simmetrica

cor_melted <- melt(mat_change_diag, na.rm = TRUE) # Converte la matrice in un data frame con tre colonne: Var1, Var2, e value
# Creiamo un vettore associativo (named vector)
# il comando paste combina Var1 e Var2 - rispettivamente righe e colonne - in un unico nome, separato da "_"
# il vettore associativo avrà valore corrispondente alle correlazioni, e nome corrispondente a quello creato con paste
cor_vector <- setNames(cor_melted$value, paste(cor_melted$Var1, cor_melted$Var2, sep = "_"))

cat("Maximum correlation:\n") # si procede alla stampa dell'etichetta e del valore del massimo del vettore delle correlazioni
print(round(cor_vector[which.max(cor_vector)], 3))
# il valore della massima correlazione risulta non problematico in quanto coinvolge un regressore e la variabile dipendente
# quindi è di fatto positivo che ci sia una correlazione rilevante

cat("\nCorrelations with absolute value > 0.5:\n") # si procede alla stampa di tutte le correlazioni maggiori di 0.5 in valore assoluto
if (length(cor_vector[cor_vector > 0.5]) > 0) {
  print(round(cor_vector[cor_vector > 0.5], 3))
} else {
  cat("  → None detected\n")
}
# risulta presente solo una correlazione in modulo maggiore di 0.5, la stessa identificata come massimo.
# Si conclude che tutte le correlazioni sono di bassa rilevanza e l'unica più importante, da un punto di vista numerico,
# è tra un regressore e la variabile dipendente ed è quindi benefica allo scopo dell'apprendimento statistico

# --- Calcolo e valutazione della Multicollinearità ---
cat("\n----------------- A. MULTICOLLINEARITY (VIF) ----------------\n\n")
# Multicollinearità: alta correlazione tra diverse variabili indipendenti
# install.packages("car") # si scarica la libreria per la collinearità
library(car) # si aggiunge suddetta libreria all'ambiente di lavoro

fit = lm(Y~., data=dataset)
vif = vif(fit) # misura la multicollinearità, non valuta le variabili dipendenti nel calcolo
#Sono tutte al di sotto del 3 e dunque non è un problema
vif
# CODICE PER CALCOLARE VALORE MAX, MEDIO E IL NUMERO DI VALORI SUPERIORI DI 5 e 10 NELLA COLLINEARITA'
# calcoliamo massimo minimo e medio VIF
x_max_vif <- which.max(vif)  # nella variabile x_max_vif restituisce direttamente l'indice del regressore a cui si trova la max collinearità
vif_max   <- vif[x_max_vif]  # restituisce, nella variabile vif_max, il valore associato all'indice x_max_vif nel vettore vif

x_min_vif <- which.min(vif)   # nella variabile x_min_vif restituisce direttamente l'indice del regressore a cui si trova la min collinearità
vif_min   <- vif[x_min_vif]   # restituisce, nella variabile vif_min, il valore associato all'indice x_min_vif nel vettore vif

vif_mean  <- mean(vif) # nella variabile vif_mean salvo il valore medio del vettore vif

# valori soglia di VIF 
vif_maggiori_5 <- vif[vif > 5] # vif_maggiori_5 è un array che contiene tutti gli elementi di vif maggiori di 5
cat("Max VIF:", round(vif_max, 2), " →", names(vif_max), "\n")
cat("Min VIF:", round(vif_min, 2), " →", names(vif_min), "\n")
cat("Mean VIF:", round(vif_mean, 2), "\n")
cat("Number of variables with VIF > 5:", length(vif_maggiori_5), "\n")
#Non sono presenti valori maggiori di 5, questo insieme all'analisi del valore medio, massimo e minimo
#suggerisce che la multicollinearità non è un problema in questo specifico problema

# ---------------------------------------------------------------- PUNTO B ----------------------------------------------------------------
cat("\n-------------------- B. OLS ESTIMATION ----------------------\n\n")
cat("Estimated coefficients (Beta hats):\n")
# calcolo delle beta cappello
Y <- as.matrix(train_set$Y)
X <- as.matrix(cbind(1, train_set[, !(names(train_set) %in% "Y")]))
B <- solve(t(X) %*% X) %*% (t(X) %*% Y)
B

x = X[,-1]
y = Y

# calcolo del p-value
s2 <- (sum((Y - X %*% B)^2))/(nrow(X) - ncol(X)) # sigma quadro
VCM <- s2*solve(t(X)%*%X)
SE <- sqrt(diag(VCM))
t <- B/SE
pv <- 2*pt(abs(t),nrow(X) - ncol(X), lower.tail = FALSE)
cat("\nP-values of each coefficient:\n")
print(round(pv, 4))

mse.ols = mean((cbind(1, x_test) %*% B - y_test)^2) 
cat("\nMean Squared Error (OLS):", round(mse.ols, 4), "\n")


# ----------------------------------------------------------- PUNTO C ---------------------------------------------------------
cat("\n------------------ C. LINEAR MODEL WITH LM ------------------\n\n")
# Si calcola un modello di regressione lineare - multipla perchè ho più regressori
# con Y variabile dipendente e le altre - i regressori - indicate col . come indipendenti sul dataset chiamato dataset.
fit = lm(Y~., data=train_set)

# names(summary(fit)) comando che dice cosa sta restituendo il summary
#summary(fit) # sono i parametri del modello di regressione multipla che mi vengono restituiti, completi di informazioni aggiuntive quali il p-value
#summary(fit)$coefficients[,4] # il quarto campo della tabella coefficients rappresenta il p-value

# confronto parametri con i valori del punto b
Bfit <- summary(fit)$coefficients[,1]
cat("Coefficients from summary(lm):\n")
print(round(Bfit, 4))
cat("\nDo manual and lm() coefficients match? → ")
print(identical(round(as.vector(Bfit), 7), round(as.vector(B), 7)))

# confronto p-value con i valori del punto b
pvfit <- summary(fit)$coefficients[,4]
cat("\nDo manual and lm() p-values match? → ")
print(identical(round(as.vector(pvfit), 7), round(as.vector(pv), 7)))

mse.ols = mean((cbind(1, x_test) %*% Bfit - y_test)^2) 
cat("\nMSE using lm coefficients:", round(mse.ols, 4), "\n")

# ----------------------------------------------------------- PUNTO D ---------------------------------------------------------
#install.packages("leaps")
library(leaps)
library(boot)

# FORWARD
cat("\n=================== D. FORWARD SELECTION ====================\n\n")
regfit.fwd = regsubsets(Y~., data=train_set, nvmax=30, method="forward")
fwd_summary = summary(regfit.fwd)
#fwd_summary
print("Minimum Cp model:")
fwd_min_cp_index <- which.min(fwd_summary$cp); fwd_min_cp_index
# calcolo mse di test
fwd_cp_coefs <- coef(regfit.fwd, id = fwd_min_cp_index) 
selected_vars.fwd_cp <- names(fwd_cp_coefs)[-1]  # variabili selezionate da forward a minimo Cp
x_test_selected <- x_test[, selected_vars.fwd_cp]
mse.fwd_cp <- mean((y_test - (cbind(1, x_test_selected) %*% fwd_cp_coefs))^2); mse.fwd_cp 
cat("Forward Selection - Best model by Cp:\n")
cat("  → Number of predictors:", fwd_min_cp_index, "\n")
cat("  → Test MSE (Cp):", round(mse.fwd_cp, 4), "\n\n")


print("Minimum bic model:")
fwd_min_bic_index <- which.min(fwd_summary$bic)
# calcolo mse di test
fwd_bic_coefs <- coef(regfit.fwd, id = fwd_min_bic_index) 
selected_vars.fwd_bic <- names(fwd_bic_coefs)[-1]  # variabili selezionate da forward a minimo BIC, nota: sono gli stessi di Cp
x_test_selected <- x_test[, selected_vars.fwd_bic]
mse.fwd_bic <- mean((y_test - (cbind(1, x_test_selected) %*% fwd_bic_coefs))^2); mse.fwd_bic 
cat("Forward Selection - Best model by BIC:\n")
cat("  → Number of predictors:", fwd_min_bic_index, "\n")
cat("  → Test MSE (BIC):", round(mse.fwd_bic, 4), "\n\n")


# there is no predict() method for regsubsets(). 
# Since we will be using this function again, we can capture our steps above and write our own predict method.
predict.regsubsets = function(object,newdata,id,...){ # ... <-> ellipsis
  form=as.formula(object$call[[2]])
  mat=model.matrix(form, newdata)
  coefi=coef(object, id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}

# cross-validation fatta con regsubsets
k=5
folds = sample(1:k,nrow(train_set),replace=TRUE)
cv.errors = matrix(NA,k,30, dimnames=list(NULL, paste(1:30)))
# write a for loop that performs cross-validation
for(j in 1:k){
  best.fit=regsubsets(Y~., data=train_set[folds!=j,], nvmax=30, method="forward")
  #print(summary(best.fit))
  for(i in 1:30){
    pred = predict(best.fit, train_set[folds==j,], id=i)
    cv.errors[j,i] = mean((train_set$Y[folds==j]-pred)^2)
  }
}
# This has given us a 10×19 matrix, of which the (i,j)th element corresponds to the test MSE for the i-th cross-validation fold for the best j-variable model.
#mean.cv.errors=apply(cv.errors, 2, mean); mean.cv.errors# Column average
mean.cv.errors=colMeans(cv.errors) # the same. It can be faster...

par(mfrow=c(1,1))
png("graphs/cv_forward.png", width = 800, height = 600)
plot(mean.cv.errors, type="b")
dev.off()
# We now perform best subset selection on the full data set to obtain the 10-variable model.
reg.best.fwd = regsubsets (Y~., data=train_set, nvmax=30)
# coef(reg.best, 11)

ose = sd(mean.cv.errors) / sqrt(30)
ose_threshold <- min(mean.cv.errors) + ose
optimal_model <- min(which(mean.cv.errors <= ose_threshold))
print("Best model by 5-fold cross-validation:")
optimal_model
# calcolo mse di test
fwd_cv_coefs <-  coef(reg.best.fwd, optimal_model) #it selects the 10-variable model (an 11-var on the textbook)
selected_vars.fwd_cv_coefs <- names(fwd_cv_coefs)[-1]  # variabili selezionate da CV
x_test_selected <- x_test[, selected_vars.fwd_cv_coefs]
mse.fwd_cv <- mean((y_test - (cbind(1, x_test_selected) %*% fwd_cv_coefs))^2); mse.fwd_cv
cat("Forward Selection - Best model by Cross-Validation:\n")
cat("  → Selected model:", optimal_model, "\n")
cat("  → Test MSE (CV):", round(mse.fwd_cv, 4), "\n\n")


# cross-validation fatta con glm
# cv.error <- numeric(30)  # Inizializza un vettore numerico di lunghezza 30
# glm.fit = regfit.fwd
# #summary(glm.fit)
# for (i in 1:30){
#   formula_i <- as.formula(paste("Y ~", paste(names(coef(glm.fit, id = i))[-1], collapse = "+"))) # meglio regsubsets
#   # Adatta il modello GLM sul dataset
#   glm_model <- glm(formula_i, data = train_set)
#   cv.error[i]=cv.glm(train_set, glm_model, K=5)$delta[1] # calcoliamo gli errori
# }
# #min(mean.cv.errors)
# ose = sd(cv.error) / sqrt(30)
# ose_threshold <- min(cv.error) + ose
# optimal_model <- min(which(cv.error <= ose_threshold))
# print("Modello scelto per 5-fold cross-validation:")
# optimal_model
# plot(cv.error)
# 
# # calcolo mse di test
# fwd_cv_coefs <- coef(glm.fit, id = optimal_model)
# selected_vars.fwd_cv_coefs <- names(fwd_cv_coefs)[-1]  # variabili selezionate da CV
# x_test_selected <- x_test[, selected_vars.fwd_cv_coefs]
# mse.fwd_cv <- mean((y_test - (cbind(1, x_test_selected) %*% fwd_cv_coefs))^2); mse.fwd_cv


# BACKWARD
cat("\n================== D. BACKWARD SELECTION ====================\n\n")
regfit.bwd = regsubsets(Y~., data=train_set, nvmax=30, method="backward")
bwd_summary = summary(regfit.bwd)
print("Minimum Cp model:")
bwd_min_cp_index <- which.min(bwd_summary$cp); bwd_min_cp_index
# calcolo mse di test
bwd_cp_coefs <- coef(regfit.bwd, id = bwd_min_cp_index)
selected_vars.bwd_cp <- names(bwd_cp_coefs)[-1]  # variabili selezionate da backward a minimo Cp
x_test_selected <- x_test[, selected_vars.bwd_cp]
mse.bwd_cp <- mean((y_test - (cbind(1, x_test_selected) %*% bwd_cp_coefs))^2); mse.bwd_cp  
cat("Backward Selection - Best model by Cp:\n")
cat("  → Number of predictors:", bwd_min_cp_index, "\n")
cat("  → Test MSE (Cp):", round(mse.bwd_cp, 4), "\n\n")

png("graphs/bwd_cp.png", width = 800, height = 600)
plot(regfit.bwd, scale = "Cp")
dev.off()

png("graphs/bwd_bic.png", width = 800, height = 600)
plot(regfit.bwd, scale = "bic")
dev.off()

png("graphs/bwd_adjr2.png", width = 800, height = 600)
plot(regfit.bwd, scale = "adjr2")
dev.off()

print("Minimum bic model:")
bwd_min_bic_index <- which.min(bwd_summary$bic); bwd_min_bic_index
# calcolo mse di test
bwd_bic_coefs <- coef(regfit.bwd, id = bwd_min_bic_index) 
selected_vars.bwd_bic <- names(bwd_bic_coefs)[-1]  # variabili selezionate da backward a minimo BIC
x_test_selected <- x_test[, selected_vars.bwd_bic]
mse.bwd_bic <- mean((y_test - (cbind(1, x_test_selected) %*% bwd_bic_coefs))^2); mse.bwd_bic 
cat("Backward Selection - Best model by BIC:\n")
cat("  → Number of predictors:", bwd_min_bic_index, "\n")
cat("  → Test MSE (BIC):", round(mse.bwd_bic, 4), "\n\n")

# cross-validation fatta con glm
cv.error <- numeric(30)  # Inizializza un vettore numerico di lunghezza 30
glm.fit = regfit.bwd
#summary(glm.fit)
for (i in 1:30){
  formula_i <- as.formula(paste("Y ~", paste(names(coef(glm.fit, id = i))[-1], collapse = "+")))
  # Adatta il modello GLM sul dataset
  glm_model <- glm(formula_i, data = train_set)
  cv.error[i]=cv.glm(train_set, glm_model, K=5)$delta[1] # calcoliamo gli errori
}
#min(mean.cv.errors)
ose = sd(cv.error) / sqrt(30)
ose_threshold <- min(cv.error) + ose
optimal_model <- min(which(cv.error <= ose_threshold))
print("Modello scelto per 5-fold cross-validation:")
optimal_model
png("graphs/bwd_cv_error.png", width = 800, height = 600)
plot(cv.error, type="b", main="CV error - Backward Selection", xlab="Modello", ylab="Errore")
dev.off()
# calcolo mse di test
bwd_cv_coefs <- coef(glm.fit, id = optimal_model) 
selected_vars.bwd_cv <- names(bwd_cv_coefs)[-1]  # variabili selezionate da CV
x_test_selected <- x_test[, selected_vars.bwd_cv]
mse.bwd_cv <- mean((y_test - (cbind(1, x_test_selected) %*% bwd_cv_coefs))^2); mse.bwd_cv
cat("Backward Selection - Best model by Cross-Validation:\n")
cat("  → Selected model:", optimal_model, "\n")
cat("  → Test MSE (CV):", round(mse.bwd_cv, 4), "\n\n")

# MIXED
cat("\n==================== D. MIXED SELECTION =====================\n\n")
regfit.hybrid = regsubsets(Y~., data=train_set, nvmax=30, method="seqrep")
hyb_summary = summary(regfit.hybrid)
print("Minimum Cp model:")
hyb_min_cp_index <- which.min(hyb_summary$cp); hyb_min_cp_index
# calcolo mse di test
hyb_cp_coefs <- coef(regfit.bwd, id = hyb_min_cp_index) 
selected_vars.hyb_cp <- names(hyb_cp_coefs)[-1]  # variabili selezionate da hybrid a minimo Cp
x_test_selected <- x_test[, selected_vars.hyb_cp]
mse.hyb_cp <- mean((y_test - (cbind(1, x_test_selected) %*% hyb_cp_coefs))^2); mse.hyb_cp 
cat("Mixed Selection - Best model by Cp:\n")
cat("  → Number of predictors:", hyb_min_cp_index, "\n")
cat("  → Test MSE (Cp):", round(mse.hyb_cp, 4), "\n\n")
#hyb_summary$cp

print("Minimum bic model:")
hyb_min_bic_index <- which.min(hyb_summary$bic)
# calcolo mse di test
hyb_bic_coefs <- coef(regfit.bwd, id = hyb_min_bic_index) 
selected_vars.hyb_bic <- names(hyb_bic_coefs)[-1]  # variabili selezionate da backward a minimo BIC
x_test_selected <- x_test[, selected_vars.hyb_bic]
mse.hyb_bic <- mean((y_test - (cbind(1, x_test_selected) %*% hyb_bic_coefs))^2); mse.hyb_bic 
cat("Mixed Selection - Best model by BIC:\n")
cat("  → Number of predictors:", hyb_min_bic_index, "\n")
cat("  → Test MSE (BIC):", round(mse.hyb_bic, 4), "\n\n")

selected_vars.hyb_bic
# cross-validation fatta con glm
cv.error <- numeric(30)  # Inizializza un vettore numerico di lunghezza 30
glm.fit = regfit.hybrid
#summary(glm.fit)
for (i in 1:30){
  formula_i <- as.formula(paste("Y ~", paste(names(coef(glm.fit, id = i))[-1], collapse = "+")))
  # Adatta il modello GLM sul dataset
  glm_model <- glm(formula_i, data = train_set)
  
  cv_result <- cv.glm(train_set, glm_model, K=5)
  cv.error[i]=cv_result$delta[1] # calcoliamo gli errori
}
#min(mean.cv.errors)
ose = sd(cv.error) / sqrt(30)
ose_threshold <- min(cv.error) + ose
optimal_model <- min(which(cv.error <= ose_threshold))
print("Modello scelto per 5-fold cross-validation:")
optimal_model
png("graphs/hyb_cv_error.png", width = 800, height = 600)
plot(cv.error, type="b", main="CV error - Mixed Selection", xlab="Modello", ylab="Errore")
dev.off()
# calcolo mse di test
hyb_cv_coefs <- coef(glm.fit, id = optimal_model) 
selected_vars.hyb_cv <- names(hyb_cv_coefs)[-1]  # variabili selezionate da CV
x_test_selected <- x_test[, selected_vars.hyb_cv]
mse.hyb_cv <- mean((y_test - (cbind(1, x_test_selected) %*% hyb_cv_coefs))^2); mse.hyb_cv
cat("Mixed Selection - Best model by Cross-Validation:\n")
cat("  → Selected model:", optimal_model, "\n")
cat("  → Test MSE (CV):", round(mse.hyb_cv, 4), "\n\n")


# RIDGE REGRESSION
cat("\n===================== D. RIDGE REGRESSION =====================\n\n")
library(glmnet)

cv.out=cv.glmnet(x, y, alpha=0)
cv.out$lambda.1se
bestlam <- cv.out$lambda.min
ridge.final = glmnet(x, y, alpha=0, lambda=bestlam)
#summary(ridge.final$beta)

ridge.coef = predict(ridge.final, type="coefficients")
#print(ridge.coef)
selected_vars.ridge = names(ridge.coef)

ridge.pred = predict(ridge.final, s=bestlam, newx=x_test)
mse.ridge <- mean((ridge.pred-y_test)^2); mse.ridge
cat("Ridge Regression Results:\n")
cat("  → Optimal lambda:", round(bestlam, 5), "\n")
cat("  → Test MSE:", round(mse.ridge, 4), "\n\n")

# LASSO
cat("\n====================== D. LASSO REGRESSION ======================\n\n")
library(glmnet)

x = model.matrix(Y~., train_set)[,-1]
y = train_set$Y

cv.out=cv.glmnet(x, y, alpha=1)
cv.out$lambda.1se
bestlam <- cv.out$lambda.min
lasso.final = glmnet(x, y, alpha=1, lambda=bestlam)
#summary(lasso.final$beta)

lasso.coef = predict(lasso.final, type="coefficients")
print(lasso.coef)
selected_vars.lasso = names(lasso.coef)


lasso.pred = predict(lasso.final, s=bestlam, newx=x_test)
mse.lasso <- mean((lasso.pred-y_test)^2); mse.lasso
cat("Lasso Regression Results:\n")
cat("  → Optimal lambda:", round(bestlam, 5), "\n")
cat("  → Test MSE:", round(mse.lasso, 4), "\n\n")

cat("\n====================== CONCLUSIONS ======================\n\n")
# Creo un vettore con i vari MSE di test calcolati
mse_values <- setNames(
  c(mse.bwd_bic, mse.bwd_cp, mse.bwd_cv, mse.fwd_bic, mse.fwd_cp, mse.fwd_cv, 
    mse.hyb_bic, mse.hyb_cp, mse.hyb_cv, mse.lasso, mse.ridge),
  c("bwd_bic", "bwd_cp", "bwd_cv", "fwd_bic", "fwd_cp", "fwd_cv", 
    "hyb_bic", "hyb_cp", "hyb_cv", "lasso", "ridge")
)
model_min_mse <- mse_values[which.min(mse_values)];
cat("\n================== H. FINAL MODEL COMPARISON ==================\n\n")
cat("Test MSE for all models:\n")
print(round(mse_values, 4))

cat("\nBest model based on test MSE:\n")
cat("  →", names(model_min_mse), "with MSE =", round(mse_values[model_min_mse], 4), "\n\n")

model_min_mse <- names(model_min_mse)

# ----------- punto E ----------
cat("================== E. FINAL MODEL DETAILS ==================\n\n")
cat("Significant predictors according to the best model:\n")

if (model_min_mse == "lasso") {
  print("Lasso è la strategia migliore")
  library(glmnet)
  x = model.matrix(Y~., dataset)[,-1]
  y = dataset$Y
  
  cv.out=cv.glmnet(x, y, alpha=1)
  bestlam <- cv.out$lambda.min
  lasso.final = glmnet(x, y, alpha=1, lambda=bestlam)
  
  lasso.coef = predict(lasso.final, type="coefficients")
  print("Ecco i regressori scelti e i rispettivi coefficienti:")
  print(lasso.coef)
} else if (model_min_mse == "ridge") {
  print("Ridge regression è la strategia migliore")
  library(glmnet)
  x = model.matrix(Y~., dataset)[,-1]
  y = dataset$Y
  
  cv.out=cv.glmnet(x, y, alpha=0)
  bestlam <- cv.out$lambda.min
  ridge.final = glmnet(x, y, alpha=0, lambda=bestlam)
  
  ridge.coef = predict(ridge.final, type="coefficients")
  print("Ecco i regressori scelti e i rispettivi coefficienti:")
  print(ridge.coef)
} else {
  library(car) # si aggiunge suddetta libreria all'ambiente di lavoro
  model_min_mse <- paste0("selected_vars.", model_min_mse)
  regressors <- get(model_min_mse)
  formula <- as.formula(paste("Y ~", paste(regressors, collapse = " + ")))
  fit = lm(formula, data=dataset)
  summary(fit)
}