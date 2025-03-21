library(data.table)
library(ggplot2)
library(ggvenn)
library(RColorBrewer)
library(pheatmap)

#######
# 1. Collapse all sequences to one file all_sequences.fa
# 2. makeblastdb -in all_sequences.fa -dbtype prot -out all_sequences
# 3. blastp -query all_sequences.fa -db all_sequences.fa -outfmt "6 qseqid sseqid evalue bitscore"  -max_hsps 1 -out all_vs_all.tsv
#######
path_to_blast <- '/path/to/blast/'
blast_results <- fread(paste0(path_to_blast, 'all_vs_all.tsv'), sep='\t')
colnames(blast_results) <- c("query_protein", "search_protein", "evalue", "bitscore")


make_degree_table <- function(df) {
  df2 <- copy(df)
  id1 <- df2$Id1
  df2$Id1 <- df2$Id2
  df2$Id2 <- id1
  return(df2)
}

save_pheatmap_png <- function(x, filename, width=8, height=6) {
  png(filename, width = 7200, height = 5400, res=600)
  grid::grid.newpage()
  grid::grid.draw(x$gtable)
  dev.off()
}

for(dataset in c('gold_stand', 'gold_stand_dscript')){
  print(paste0('Processing dataset: ', dataset))
  path_to_gold_stand <- paste0('../', dataset, '/')
  train <- fread(paste0(path_to_gold_stand, dataset, '_train_all_seq.csv'))
  val <- fread(paste0(path_to_gold_stand, dataset, '_val_all_seq.csv'))
  test <- fread(paste0(path_to_gold_stand, dataset, '_test_all_seq.csv'))
  
  # Visualize that the sets are not overlapping
  uniq_train <- union(unique(train$Id1), unique(train$Id2))
  uniq_val <- union(unique(val$Id1), unique(val$Id2))
  uniq_test <- union(unique(test$Id1), unique(test$Id2))
  
  ggvenn(
    data= list(
      Training = uniq_train,
      Validation = uniq_val,
      Testing = uniq_test
    ),
    fill_color = RColorBrewer::brewer.pal(3, 'Set2'),
    stroke_size = 0.5,
    set_name_size = 6
  )
  
  ggsave(
    paste0(dataset, '_venn.pdf'),,
    device="pdf",
    height=7,
    width=7
  )
  
  # Visualize the node degrees
  train2 <- make_degree_table(train)
  all_data <- rbind(train, train2)
  all_data$Set <- c("Training")
  val2 <- make_degree_table(val)
  val2$Set <- c("Validation")
  all_data <- rbind(all_data, val2)
  test2 <- make_degree_table(test)
  test2$Set <- c("Test")
  all_data <- rbind(all_data, test2)
  
  all_degrees <- all_data[, .N, by=c("Id1", "Interact", "Set")]
  all_degrees[, Set := factor(Set, levels=c("Training", "Validation", "Test"))]
  # cast to wide format
  all_degrees <- dcast(all_degrees, Id1 + Set ~ Interact, value.var="N", fill=0)
  colnames(all_degrees) <- c("Protein", "Set", "Negative", "Positive")
  ggplot(all_degrees, aes(x=Positive, y=Negative)) +
    geom_point() +
    geom_abline(intercept=0, slope=1, color="red") +
    facet_wrap(~Set, scales="free") +
    labs(x="Degree in the positive dataset", y="Degree in the negative dataset") +
    theme_minimal()
  ggsave(
    paste0(dataset, '_degrees.pdf'),
    device="pdf",
    height=5,
    width=10
  )
  
  train_bitscores <- blast_results[query_protein %in% uniq_train & search_protein %in% uniq_train]
  train_bitscores <- dcast(train_bitscores, query_protein ~ search_protein, value.var="bitscore")
  train_bitscores <- as.matrix(train_bitscores[, -1])
  print(paste("Mean before removing self-hits and setting NA to 0 train:", mean(train_bitscores, na.rm = T)))
  train_bitscores[is.na(train_bitscores)] <- 0
  diag(train_bitscores) <- 0
  val_bitscores <- blast_results[query_protein %in% uniq_val & search_protein %in% uniq_val]
  val_bitscores <- dcast(val_bitscores, query_protein ~ search_protein, value.var="bitscore")
  val_bitscores <- as.matrix(val_bitscores[, -1])
  print(paste("Mean before removing self-hits and setting NA to 0 val:", mean(val_bitscores, na.rm = T)))
  val_bitscores[is.na(val_bitscores)] <- 0
  diag(val_bitscores) <- 0
  test_bitscores <- blast_results[query_protein %in% uniq_test & search_protein %in% uniq_test]
  test_bitscores <- dcast(test_bitscores, query_protein ~ search_protein, value.var="bitscore")
  test_bitscores <- as.matrix(test_bitscores[, -1])
  print(paste("Mean before removing self-hits and setting NA to 0 test:", mean(test_bitscores, na.rm = T)))
  test_bitscores[is.na(test_bitscores)] <- 0
  diag(test_bitscores) <- 0
  
  print("Bitscores:")
  print(paste(dataset, "mean bitscore training", mean(train_bitscores)))
  print(paste(dataset, "mean bitscore validation", mean(val_bitscores)))
  print(paste(dataset, "mean bitscore test", mean(test_bitscores)))
        
  train_test <- blast_results[(query_protein %in% uniq_train & search_protein %in% uniq_test) | (query_protein %in% uniq_test & search_protein %in% uniq_train)]
  train_test <- dcast(train_test, query_protein ~ search_protein, value.var="bitscore")
  train_test <- as.matrix(train_test[, -1])
  print(paste("Mean before removing self-hits and setting NA to 0 train-test:", mean(train_test, na.rm = T)))
  train_test[is.na(train_test)] <- 0
  train_val <- blast_results[(query_protein %in% uniq_train & search_protein %in% uniq_val) | (query_protein %in% uniq_val & search_protein %in% uniq_train)]
  train_val <- dcast(train_val, query_protein ~ search_protein, value.var="bitscore")
  train_val <- as.matrix(train_val[, -1])
  print(paste("Mean before removing self-hits and setting NA to 0 train-val:", mean(train_val, na.rm = T)))
  train_val[is.na(train_val)] <- 0
  val_test <- blast_results[(query_protein %in% uniq_val & search_protein %in% uniq_test) | (query_protein %in% uniq_test & search_protein %in% uniq_val)]
  val_test <- dcast(val_test, query_protein ~ search_protein, value.var="bitscore")
  val_test <- as.matrix(val_test[, -1])
  print(paste("Mean before removing self-hits and setting NA to 0 val-test:", mean(val_test, na.rm = T)))
  val_test[is.na(val_test)] <- 0
  
  print(paste(dataset, "mean bitscore training-test", mean(train_test)))
  print(paste(dataset, "mean bitscore training-validation", mean(train_val)))
  print(paste(dataset, "mean bitscore validation-test", mean(val_test)))
  
  all_unique_proteins <- union(uniq_train, union(uniq_val, uniq_test))
  blast_results_filtered <- blast_results[(query_protein %in% all_unique_proteins) & (search_protein %in% all_unique_proteins)]
  blast_results_filtered <- dcast(blast_results_filtered, query_protein ~ search_protein, value.var="bitscore")
  blast_results_filtered <- as.data.frame(blast_results_filtered[, -1])
  rownames(blast_results_filtered) <- colnames(blast_results_filtered)
  
  rows_and_columns <- c(uniq_train, uniq_val, uniq_test)
  rows_and_columns <- rows_and_columns[rows_and_columns %in% colnames(blast_results_filtered)]
  blast_results_filtered <- blast_results_filtered[rows_and_columns, rows_and_columns]
  
  blast_results_filtered <- as.matrix(blast_results_filtered)
  diag(blast_results_filtered) <- NA
  rownames(blast_results_filtered) <- rows_and_columns
  colnames(blast_results_filtered) <- rows_and_columns
  
  # remove rows/columns with too many NA values
  if(dataset=="gold_stand"){
    threshold <- 100
  }else{
    threshold <- 80
  }
  filtered_cols <- colnames(blast_results_filtered[, colSums(!is.na(blast_results_filtered)) >= threshold])
  mask <- rows_and_columns %in% filtered_cols
  blast_results_filtered <- blast_results_filtered[filtered_cols, filtered_cols]
  
  uniq_train <- uniq_train[uniq_train %in% filtered_cols]
  uniq_val <- uniq_val[uniq_val %in% filtered_cols]
  uniq_test <- uniq_test[uniq_test %in% filtered_cols]
  anno <- c(rep("Training", length(uniq_train)), rep("Validation", length(uniq_val)), rep("Test", length(uniq_test)))
  annotation_df = data.frame(
    Set = anno
  )
  annotation_df$Set <- factor(annotation_df$Set, levels=c("Training", "Validation", "Test"))
  rownames(annotation_df) <- colnames(blast_results_filtered)
  
  my_heatmap <- pheatmap(
    blast_results_filtered,
    border_color=NA,
    breaks=exp(seq(log(min(blast_results_filtered, na.rm=T)), log(max(blast_results_filtered, na.rm=T)), length.out = 101)),
    na_col="white",
    annotation_col=annotation_df,
    annotation_row=annotation_df,
    cluster_rows=FALSE,
    cluster_cols=FALSE,
    show_rownames=FALSE,
    show_colnames=FALSE,
  )
  
  save_pheatmap_png(my_heatmap, paste0(dataset ,"_heatmap.png"))
}


