# ==============================================================================
# Pipeline de Pré-processamento e Integração Transcriptômica
# Descrição: Carregamento de matrizes de expressão, estruturação em Seurat v5,
#            gerenciamento estrito de conflitos e preparação de dados para ML.
# ==============================================================================

# 1. Gerenciamento Estrito de Conflitos
library(conflicted)
options(conflicted.policy = "strict") 

library(Seurat)
library(tidyverse)

# Resolver conflitos clássicos entre tidyverse e Seurat
conflict_prefer("filter", "dplyr")
conflict_prefer("select", "dplyr")
conflict_prefer("rename", "dplyr")

message("[-] Iniciando o pipeline de pré-processamento...")

# 2. Carregamento dos dados fictícios (Mock Data)
expression_matrix <- read.csv("/home/nara/projects/GSE236581/Single-cell/metadata/mock_expression.csv", row.names = 1, check.names = FALSE)
metadata_patients <- read.csv("/home/nara/projects/GSE236581/Single-cell/metadata/mock_metadata.csv", row.names = 1)

# Converte para matriz esparsa para otimização de memória (Padrão Single-Cell)
sparse_matrix <- as(as.matrix(expression_matrix), "dgCMatrix")

# 3. Criação do Objeto Seurat v5
# Demonstra a habilidade de instanciar e gerenciar objetos da classe Seurat
seurat_obj <- CreateSeuratObject(counts = sparse_matrix, project = "Melanoma_Mock")
seurat_obj <- AddMetaData(seurat_obj, metadata = metadata_patients)

message("[+] Objeto Seurat v5 instanciado com sucesso:")
print(seurat_obj)

# 4. Processamento Preditivo e Integração via Tidyverse
# Para evitar bugs de exclusão de assays padrão do Seurat com dados simulados,
# extraímos os dados de expressão e consolidamos usando pipelines robustos de R.
message("[-] Transpondo e preparando matriz de expressão para integração...")
expression_tbl <- as.data.frame(t(as.matrix(sparse_matrix))) %>%
  rownames_to_column(var = "Patient_ID")

# 5. Mesclagem com Metadados Clínicos (Garantindo consistência do ID e da Target)
final_ml_dataset <- metadata_patients %>%
  rownames_to_column(var = "Patient_ID") %>%
  select(Patient_ID, Response, Cohort) %>%
  inner_join(expression_tbl, by = "Patient_ID")

# 6. Salvando o Dataset Final para o Pipeline de Machine Learning
write.csv(final_ml_dataset, "processed_expression_for_ml.csv", row.names = FALSE)
message("[SUCCESS] Arquivo 'processed_expression_for_ml.csv' gerado sem erros!")
