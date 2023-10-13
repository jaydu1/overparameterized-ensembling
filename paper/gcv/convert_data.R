Sys.setenv("OMP_NUM_THREADS" = 4)
Sys.setenv("OPENBLAS_NUM_THREADS" = 4)
Sys.setenv("MKL_NUM_THREADS" = 6)
Sys.setenv("VECLIB_MAXIMUM_THREADS" = 4)
Sys.setenv("NUMEXPR_NUM_THREADS" = 6)

library(Seurat)

# https://gist.github.com/stephenturner/6046337d1237ecd627a8
# https://github.com/hhoeflin/hdf5r/issues/94
# Sys.setenv('LD_LIBRARY_PATH'='/home/jinhongd/hdf5-1.10.5/lib:$LD_LIBRARY_PATH')
# install.packages("hdf5r", configure.args="--with-hdf5=/home/jinhongd/hdf5-1.10.5/bin/h5cc")
# remotes::install_github("mojaveazure/seurat-disk")
dyn.load('/home/jinhongd/hdf5-1.10.5/lib/libhdf5_hl.so.100')
library(SeuratDisk)

library(ggplot2)
library(cowplot)

library(dplyr)


pbmc_multimodal <- LoadH5Seurat("pbmc_multimodal.h5seurat")
pbmc_multimodal <- pbmc_multimodal[,!pbmc_multimodal@meta.data$celltype.l1 %in% c('other', 'other T')]
pbmc_multimodal <- pbmc_multimodal[,pbmc_multimodal@meta.data$time==3]


table(pbmc_multimodal@meta.data$celltype.l1)
#     B CD4 T CD8 T    DC  Mono    NK 
# 4558 15728  8849  1031 14312  6303 
celltypes <- pbmc_multimodal@meta.data$celltype.l1
RNA <- pbmc_multimodal@assays$SCT@counts[VariableFeatures(pbmc_multimodal),]
# RNA <- RNA[rowSums(RNA>0)>=500,]

pbmc_multimodal <- FindVariableFeatures(pbmc_multimodal, assay='ADT', nfeatures = 50)
VariableFeatures(pbmc_multimodal@assays$ADT)
ADT <- pbmc_multimodal@assays$ADT@counts[VariableFeatures(pbmc_multimodal@assays$ADT),]
# ADT <- ADT[rowSums(ADT>0)>=500,]

library(hdf5r)
file.h5 <- H5File$new("pbmc_count.h5", mode = "w")
file.h5[["ADT"]] <- as.matrix(ADT)
file.h5[["RNA.shape"]] <- RNA@Dim
file.h5[["RNA.data"]] <- RNA@x
file.h5[["RNA.indices"]] <- RNA@i
file.h5[["RNA.indptr"]] <- RNA@p
file.h5[["cell_ids"]] <- colnames(RNA)
file.h5[["gene_names"]] <- rownames(RNA)
file.h5[["ADT_names"]] <- rownames(ADT)
file.h5[["celltype"]] <- pbmc_multimodal@meta.data$celltype.l1
file.h5$close_all()