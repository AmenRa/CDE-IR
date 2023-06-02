#!/bin/bash




NAME=shared_encoder_absolute
BS=32
LR=2e-5
PET=absolute

python -m pipeline.training model.name=$NAME dataloader.batch_size=$BS training.learning_rate=$LR model.config.position_embedding_type=$PET
python -m pipeline.reranking_runs model.name=$NAME dataloader.batch_size=$BS training.learning_rate=$LR model.params.position_embedding_type=$PET

# absolute
# relative_key
# relative_key_query
# none