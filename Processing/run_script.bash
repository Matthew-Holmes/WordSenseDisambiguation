
python script.py \
    --input-csv ../Data/Processed/SemCoreChunks/chunk_0.csv \
    --output-csv ../Data/Processed/ProcessedSemCoreChunks/chunk_0.csv \
    --chunk-store ../Data/Processed/jax_store/chunk_0 \
    --tokenizer /home/matt/.llama/checkpoints/Llama3.2-1B-hf-tok/tokenizer.model \
    --model-weights ../Data/ModelWeights/llama_jax_weights.pkl
