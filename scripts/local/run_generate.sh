
CUDA_VISIBLE_DEVICES="$4" python src/eval/generate.py \
         --model-path "$1" \
         --input-path "$2" \
         --output-path "$3"
