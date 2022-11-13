export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="../../outputs/iconresults"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=3434554 \
  --resolution=512 \
  --train_batch_size=1 \
  #--train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \ # slow down learning rate
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=1497 \
  --sample_batch_size=4 \
  --max_train_steps=40000 \
  --save_interval=250 \
  --save_sample_prompt="black and white hyena icon" \
  --concepts_list="concepts_list.json"
