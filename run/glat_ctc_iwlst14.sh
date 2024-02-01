#export CUDA_VISIBLE_DEVICES=1
#export CUDA_LAUNCH_BLOCKING=1
# glat_context_p = 0.5 

data_dir=data-bin/distill_iwslt14.tokenized.de-en
save_path=model/iwslt14_ctc
log=${save_path}/train.log
src=en
tgt=de
layers=5
dim=256
update=250000
max_token=8192
#线性退火
lr_scheduler=anneal
python3 train.py ${data_dir} --arch glat_ctc --noise full_mask --share-all-embeddings \
    --criterion ctc_loss --label-smoothing 0.1 --lr 5e-4  --stop-min-lr 1e-9 \
    --lr-scheduler ${lr_scheduler}  --init-lr 2.2e-4 --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task translation_lev_modified --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers ${layers} --encoder-embed-dim ${dim} --decoder-layers ${layers} --decoder-embed-dim ${dim} --fp16 \
    --max-source-positions 1000 --max-target-positions 1000 --max-update ${update} --seed 0 --clip-norm 2\
    --save-dir ${save_path} --length-loss-factor 0 --log-interval 100 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir glat_plugins \
    --no-progress-bar \
    --keep-last-epochs 3 \
    --keep-best-checkpoints 5 | tee -a ${log}