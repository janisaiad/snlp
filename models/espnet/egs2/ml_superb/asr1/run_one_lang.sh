#!/usr/bin/env bash
# Run ML-SUPERB for a single (language, duration). Use this to reproduce incrementally.
# Usage:
#   ./run_one_lang.sh --single_lang eng1 --duration 10min
#   ./run_one_lang.sh --single_lang fra1 --duration 10min --asr_config conf/tuning/train_asr_s3prl_10min.yaml
# Requires: MLSUPERB set (or db.sh default), and data prepared under $MLSUPERB.
set -e
set -u
set -o pipefail

stage=1
stop_stage=13
nj=4
inference_nj=2
gpu_inference=false
expdir=exp

single_lang=eng1
duration=10min
asr_config=conf/tuning/train_asr_fbank_single.yaml
inference_config=conf/decode_asr.yaml
asr_args=

. utils/parse_options.sh || exit 1

train_set=train_${duration}_${single_lang}
train_dev=dev_10min_${single_lang}
test_set="${train_dev} test_10min_${single_lang}"
asr_tag="$(basename "${asr_config}" .yaml)_${single_lang}_${duration}"

if [ "${single_lang}" == "cmn" ] || [ "${single_lang}" == "jpn" ]; then
  token_type=word
else
  token_type=char
fi

local_data_opts="--duration ${duration} --lid false --multilingual false --single_lang ${single_lang}"

echo "Running ML-SUPERB mono: lang=${single_lang} duration=${duration} asr_config=${asr_config} (stage ${stage}..${stop_stage})"

_extra=()
[ -n "${asr_args}" ] && _extra+=(--asr_args "${asr_args}")

./asr.sh \
  --ngpu 1 \
  --stage ${stage} \
  --stop_stage ${stop_stage} \
  --nj ${nj} \
  --inference_nj ${inference_nj} \
  --gpu_inference ${gpu_inference} \
  --lang ${single_lang} \
  --inference_asr_model "valid.loss.ave.pth" \
  --local_data_opts "${local_data_opts}" \
  --use_lm false \
  --token_type ${token_type} \
  --feats_type raw \
  --feats_normalize utterance_mvn \
  --asr_config "${asr_config}" \
  --inference_config "${inference_config}" \
  --train_set "${train_set}" \
  --valid_set "${train_dev}" \
  --test_sets "${test_set}" \
  --asr_tag "${asr_tag}" \
  --expdir ${expdir} \
  --asr_stats_dir ${expdir}/asr_stats_${single_lang}_${duration} \
  --local_score_opts "false false monolingual" \
  "${_extra[@]}"

echo "Done. Check ${expdir}/${asr_tag}/ and decode logs for CER."
