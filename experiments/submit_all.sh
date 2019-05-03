#!/bin/bash

# Prepares the BirdCLEF 2018 submissions defined at the bottom.

submit() {
  name="$1"  # run name
  echo "$name:"
  echo "validation set results:"
  ./eval.py --dataset=birdclef --labelfile-background=bg.tsv "${@:2}"
  mono_args=()
  scape_args=()
  for part in "${@:2}"; do
    if [ "${part:0:2}" == "--" ]; then
      args+=("$part")
      continue
    fi
    predfile="${part%:*}"
    predfile="${predfile#birdclef/}"
    modelfile="${predfile%.pred.pkl}.npz"
    # monodirectional predictions
    if [ ! -f birdclef/submissions/mono/parts/"$predfile" ]; then
      echo "computing birdclef/submissions/mono/parts/$predfile..."
      mkdir -p birdclef/submissions/mono/parts
      if [ "${modelfile:0:4}" == "meta" ]; then
        ./predict.py --dataset=birdclef --var arch.output=linear \
          --filelists=test \
          "birdclef/$modelfile" "birdclef/submissions/mono/parts/$predfile"
      else
        ./predict.py --dataset=birdclef --var arch.output=linear \
          --filelists=test \
          --var arch.output_bg= --var len_min=3 --var len_max=63 --split-pool \
          "birdclef/$modelfile" "birdclef/submissions/mono/parts/$predfile"
      fi
    fi
    mono_args+=("birdclef/submissions/mono/parts/${part#birdclef/}")
    # soundscape predictions
    if [ ! -f birdclef/submissions/soundscapes/parts/"$predfile" ]; then
      echo "computing birdclef/submissions/soundscapes/parts/$predfile..."
      mkdir -p birdclef/submissions/soundscapes/parts
      if [ "${modelfile:0:4}" == "meta" ]; then
        ./predict.py --dataset=birdclef --var arch.output=linear \
          --filelists=soundscapes_val,soundscapes_test \
          "birdclef/$modelfile" "birdclef/submissions/soundscapes/parts/$predfile"
      else
        ./predict.py --dataset=birdclef --var arch.output=linear \
          --filelists=soundscapes_val,soundscapes_test \
          --var arch.output_bg= --var len_min=0 --var len_max=5 \
          "birdclef/$modelfile" "birdclef/submissions/soundscapes/parts/$predfile"
      fi
    fi
    scape_args+=("birdclef/submissions/soundscapes/parts/${part#birdclef/}")
  done
  # create monodirectional submission file
  ./eval.py --dataset=birdclef --filelist=test \
    --save-predictions="birdclef/submissions/mono/$name.pkl" "${mono_args[@]}"
  ./submit_csv.py --dataset=birdclef --filelist=test --softmax --mode=mono \
    "birdclef/submissions/mono/$name."{pkl,csv}
  # create soundscape submission file (validation set)
  ./eval.py --dataset=birdclef --filelist=soundscapes_val --no-maxpool \
    --save-predictions="birdclef/submissions/soundscapes/$name.val.pkl" "${scape_args[@]}"
  python ./submit_csv.py --dataset=birdclef --filelist=soundscapes_val --softmax --mode=soundscape \
    "birdclef/submissions/soundscapes/$name.val."{pkl,csv}
  # create soundscape submission file (test set)
  ./eval.py --dataset=birdclef --filelist=soundscapes_test --no-maxpool \
    --save-predictions="birdclef/submissions/soundscapes/$name.pkl" "${scape_args[@]}"
  python ./submit_csv.py --dataset=birdclef --filelist=soundscapes_test --softmax --mode=soundscape \
    "birdclef/submissions/soundscapes/$name."{pkl,csv}
}



# Run 1: audio-only ensemble
# ./blender.py --dataset=birdclef --labelfile-background=bg.tsv --strategy=hyperopt_zero \
#  birdclef/{dummy,resnet1}_{lme1,att16,att64}_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1{,_mixfgbg}{,_ban1}.pred.pkl
#
# MAP: 0.740, MAP-bg: 0.665, top-1: 0.67, top-2: 0.75, top-3: 0.79, top-4: 0.81, top-5: 0.83
#submit run1 \
#  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:2 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:8 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:8 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:2 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:10 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:8 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_bs24_mixfgbg.pred.pkl:1 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_bs24_mixfgbg_ban1.pred.pkl:2
# MAP: 0.749, MAP-bg: 0.672, top-1: 0.67, top-2: 0.76, top-3: 0.80, top-4: 0.82, top-5: 0.84
#submit run1 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.815023 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:5.13242 \
#  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.836264 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:3.37462 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:1.09705 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:4.84133 \
#  birdclef/resnet1_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.8625
# MAP: 0.752, MAP-bg: 0.677, top-1: 0.68, top-2: 0.77, top-3: 0.81, top-4: 0.82, top-5: 0.84
#submit run1 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.21436974 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.71880269 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.083408 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.04421017 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.83124645 \
#  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.1619095 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.86909047 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.71671056 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.57830556 \
#  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.38605772 \
#  birdclef/resnet1_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.21795217 \
#  birdclef/resnet1_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.34983664
# MAP: 0.752, MAP-bg: 0.679, top-1: 0.68, top-2: 0.76, top-3: 0.80, top-4: 0.82, top-5: 0.84
#submit run1 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.695062 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.982253 \
#  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.0924406 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.683036 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.193518 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.503312 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.846705 \
#  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.672716 \
#  birdclef/resnet1_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.450889
# MAP: 0.754, MAP-bg: 0.677, top-1: 0.68, top-2: 0.77, top-3: 0.80, top-4: 0.82, top-5: 0.84
submit run1 \
  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.0564629 \
  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.0785767 \
  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.720805 \
  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.376062 \
  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.736898 \
  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.448296 \
  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.833018 \
  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.604028 \
  birdclef/resnet1_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.405618

# Run 2: audio + meta ensemble
# ./blender.py --dataset=birdclef --labelfile-background=bg.tsv --strategy=hyperopt_zero \
#   birdclef/{dummy,resnet1}_{lme1,att16,att64}_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1{,_mixfgbg}{,_ban1}.pred.pkl \
#   birdclef/meta-256-512_fdrop0{,_noblur,_lessblur,_moreblur}_dtenc3{,_no{date,elev,loc,time}}{,_mixfgbg}{,_ban1}.pred.pkl \
#   birdclef/meta-256-512_fdrop0_noblur{,_no{date,elev,loc,time}}_mixfgbg.pred.pkl
# MAP: 0.792, MAP-bg: 0.727, top-1: 0.73, top-2: 0.81, top-3: 0.84, top-4: 0.86, top-5: 0.87
#submit run2 \
#  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:4 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg{,_ban1}.pred.pkl:6 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:2 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:10 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:8 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3_mixfgbg{,_ban1}.pred.pkl:.8 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3{,_ban1}.pred.pkl:.8 \
#  birdclef/meta-256-512_fdrop0_noblur_no{date,elev,loc,time}_mixfgbg.pred.pkl
# MAP: 0.803, MAP-bg: 0.736, top-1: 0.74, top-2: 0.82, top-3: 0.85, top-4: 0.87, top-5: 0.88
#submit run2 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:2.52437636e-01 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:8.35424137e-01 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:3.25274649e-01 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:9.27978507e-01 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:3.06096622e-04 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:3.51499081e-03 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:7.86101468e-01 \
#  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:9.41742056e-02 \
#  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:5.17761538e-01 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:6.51270665e-01 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:1.03119716e-01 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:6.16118672e-01 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:4.39044552e-01 \
#  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:5.58097211e-01 \
#  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:9.45345897e-01 \
#  birdclef/resnet1_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:2.32240086e-01 \
#  birdclef/resnet1_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:6.20056660e-01 \
#  birdclef/meta-256-512_fdrop0_dtenc3_mixfgbg.pred.pkl:1.48844546e-01 \
#  birdclef/meta-256-512_fdrop0_dtenc3_nodate_mixfgbg.pred.pkl:2.55949481e-01 \
#  birdclef/meta-256-512_fdrop0_dtenc3_noelev_mixfgbg.pred.pkl:7.86944358e-01 \
#  birdclef/meta-256-512_fdrop0_dtenc3_noloc_mixfgbg.pred.pkl:1.27427991e-01 \
#  birdclef/meta-256-512_fdrop0_dtenc3_notime_mixfgbg.pred.pkl:7.81969905e-01 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3.pred.pkl:1.54946978e-01 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3_ban1.pred.pkl:1.51683278e-01 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3_mixfgbg.pred.pkl:6.81090638e-02 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3_mixfgbg_ban1.pred.pkl:6.56018579e-02
# MAP: 0.807, MAP-bg: 0.740, top-1: 0.74, top-2: 0.83, top-3: 0.86, top-4: 0.87, top-5: 0.89
#submit run2 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.918652 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.48512 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.0763527 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.739055 \
#  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.779954 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.61828 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.757109 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.459323 \
#  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.662834 \
#  birdclef/meta-256-512_fdrop0_dtenc3_nodate_mixfgbg.pred.pkl:0.633108 \
#  birdclef/meta-256-512_fdrop0_dtenc3_noelev_mixfgbg.pred.pkl:0.193415 \
#  birdclef/meta-256-512_fdrop0_dtenc3_notime_mixfgbg.pred.pkl:0.734165 \
#  birdclef/meta-256-512_fdrop0_moreblur_dtenc3_notime_mixfgbg.pred.pkl:0.762432 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3.pred.pkl:0.0795803 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3_mixfgbg_ban1.pred.pkl:0.123712
# MAP: 0.809, MAP-bg: 0.740, top-1: 0.74, top-2: 0.83, top-3: 0.86, top-4: 0.87, top-5: 0.89
submit run2 \
  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.64825 \
  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.784622 \
  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.0703957 \
  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.632461 \
  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.206587 \
  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.93497 \
  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.921497 \
  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.651185 \
  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.419165 \
  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.871152 \
  birdclef/resnet1_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.868286 \
  birdclef/meta-256-512_fdrop0_dtenc3_mixfgbg.pred.pkl:0.0179336 \
  birdclef/meta-256-512_fdrop0_dtenc3_notime_mixfgbg.pred.pkl:0.157289 \
  birdclef/meta-256-512_fdrop0_moreblur_dtenc3_nodate_mixfgbg.pred.pkl:0.80044 \
  birdclef/meta-256-512_fdrop0_moreblur_dtenc3_notime_mixfgbg.pred.pkl:0.999421 \
  birdclef/meta-256-512_fdrop0_noblur_mixfgbg.pred.pkl:0.299079 \
  birdclef/meta-256-512_fdrop0_noblur_noelev_mixfgbg.pred.pkl:0.217395 \
  birdclef/meta-256-512_fdrop0_noblur_notime_mixfgbg.pred.pkl:0.265046 \
  birdclef/meta-256-512_fdrop0_noblur_dtenc3_mixfgbg.pred.pkl:0.224108
# MAP: 0.811, MAP-bg: 0.740, top-1: 0.75, top-2: 0.83, top-3: 0.86, top-4: 0.87, top-5: 0.89
#submit run2 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.999832 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.191087 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.941585 \
#  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.351059 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.979438 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.652418 \
#  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.301435 \
#  birdclef/resnet1_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.757596 \
#  birdclef/meta-256-512_fdrop0_dtenc3_nodate_mixfgbg.pred.pkl:0.325907 \
#  birdclef/meta-256-512_fdrop0_dtenc3_noelev_mixfgbg.pred.pkl:0.257896 \
#  birdclef/meta-256-512_fdrop0_dtenc3_notime_mixfgbg.pred.pkl:0.901129 \
#  birdclef/meta-256-512_fdrop0_moreblur_dtenc3_nodate_mixfgbg.pred.pkl:0.928907 \
#  birdclef/meta-256-512_fdrop0_moreblur_dtenc3_notime_mixfgbg.pred.pkl:0.393657 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3_mixfgbg.pred.pkl:0.0595626
# We're still taking the 0.809 model because it seems more stable. Comparing the
# .807, .809 and .811 models on 50 random subsets of 10% of the validation set,
# the .809 model wins against the other two in over half of the cases, both in
# terms of map-fg, map-bg and map-fg+map-bg.

# Run 3: audio + meta ensemble with postfilter
# MAP: 0.818, MAP-bg: 0.736, top-1: 0.75, top-2: 0.83, top-3: 0.87, top-4: 0.89, top-5: 0.90
#submit run3 \
#  --birdclef-filter \
#  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:4 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg{,_ban1}.pred.pkl:6 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:2 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:10 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:8 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3_mixfgbg{,_ban1}.pred.pkl:.8 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3{,_ban1}.pred.pkl:.8 \
#  birdclef/meta-256-512_fdrop0_noblur_no{date,elev,loc,time}_mixfgbg.pred.pkl
# MAP: 0.832, MAP-bg: 0.748, top-1: 0.77, top-2: 0.85, top-3: 0.88, top-4: 0.90, top-5: 0.91
#submit run3 \
#  --birdclef-filter \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.918652 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.48512 \
#  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.0763527 \
#  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.739055 \
#  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.779954 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.61828 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.757109 \
#  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.459323 \
#  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.662834 \
#  birdclef/meta-256-512_fdrop0_dtenc3_nodate_mixfgbg.pred.pkl:0.633108 \
#  birdclef/meta-256-512_fdrop0_dtenc3_noelev_mixfgbg.pred.pkl:0.193415 \
#  birdclef/meta-256-512_fdrop0_dtenc3_notime_mixfgbg.pred.pkl:0.734165 \
#  birdclef/meta-256-512_fdrop0_moreblur_dtenc3_notime_mixfgbg.pred.pkl:0.762432 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3.pred.pkl:0.0795803 \
#  birdclef/meta-256-512_fdrop0_noblur_dtenc3_mixfgbg_ban1.pred.pkl:0.123712
submit run3 \
  --birdclef-filter \
  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.64825 \
  birdclef/dummy_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.784622 \
  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.0703957 \
  birdclef/dummy_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg_ban1.pred.pkl:0.632461 \
  birdclef/dummy_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.206587 \
  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.93497 \
  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.921497 \
  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.651185 \
  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1.pred.pkl:0.419165 \
  birdclef/resnet1_att16_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_mixfgbg.pred.pkl:0.871152 \
  birdclef/resnet1_att64_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_ban1.pred.pkl:0.868286 \
  birdclef/meta-256-512_fdrop0_dtenc3_mixfgbg.pred.pkl:0.0179336 \
  birdclef/meta-256-512_fdrop0_dtenc3_notime_mixfgbg.pred.pkl:0.157289 \
  birdclef/meta-256-512_fdrop0_moreblur_dtenc3_nodate_mixfgbg.pred.pkl:0.80044 \
  birdclef/meta-256-512_fdrop0_moreblur_dtenc3_notime_mixfgbg.pred.pkl:0.999421 \
  birdclef/meta-256-512_fdrop0_noblur_mixfgbg.pred.pkl:0.299079 \
  birdclef/meta-256-512_fdrop0_noblur_noelev_mixfgbg.pred.pkl:0.217395 \
  birdclef/meta-256-512_fdrop0_noblur_notime_mixfgbg.pred.pkl:0.265046 \
  birdclef/meta-256-512_fdrop0_noblur_dtenc3_mixfgbg.pred.pkl:0.224108

# Run 4: single best audio + meta model
# MAP: 0.768, MAP-bg: 0.698, top-1: 0.70, top-2: 0.78, top-3: 0.82, top-4: 0.84, top-5: 0.85
submit run4 \
  birdclef/resnet1_lme1_fdrop05_fM10k_powlearn_shift5_fs1024_mc2cgr1dgr1_meta-256-512_fdrop0_dtenc3_pretrained.pred.pkl
