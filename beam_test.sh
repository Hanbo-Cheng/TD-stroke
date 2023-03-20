export CUDA_VISIBLE_DEVICES=2
python -u ./beam_test_simple_tree.py \
    10 \
    /yrfs2/cv9/cjwu4/007_OffHME/data_for_simple_tree/ \
    ./beam_results/ \
    ./results001/WAP_params.pkl,./results002/WAP_params.pkl,./results003/WAP_params.pkl \
    valid