dlp submit -a cjwu4 \
                -n stb16-s-np \
                -d "simple_tree_v8_2" \
                -i reg.deeplearning.cn/dlaas/pytorch_gpu:1.10_cuda9.2-cudnn7-py3.6_mygao2 \
                -e train_2.sh \
                --useGpu \
                -g 1 \
                -l ./log_tree_002.log \
                -k TeslaV100-PCIE-12GB \
                -t PtJob
