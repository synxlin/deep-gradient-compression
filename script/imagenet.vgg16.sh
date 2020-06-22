# fp32 values, int64 indices, no warmup
mpirun -np ${1} -H ${2} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo python train.py --configs configs/imagenet/vgg16_bn.py configs/dgc/wm0.py

# fp16 values, int32 indices, no warmup
mpirun -np ${1} -H ${2} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo python train.py --configs configs/imagenet/vgg16_bn.py configs/dgc/wm0.py configs/dgc/fp16.py configs/dgc/int32.py
