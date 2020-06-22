# fp32 values, int64 indices, warmup coeff: [0.25, 0.063, 0.015, 0.004, 0.001] -> 0.001
mpirun -np ${1} -H ${2} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo python train.py --configs configs/cifar/resnet20.py configs/dgc/wm5.py

# fp32 values, int64 indices, wamup coeff: [1, 1, 1, 1, 1] -> 0.001
mpirun -np ${1} -H ${2} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo python train.py --configs configs/cifar/resnet20.py configs/dgc/wm5m.py

# fp16 values, int32 indices, warmup coeff: [0.25, 0.063, 0.015, 0.004, 0.001] -> 0.001
mpirun -np ${1} -H ${2} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo python train.py --configs configs/cifar/resnet20.py configs/dgc/wm5.py configs/dgc/fp16.py configs/dgc/int32.py

# fp16 values, int32 indices, wamup coeff: [1, 1, 1, 1, 1] -> 0.001
mpirun -np ${1} -H ${2} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo python train.py --configs configs/cifar/resnet20.py configs/dgc/wm5m.py configs/dgc/fp16.py configs/dgc/int32.py
