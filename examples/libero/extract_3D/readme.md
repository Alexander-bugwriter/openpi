
把html文件和点云文件放在一个目录下
用python -m http.server 8000 然后访问http://localhost:8000/pointcloud_viewer.html就能看点云可视化
libero_h5需要在lerobot环境下运行 把h5df转化成lerobot格式
regenerate脚本需要在libero环境下运行 过滤原始h5df数据并且生成3D点云数据

你不要帮我生成完整代码,请你生成需要修改部分的代码就行了,给我的时候给出一部分上下文方便我定位。我希望regenerate脚本在保存voxel的时候能用task.name_i来命名。比如说 libero_10_task_name_第i次重放。然后和regenerate脚本里生成的hdf5文件里的每次重放能够一一对应。保存的时候在hdf5输出的隔壁目录里保存voxel文件。在libero_h5转化的时候能够生成一个voxel_mapping 映射voxel的文件名和对应的lerobot格式的episode id。

python examples/libero/extract_3D/regenerate_libero_dataset.py  --libero_task_suite libero_90 --libero_raw_data_dir /opt/liblibai-models/user-workspace2/dataset/libero_h5/libero_90 --libero_target_dir /opt/liblibai-models/user-workspace2/dataset/lyh_libero_3D/libero_90_no_noops

python examples/libero/extract_3D/regenerate_libero_dataset.py  --libero_task_suite libero_10 --libero_raw_data_dir :~/PycharmProjects/libero_dataset/libero_100/libero_10 --libero_target_dir /media/lyh/Seagate Backup Plus Drive/newstart-university/lab/vla/服务器项目/libero_3D/libero_10_no_noops
python examples/libero/extract_3D/regenerate_libero_dataset.py \
  --libero_task_suite libero_10 \
  --libero_raw_data_dir ~/PycharmProjects/libero_dataset/libero_100/libero_10 \
  --libero_target_dir "/media/lyh/Seagate Backup Plus Drive/newstart-university/lab/vla/服务器项目/libero_3D/libero_10_no_noops_test"


转换脚本
export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE
python examples/libero/extract_3D/libero_h5.py \
--src-paths "/media/lyh/Seagate Backup Plus Drive/newstart-university/lab/vla/服务器项目/libero_3D/libero_10_no_noops" \ 
--output-path "/media/lyh/Seagate Backup Plus Drive/newstart-university/lab/vla/服务器项目/libero_3D/libero_10_3D" \
--executor local \
--tasks-per-job 2 \
--workers 5
python examples/libero/extract_3D/libero_h5.py --src-paths "/media/lyh/Seagate Backup Plus Drive/newstart-university/lab/vla/服务器项目/libero_3D/libero_10_no_noops" --output-path "/media/lyh/Seagate Backup Plus Drive/newstart-university/lab/vla/服务器项目/libero_3D/libero_10_3D" --executor local --tasks-per-job 2 --workers 5
