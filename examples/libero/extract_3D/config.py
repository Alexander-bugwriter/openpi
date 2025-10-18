# LIBERO_FEATURES = {
#     "observation.images.image": {
#         "dtype": "video",
#         "shape": (256, 256, 3),
#         "names": ["height", "width", "rgb"],
#     },
#     "observation.images.wrist_image": {
#         "dtype": "video",
#         "shape": (256, 256, 3),
#         "names": ["height", "width", "rgb"],
#     },
#     "observation.state": {
#         "dtype": "float32",
#         "shape": (8,),
#         "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"]},
#     },
#     "observation.states.ee_state": {
#         "dtype": "float32",
#         "shape": (6,),
#         "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw"]},
#     },
#     "observation.states.joint_state": {
#         "dtype": "float32",
#         "shape": (7,),
#         "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
#     },
#     "observation.states.gripper_state": {
#         "dtype": "float32",
#         "shape": (2,),
#         "names": {"motors": ["gripper", "gripper"]},
#     },
#     "action": {
#         "dtype": "float32",
#         "shape": (7,),
#         "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
#     },
# }
LIBERO_FEATURES = {
    "image": {
        "dtype": "image",  # 使用 image 而不是 video
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {  # 注意键名是 state，不是 observation.state
        "dtype": "float32",
        "shape": (8,),
        "names": ["state"],  # 或者更详细: ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
    },
    "actions": {  # 注意是复数
        "dtype": "float32",
        "shape": (7,),
        "names": ["actions"],  # 或者更详细: ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    },
}