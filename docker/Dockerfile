ARG L4T_TAG=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ROS_DISTRO=humble \
    PROJ=/mnt/nvme/emanuele_project

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      locales curl gnupg2 lsb-release ca-certificates \
      software-properties-common \
 && locale-gen en_US en_US.UTF-8 \
 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
 && add-apt-repository -y universe \
 && (apt-get install -y --only-upgrade systemd udev libsystemd0 libudev1 || true) \
 && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}') \
 && curl -fL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" \
 && dpkg -i /tmp/ros2-apt-source.deb \
 && rm /tmp/ros2-apt-source.deb

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ros-humble-ros-base \
      ros-humble-angles \
      ros-humble-tf2-geometry-msgs \
      ros-humble-realsense2-camera \
      ros-humble-realsense2-description \
      python3-pip \
      python3-colcon-common-extensions \
      python3-rosdep \
      git nano wget sudo \
 && rm -rf /var/lib/apt/lists/* \
 && (rosdep init || true)

RUN apt-get update \
 && (apt-get install -y --no-install-recommends \
      ros-humble-image-transport \
      ros-humble-image-transport-plugins \
      ros-humble-web-video-server \
    || echo "ATTENZIONE: pacchetti opzionali di debug non installati") \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && (apt-get install -y --no-install-recommends \
      ros-humble-usb-cam \
      ros-humble-v4l2-camera \
      v4l-utils \
    || echo "ATTENZIONE: driver camera non installati") \
 && rm -rf /var/lib/apt/lists/*

ARG ORT_BASE=https://github.com/ultralytics/assets/releases/download/v0.0.0
ARG ORT_WHL=onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
RUN pip3 install --no-cache-dir ${ORT_BASE}/${ORT_WHL} \
 && pip3 install --no-cache-dir "numpy==1.24.4" "opencv-python-headless==4.11.0.86"

RUN bash -c "source /opt/ros/humble/setup.bash && python3 -c 'import rclpy, numpy, cv2, onnxruntime as o; print(numpy.__version__, cv2.__version__, o.__version__, o.get_available_providers())'"

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN (getent passwd ${USER_UID} | cut -d: -f1 | xargs -r userdel -r || true) \
 && (getent group ${USER_GID} | cut -d: -f1 | xargs -r groupdel || true) \
 && groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
 && usermod -aG video,dialout,sudo ${USERNAME} \
 && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME}

RUN echo "source /opt/ros/humble/setup.bash" >> /home/${USERNAME}/.bashrc \
 && echo "[ -f ${PROJ}/ros2_ws_emanuele/install/setup.bash ] && source ${PROJ}/ros2_ws_emanuele/install/setup.bash" >> /home/${USERNAME}/.bashrc

USER ${USERNAME}
RUN rosdep update --rosdistro humble || true
WORKDIR ${PROJ}
CMD ["bash"]