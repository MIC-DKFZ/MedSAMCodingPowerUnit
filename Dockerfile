FROM pytorch/pytorch:latest

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
# required by cv2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir -p /opt/app /inputs /outputs /model \
    && chown user:user /opt/app /inputs /outputs /model

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY --chown=user:user . .
RUN python -m pip install --user -e .
COPY --chown=user:user _model_for_container .

# ENTRYPOINT ["/bin/bash"]
# build via: docker build . -t cvpr:your_tag