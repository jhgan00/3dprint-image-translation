FROM nvidia/cuda:11.6.2-base-ubuntu20.04
RUN apt-get update -y
RUN apt-get install -y python3 python3-pip python-is-python3
COPY ./ /workspace/3dprint-image-translation
WORKDIR /workspace/3dprint-image-translation
ENV PATH=/root/.local/bin:$PATH
RUN pip install --user -r requirements.txt
CMD /bin/bash
