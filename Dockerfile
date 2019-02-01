FROM ubuntu
#nvcr.io/nvidia/tensorflow:18.11-py3

# Install GIT
RUN apt-get update && apt-get install git -y

# TODO remove
RUN apt-get update && apt-get install python3 python3-pip -y

# Clone Code
RUN git clone https://github.com/perara/deep-logistics.git /code

# Set code to default workdir
WORKDIR /code

RUN pip3 install -r requirements.txt

# TODO -
# 1. Add deep-logistics-ml to github
# 2. Create Setup.py
# Try to run this on cair... First try on pc home via ssh?
ENTRYPOINT ["/code/"]
