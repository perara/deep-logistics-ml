FROM nvcr.io/uiatekreal/cair

# Install GIT
RUN apt-get update && apt-get install git libgtk2.0-dev libsm6 libxext6 xvfb -y

# Upgrade PIP
RUN python3 -m pip install --upgrade pip

WORKDIR /root/

# Clone Code
RUN git clone https://github.com/perara/deep-logistics-ml.git deep-logistics-ml-git
RUN git clone https://github.com/perara/deep-logistics.git deep-logistics-git

# Install dependencies
RUN pip3 install -r deep-logistics-git/requirements.txt
RUN pip3 install -r deep-logistics-ml-git/requirements.txt

RUN chmod +x /root/deep-logistics-ml-git/runner.sh

CMD ["/root/deep-logistics-ml-git/runner.sh"]



