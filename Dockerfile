FROM python:3.7

ARG PIP_NO_CACHE_DIR=True

# install build dependencies
RUN apt-get update && apt-get install -y g++

WORKDIR /app

RUN pip install --upgrade pip

COPY ./ .

RUN pip install .

CMD python -c "import mrptlib; print('MRPT has been installed')"
