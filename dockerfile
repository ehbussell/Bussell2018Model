FROM python:3.6

RUN apt-get update && apt-get install -y \
    ca-certificates \
    cmake \
    g++ \
    gfortran \
    make \
    wget

RUN wget --no-check-certificate https://files.inria.fr/bocop/Bocop-2.0.5-linux-src.tar.gz && \
    gunzip Bocop-2.0.5-linux-src.tar.gz && \
    tar -xvf Bocop-2.0.5-linux-src.tar && \
    rm -rf Bocop-2.0.5-linux-src.tar && \
    mv Bocop-2.0.5-linux-src Bocop-2.0.5 && \
    cd Bocop-2.0.5/examples/goddard && \
    touch bocop && \
    bash build.sh && \
    ./bocop