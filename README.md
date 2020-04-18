To Develop
=================

Install docker from [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
<br>
To run docker without sudo follow the steps [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user)

---------------------------

To build the container run the command
```
docker build -t flask:latest .
```

The build will download and install all the required packages.

To run the docker image
```
docker run -p 5000:5000 flask
``` 

This redirects port 5000 of the docker image to the port 5000 of localhost. Open ```localhost:5000``` in your browser.

------------
To stop the container find ID of current container. It will be the first column of the output from
```
docker ps
```

To stop, run
```
docker stop <container-id>
```
