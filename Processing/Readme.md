# Processing

In this directly we'll collect all the scripts and steps to setup a minikube cluster, and iteratively process the dataset.

## Step 1 - Setting up minikube

### 1. install Docker

If you do not already have Docker installed, then set that up. (if you are running in WSL, then install Docker using Docker desktop, and enable WSL 2 - don't install natively from Ubuntu command line)

#### WSL

If using WSL, then first check your version

```powershell
wsl -l -v
```

You want to be running version 2, if not then run:

```powershell
wsl --set-version Ubuntu 2
```

To get this to work I had to:
* update some kernels (from a link in the failed attempt message)
* enable the Virtual Machine Platform (from: turn windows features on or off)

Once you've configured Docker, run:

```bash
docker --version
```
To check that is has been installed correctly. You can test it further with:

```bash
docker run hello-world
```

### 2. install minikube

Follow the instructions on their website, depends on CPU architecture, for me it was:

```bash
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
```

(on WSL)







