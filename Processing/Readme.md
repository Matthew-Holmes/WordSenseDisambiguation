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

To check that is has been installed correctly





