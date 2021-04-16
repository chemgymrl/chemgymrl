[chemgymrl.com](https://chemgymrl.com/)

## Installation

In this tutorial we will be going over how to install the chemgymrl library. For those of you who have used pyhton and
opensource libraries in the past before this process will be very familiar to you, if you haven't, no worries we have 
created a [video](https://youtu.be/tE8aVln64_0) which will be of some help, you will find the same instructions here.


<div style="text-align: center; margin-bottom: 2em;">
<iframe width="560" height="315" src="https://youtu.be/tE8aVln64_0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

### Clone Repository

The first step of the install process is to clone the repository, that can be done using the following command line
instructions:
```commandline
cd path/to/desired/install/location
git clone https://github.com/chemgymrl/chemgymrl.git
```
now that the repo has been installed we need to enter into the repository:

```commandline
cd chemgymrl
```

### Python Virtual Environment

ChemGymRL is set to use python 3, and more specifically python 3.8. The first step of this next part is to install
[python](https://python.org), if you already have python then the next step is to create a virtual environment using
your favourite virtual environment tool. In this tutorial we will use virtualenv, but anaconda works as well. The next
steps will show you how to create and activate the correct virtual environment:

```commandline
python3.8 -m venv chemgymrl
source chemgymrl/bin/activate
```

Now that the virtual environment is created and activated we will now look to install all the correct packages.

### Install Library
Now that everything is set up we simply seed to install the library.
 
```commandline
pip install .
```

### Test The Installation

Now we simply need to test that the installation of the library is complete. To do so we simply need to run the
following:

```commandline
cd tests/unit
python3.8 -m unittest discover -p "*test*.py"
```

After you run this, you should get a line that says that all tests passed, if you get any errors please look at the
troubleshooting page. Thank you for using ChemGymRL.
