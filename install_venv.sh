
#!/bin/bash
virtualenv -p python ./venv
source ./venv/bin/activate && python setup.py install