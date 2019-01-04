PAUSE Make sure to increment version in setup.py. Continue?
python setup.py sdist
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
REM pip install regim --upgrade
REM pip show regim

REM pip install yolk3k
REM yolk -V regim