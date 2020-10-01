rem For Windows
mkdir src
copy ..\src src
swig -c++ -python -py3 xbcf.i
python setup.py sdist bdist_wheel
