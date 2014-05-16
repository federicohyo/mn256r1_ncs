echo "COMPILING MAPPER LIBS..."
cd _utils/mapperlib/
sh buil-python-wrapper.sh
echo "COMPILING BIASGEN LIBS..."
cd ../biasgenlib/
sh buil-python-wrapper.sh
cd ..
echo "DONE"

