echo "MAKING BINARIES"
make

echo ""
echo "DOWNLOADING TRAINING TEXT DATA"
if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi

echo ""
echo "BUILDING VOCABULARY"
./build_dict -train text8 -min-count 200 -save-vocab text8.vocab_200

echo ""
echo "TRAINING EMBEDDINGS"
./cbow -train text8 -words-vocab text8.vocab_200 -output vectors.bin -size 200 -window 2 -iter 5 -threads 8 -optimization ns -negative 5

echo ""
echo "RUN SIMILARITY METER"
./distance vectors.bin
