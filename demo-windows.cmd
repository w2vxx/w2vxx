@echo off

if not exist distance.exe (
  echo.
  echo "MAKING BINARIES"
  nmake -f makefile.msvc
)

if not exist text8 (
  echo.
  echo DOWNLOADING TRAINING TEXT DATA 
  echo   please wait...
  cscript //Nologo helpers/demo.cmd.wget.js http://mattmahoney.net/dc/text8.zip text8.zip
  cscript //Nologo helpers/demo.cmd.unzip.vbs %cd% %cd%\text8.zip
)

echo.
echo BUILDING VOCABULARY
build_dict -train text8 -min-count 200 -save-vocab text8.vocab_200

echo.
echo TRAINING EMBEDDINGS
cbow -train text8 -words-vocab text8.vocab_200 -output vectors.bin -size 200 -window 2 -iter 5 -threads 8 -optimization ns -negative 5

echo.
echo RUN SIMILARITY METER
distance vectors.bin
