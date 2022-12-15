All the usage is similar to DeepVirFinder, you just should change some file directory.

command for encode.py
python encode.py -i "Your input fasta file path + file name" -l "the cutoff length" -p "virus or host"

command for train.py
python train.pu -i "Your input train encode file path + file name" -j "Your input validation encode file path + file name" + -o "the output directory"\
                -f "filter length" -n "filter number" -d "dense number" -e "epoch time" -w "Y or N for weather train"

command for dvf.py
python dvf.py -i "Your input fasta file path + file name" -m "your model directory" -o "the output directory" -l "cutoff length" -c "core number"

The default model is DeepvirFinder based on base encode, if you want to try codon based encode, comment out the line between 22 and 36 on encode.py and use the line between 9 and 20, and 37 and 46.
Besides, change the channel_num from 4 to 64

All NCBI accession number can be acquired from our supplementary txt file. And we have provided covid-19 fasta file in the path ./COVID-19