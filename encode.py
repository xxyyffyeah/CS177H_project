import os, sys
from Bio.Seq import Seq
import numpy as np
import optparse

def encode_codon(seq) :
    codon_list = [  'AAA','AAG','AAC','AAT','AGA','AGG','AGC','AGT',
                    'ACA','ACG','ACC','ACT','ATA','ATG','ATC','ATT',
                    'GAA','GAG','GAC','GAT','GGA','GGG','GGC','GGT',
                    'GCA','GCG','GCC','GCT','GTA','GTG','GTC','GTT',
                    'CAA','CAG','CAC','CAT','CGA','CGG','CGC','CGT',
                    'CCA','CCG','CCC','CCT','CTA','CTG','CTC','CTT',
                    'TAA','TAG','TAC','TAT','TGA','TGG','TGC','TGT',
                    'TCA','TCG','TCC','TCT','TTA','TTG','TTC','TTT']
    df = pd.DataFrame(codon_list)
    df.columns = ['codon',]
    df = pd.get_dummies(df)
    Dict = {}
    for i in codon_list :
        Dict[i] = df['codon_'+i].to_list()
    encode_seq = list()
    for i in range(len(seq)-2) :
        encode_seq.append(Dict[seq[i:i+3]])
    X=np.array(encode_seq)
    pca = PCA(components=20)
    pca.fit(X)
    newX=pca.fit_transform(X)
    return newX

cur_filename = os.path.split(sys.argv[0])[1]

parser = optparse.OptionParser()
parser.add_option("-i", "--fileName", action = "store", type = "string", dest = "fileName",
									help = "fileName")
parser.add_option("-l", "--cutoff_len", action = "store", type = int, dest = "cutoff_len",
									help = "cutoff_len")
parser.add_option("-p", "--DNAType", action = "store", type = "string", dest = "DNAType",
									help = "DNAType, virus or host")

(option,args) = parser.parse_args()
if (option.fileName is None or option.cutoff_len is None or option.DNAType is None) :
    sys.stderr.write("-------------------------------------------------------------------------------\n" +
      cur_filename + " appear ERROR: missing some required command argument" + 
                   "\n-------------------------------------------------------------------------------\n")
    parser.print_help()
    sys.exit(0)
else :
    print("---------------------------------------------------------------------")
    print("Initialize required argument:\n" + "faste file for encode is: " + option.fileName + "\ncutoff_len is           : " + 
    str(option.cutoff_len) + "\nDNAType is              : " + option.DNAType)
    print("---------------------------------------------------------------------")

DNAType = option.DNAType
cutoff_len = option.cutoff_len
cutoff_lenk = option.cutoff_len/1000
fa_fileName = option.fileName
pure_filename = NCBIName = os.path.splitext((os.path.basename(fa_fileName)))[0]
dir_name = os.path.dirname(option.fileName)
out_dir_name = os.path.join(dir_name,"encode")
if not os.path.exists(out_dir_name) :
    os.makedirs(out_dir_name)

with open(fa_fileName) as fa_file :
    DNA_number = 0
    head = ''
    out_fa_file = []
    out_code_file = []
    out_codeR_file = []
    file_th = 1
    for line in fa_file :
        if line[0] == '>' :
            DNA_number += 1
            head = line.strip()
        elif line[0] != '>' :
            DNA_seq = line.strip()
            cut_start = 0
            cut_end = cut_start + cutoff_len
            while cut_end <= len(DNA_seq) :
                contig_name = head.split('/')[-1] + ": " + pure_filename + '/' + DNAType + '_' + str(cutoff_len) + '_' + str(cut_start) + "_to_" + str(cut_end)
                cut_seq = DNA_seq[cut_start:cut_end]
                if cut_seq.count('N')/len(cut_seq) <= 0.3 :
                    out_fa_file.append('>' + contig_name)
                    out_fa_file.append(cut_seq)
                    out_code_file.append(encode_DNA_seq(cut_seq))

                    cut_seq_R = Seq(cut_seq).reverse_complement()
                    out_codeR_file.append(encode_DNA_seq(cut_seq_R))
                cut_start += cutoff_len
                cut_end += cutoff_len

                if len(out_fa_file) !=0 and len(out_fa_file) % 4000000 == 0 :
                    print("the content of file is too big, create new file to store with the file number" + str(file_th))
                    out_code_file_name = pure_filename + ':' + DNAType + '_' + str(cutoff_len) + '_' + str(len(out_code_file)) + "seq" + str(file_th) + "_codefw.npy"
                    out_codeR_file_name = pure_filename + ':' + DNAType + '_' + str(cutoff_len) + '_' + str(len(out_codeR_file)) + "seq" + str(file_th) + "_codebw.npy"
                    out_fa_file_name = pure_filename + ':' + DNAType + '_' + str(cutoff_len) + '_' + str(len(out_codeR_file)) + "seq" + str(file_th) + ".fasta"
                    np.save(os.path.join(out_dir_name,out_code_file_name),np.array(out_code_file))
                    np.save(os.path.join(out_dir_name,out_codeR_file_name),np.array(out_codeR_file))
                    seqnameF = open(os.path.join(out_dir_name, out_fa_file_name), "w")
                    seqnameF.write('\n'.join(out_fa_file) + '\n')
                    seqnameF.close()
                    file_th += 1                    
                    out_fa_file = []
                    out_code_file = []
                    out_codeR_file = []

    out_code_file_name = pure_filename + ':' + DNAType + '_' + str(cutoff_len) + '_' + str(len(out_code_file)) + "seq" + '_' + str(file_th) + "th_codefw.npy"
    out_codeR_file_name = pure_filename + ':' + DNAType + '_' + str(cutoff_len) + '_' + str(len(out_codeR_file)) + "seq" + '_' + str(file_th) + "th_codebw.npy"
    out_fa_file_name = pure_filename + ':' + DNAType + '_' + str(cutoff_len) + '_' + str(len(out_codeR_file)) + "seq" + '_' + str(file_th) + "th.fasta"
    np.save(os.path.join(out_dir_name,out_code_file_name),np.array(out_code_file))
    print("create file: " + out_code_file_name + "......")
    np.save(os.path.join(out_dir_name,out_codeR_file_name),np.array(out_codeR_file))
    print("create file: " + out_codeR_file_name + "......")
    seqnameF = open(os.path.join(out_dir_name, out_fa_file_name), "w")
    seqnameF.write('\n'.join(out_fa_file) + '\n')
    seqnameF.close()
    print("create file: " + out_fa_file_name + "......")